"""Interactive demo of the lunar plate solver (local web app).

    python demo/app.py            # then open http://127.0.0.1:7860

Tab 1 solves an uploaded image (or an example).  Tab 2 renders a view from a camera that
you place with sliders, solves it, and compares the solution with the truth.  Tab 3 solves a
rigid rig of three cameras (simulated from a spacecraft orbit), with a 3-D view.
"""
import math
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lunarsolve.__main__ import make_overlay  # noqa: E402
from lunarsolve.config import MOON_RADIUS_KM as R  # noqa: E402
from lunarsolve.detect import to_gray_float  # noqa: E402
from lunarsolve.geometry import (enu_basis, gnomonic_fwd, nadir_camera, pixel_rays,  # noqa: E402
                                 ray_sphere, vec_to_latlon)
from lunarsolve.solve import Solver  # noqa: E402

EXAMPLES = ROOT / "demo" / "examples"

# ----------------------------------------------------------------------------- lazy state

_state = {}


CATALOGS = ["Robbins (1.3 M craters >= 1 km)", "Fine (2.4 M craters >= 0.6 km, Robbins + CraterNet)"]


def solver(kind=None):
    """kind: an entry of CATALOGS.  The fine catalog is loaded on first use (about 4 GB of RAM)."""
    fine = kind is not None and kind.startswith("Fine")
    key = "solver_fine" if fine else "solver"
    if key not in _state:
        if fine:
            from lunarsolve.catalog import Catalog
            from lunarsolve.config import FINE_CATALOG_NPZ, FINE_INDEX_PATH
            from lunarsolve.index import QuadIndex
            model = _state["solver"].model if "solver" in _state else None
            _state[key] = Solver(catalog=Catalog(FINE_CATALOG_NPZ), index=QuadIndex(FINE_INDEX_PATH), model=model)
        else:
            _state[key] = Solver()
        _state[key].model                 # load CraterNet now, not on the first request
    return _state[key]


def wac():
    if "wac" not in _state:
        from lunarsolve.terrain import WacMosaic
        _state["wac"] = WacMosaic()
    return _state["wac"]


def dem_albedo():
    if "dem" not in _state:
        from lunarsolve.terrain import GpuAlbedo, GpuDem
        _state["dem"] = (GpuDem(), GpuAlbedo())
    return _state["dem"]


def basemap():
    """Global WAC map (RGB, 1440x720), longitude -180 at the left edge."""
    if "basemap" not in _state:
        m = np.asarray(wac().levels[-1], dtype=np.uint8)
        m = cv2.resize(m, (1440, 720), interpolation=cv2.INTER_AREA)
        m = cv2.equalizeHist(m)
        _state["basemap"] = cv2.cvtColor(m, cv2.COLOR_GRAY2RGB)
    return _state["basemap"]


# ----------------------------------------------------------------------------- geometry helpers

def border_pixels(W, H, n=40):
    t = np.linspace(0, 1, n)
    top = np.column_stack([t * (W - 1), np.zeros(n)])
    right = np.column_stack([np.full(n, W - 1.0), t * (H - 1)])
    bottom = np.column_stack([(1 - t) * (W - 1), np.full(n, H - 1.0)])
    left = np.column_stack([np.zeros(n), (1 - t) * (H - 1)])
    return np.vstack([top, right, bottom, left])


def footprint_of_solution(sol):
    H, W = sol.shape
    b = border_pixels(W, H)
    lat, lon = sol.pixel_to_latlon(b[:, 0], b[:, 1])
    ok = np.isfinite(lat)
    return lat[ok], lon[ok]


def footprint_of_camera(C, Rwc, f, W, H):
    b = border_pixels(W, H)
    d = pixel_rays(b[:, 0], b[:, 1], C, Rwc, f, (W - 1) / 2, (H - 1) / 2)
    p, hit = ray_sphere(C, d)
    lat, lon = vec_to_latlon(p[hit])
    return lat, lon


# ----------------------------------------------------------------------------- map drawing

def draw_global(footprints, cameras):
    """footprints: list of (lat, lon, rgb); cameras: list of (lat, lon, rgb)."""
    m = basemap().copy()
    Hm, Wm = m.shape[:2]

    def xy(lat, lon):
        return ((np.asarray(lon) + 180) % 360) / 360 * Wm, (90 - np.asarray(lat)) / 180 * Hm

    for lat, lon, col in footprints:
        if len(lat) < 3:
            continue
        x, y = xy(lat, lon)
        x = x[0] + ((x - x[0] + Wm / 2) % Wm - Wm / 2)          # unwrap across the seam
        pts = np.column_stack([x, y]).round().astype(np.int32)
        for shift in (-Wm, 0, Wm):
            cv2.polylines(m, [pts + [shift, 0]], True, col, 2, cv2.LINE_AA)
    for lat, lon, col in cameras:
        x, y = xy(lat, lon)
        c = (int(x), int(y))
        cv2.circle(m, c, 16, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.circle(m, c, 16, col, 3, cv2.LINE_AA)
        cv2.drawMarker(m, c, col, cv2.MARKER_CROSS, 44, 2, cv2.LINE_AA)
    for lon in range(-180, 181, 30):
        x = int((lon + 180) / 360 * (Wm - 1))
        cv2.line(m, (x, 0), (x, Hm - 1), (90, 90, 120), 1)
        cv2.putText(m, f"{lon % 360}", (x + 3, Hm - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 230), 1)
    for lat in range(-60, 61, 30):
        y = int((90 - lat) / 180 * (Hm - 1))
        cv2.line(m, (0, y), (Wm - 1, y), (90, 90, 120), 1)
        cv2.putText(m, f"{lat:+d}", (4, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 230), 1)
    return m


def draw_local(center, footprints, cameras, size=640):
    """Local orthographic-like map (gnomonic, north up) from the WAC mosaic around a ground point."""
    lat0, lon0 = center
    ext = 1.0
    for lat, lon, _ in footprints:
        if len(lat):
            x, y, _ = gnomonic_fwd(lat, lon, lat0, lon0)
            ext = max(ext, float(np.max(np.hypot(x, y))))
    for lat, lon, _ in cameras:
        x, y, ok = gnomonic_fwd(np.array([lat]), np.array([lon]), lat0, lon0)
        if ok[0]:
            ext = max(ext, float(np.hypot(x[0], y[0])))
    ext = min(ext * 1.35, 1500.0)
    km_px = 2 * ext / size
    jj, ii = np.mgrid[0:size, 0:size].astype(np.float64)
    xk = (ii - size / 2) * km_px
    yk = (size / 2 - jj) * km_px
    e, n, u = enu_basis(lat0, lon0)
    v = u * R + e * xk[..., None] + n * yk[..., None]
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(v[..., 2]))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
    img = wac().sample(lat, lon, km_px)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    m = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    def to_px(la, lo):
        x, y, ok = gnomonic_fwd(np.asarray(la), np.asarray(lo), lat0, lon0)
        return np.column_stack([x / km_px + size / 2, size / 2 - y / km_px]), ok

    for la, lo, col in footprints:
        if len(la) < 3:
            continue
        p, ok = to_px(la, lo)
        cv2.polylines(m, [p[ok].round().astype(np.int32)], True, col, 2, cv2.LINE_AA)
    for la, lo, col in cameras:
        p, ok = to_px([la], [lo])
        if ok[0]:
            c = tuple(int(t) for t in p[0].round())
            cv2.drawMarker(m, c, col, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    # scale bar
    bar_km = 10 ** math.floor(math.log10(ext / 2))
    bar_px = int(bar_km / km_px)
    cv2.line(m, (20, size - 24), (20 + bar_px, size - 24), (255, 255, 255), 3)
    cv2.putText(m, f"{bar_km:g} km", (20, size - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(m, "N", (size - 26, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return m


# ----------------------------------------------------------------------------- formatting

def solution_markdown(sol, s, extra=""):
    st = s.stats
    lines = [f"**Detections:** {st.get('n_det', 0)}"]
    if "limb_tilt" in st:
        lines.append(f"**Limb in view:** tilt {st['limb_tilt']:.1f}°, limb altitude "
                     f"{st['limb_altitude_km']:.1f} km, fit RMS {st['limb_rms_px']:.2f} px → solved on a "
                     f"nadir-rectified view")
    if sol is None:
        lines.append("### No solution\nNo hypothesis passed the statistical test. Try a longer time budget, "
                     "give the field of view, or use an image with more visible craters.")
        return "\n\n".join(lines) + extra
    alt = f"{sol.altitude_km:.1f} ± {sol.altitude_sigma_km:.1f} km"
    rows = [
        ("Camera latitude, longitude", f"{sol.camera_lat:+.4f}°, {sol.camera_lon:.4f}°"),
        ("Altitude", alt + (" (focal length estimated)" if sol.focal_estimated else "")),
        ("Image centre", f"{sol.center_lat:+.4f}°, {sol.center_lon:.4f}°"),
        ("Ground sample distance", f"{sol.gsd_km * 1000:.0f} m/px"),
        ("Heading of image up", f"{sol.heading_deg:.1f}°"),
        ("Off-nadir angle", f"{sol.off_nadir_deg:.1f}°"),
        ("Focal length, FOV", f"{sol.focal_px:.0f} px, {sol.fov_deg:.2f}°"),
        ("Mirrored", str(sol.mirrored)),
        ("Matched craters, RMS", f"{sol.n_matches}, {sol.rms_px:.2f} px"),
        ("Log-odds", f"{sol.log_odds:.0f}"),
        ("Method", sol.method),
        ("Solve time", f"{sol.solve_time_s:.2f} s"),
    ]
    table = "| | |\n|---|---|\n" + "\n".join(f"| {a} | {b} |" for a, b in rows)
    lines.append("### Solution\n" + table)
    if sol.focal_estimated and sol.altitude_sigma_km > 0.2 * sol.altitude_km:
        lines.append("*The altitude is poorly constrained without the focal length (narrow, near-nadir view). "
                     "Give the field of view.*")
    return "\n\n".join(lines) + extra


def to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ----------------------------------------------------------------------------- tab 1: solve

def gazetteer():
    if "gaz" not in _state:
        from lunarsolve.names import Gazetteer
        _state["gaz"] = Gazetteer()
    return _state["gaz"]


def overlay_image(img, dets, sol, labels):
    """Overlay (RGB) with detections, matches, graticule and optional IAU names; and the named-feature list."""
    out = make_overlay(img, dets, sol)
    feats = []
    if sol is not None and labels:
        from lunarsolve.names import draw_labels
        feats = gazetteer().in_view(sol)
        draw_labels(out, sol, feats)
    return to_rgb(out), feats


def features_markdown(feats):
    if not feats:
        return ""
    rows = [f"| {f['name']} | {f['type'].split(',')[0]} | {f['diam_km']:.1f} km | {f['lat']:+.2f}°, {f['lon']:.2f}° |"
            for f in feats[:25]]
    more = f"\n\n*… and {len(feats) - 25} more.*" if len(feats) > 25 else ""
    return ("\n\n### Named features in view\n| Name | Type | Size | Centre |\n|---|---|---|---|\n"
            + "\n".join(rows) + more)


def click_info(sol, evt: gr.SelectData):
    """Latitude, longitude, elevation and nearest named feature of the clicked pixel."""
    if sol is None:
        return "Solve an image first, then click on it."
    from lunarsolve.names import pixel_info
    u, v = evt.index
    info = pixel_info(sol, u, v, gazetteer())
    if info["lat"] is None:
        return f"Pixel ({u}, {v}) does not hit the Moon (space)."
    where = "inside" if info["inside"] else f"{info['nearest_km']:.1f} km outside the edge of"
    return (f"**Pixel ({u}, {v})** → latitude **{info['lat']:+.4f}°**, longitude **{info['lon']:.4f}°**, "
            f"LOLA elevation {info['elevation_km']:+.2f} km. Nearest named feature: {where} "
            f"**{info['nearest']}** ({info['nearest_type'].split(',')[0]}).")


def solve_upload(path, fov, parity, max_time, labels, catalog=None):
    if not path:
        return None, None, None, "Upload an image or choose an example.", None
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None, None, None, "Cannot read this file.", None
    img = to_gray_float(raw)
    s = solver(catalog)
    sol = s.solve(img=img, fov_deg=fov if fov and fov > 0 else None, parity=parity, max_time=float(max_time))
    dets = sol.dets if sol is not None else s.detect(img)
    overlay, feats = overlay_image(img, dets, sol, labels)
    if sol is None:
        return overlay, None, None, solution_markdown(None, s), None
    fl, fo = footprint_of_solution(sol)
    red = (255, 60, 60)
    g = draw_global([(fl, fo, red)], [(sol.camera_lat, sol.camera_lon, (60, 220, 255))])
    loc = draw_local((sol.center_lat, sol.center_lon), [(fl, fo, red)],
                     [(sol.camera_lat, sol.camera_lon, (60, 220, 255))])
    return overlay, loc, g, solution_markdown(sol, s, features_markdown(feats)), sol


# ----------------------------------------------------------------------------- tab 2: simulate

def simulate(lat, lon, alt, fov, tilt, tilt_az, heading, source, sun_el, sun_az, size, give_fov, max_time, seed,
             labels=True):
    size = int(size)
    f = (size / 2) / math.tan(math.radians(fov) / 2)
    C, Rwc = nadir_camera(lat, lon % 360, alt, heading, tilt, tilt_az)
    view = dict(C=C, Rwc=Rwc, f=f, W=size, H=size)
    rng = np.random.default_rng(int(seed))
    t0 = time.time()
    if source == "LROC WAC mosaic (fixed lighting)":
        from lunarsolve.simulate import render_wac
        img = render_wac(wac(), view)
    else:
        from lunarsolve.simulate import render_dem
        dem, alb = dem_albedo()
        img, _, _ = render_dem(dem, view, rng, sun_el_deg=sun_el, sun_az_deg=sun_az, albedo=alb)
    t_render = time.time() - t0
    s = solver()
    sol = s.solve(img=img, fov_deg=fov if give_fov else None, max_time=float(max_time))
    dets = sol.dets if sol is not None else s.detect(img)
    overlay, feats = overlay_image(img, dets, sol, labels)
    tl, to_ = footprint_of_camera(C, Rwc, f, size, size)
    tlat, tlon = vec_to_latlon(C)
    blue, red = (80, 160, 255), (255, 60, 60)
    fps, cams = [], []
    extra = ""
    if sol is not None:
        sl, so = footprint_of_solution(sol)
        fps.append((sl, so, red))
        cams.append((sol.camera_lat, sol.camera_lon, red))
        from lunarsolve.geometry import angular_dist_deg
        d_pos = angular_dist_deg(sol.camera_lat, sol.camera_lon, float(tlat), float(tlon)) * math.pi / 180 * R
        extra = ("\n\n### Comparison with the truth\n| | truth | solved |\n|---|---|---|\n"
                 f"| Camera | {float(tlat):+.4f}°, {float(tlon):.4f}° | {sol.camera_lat:+.4f}°, {sol.camera_lon:.4f}° "
                 f"({d_pos:.2f} km) |\n"
                 f"| Altitude | {alt:.1f} km | {sol.altitude_km:.1f} km ({100 * (sol.altitude_km - alt) / alt:+.2f}%) |\n"
                 f"| FOV | {fov:.2f}° | {sol.fov_deg:.2f}° |\n"
                 f"| Off-nadir | {tilt:.1f}° | {sol.off_nadir_deg:.1f}° |\n")
    else:
        extra = f"\n\n*Truth: camera {float(tlat):+.4f}°, {float(tlon):.4f}°, altitude {alt:.1f} km.*"
    fps.append((tl, to_, blue))                      # truth last, so that it is drawn on top
    cams.append((float(tlat), float(tlon), blue))
    extra += f"\n\n*Render time {t_render:.1f} s. Blue: truth. Red: solution.*"
    if source != "LROC WAC mosaic (fixed lighting)" and alt / f < 0.45:
        extra += ("\n\n*Note: the image scale is finer than the LOLA model's 473 m/px, so this view looks "
                  "smooth. Use a higher altitude or a wider field of view with this renderer.*")
    center = (sol.center_lat, sol.center_lon) if sol is not None else (float(tlat), float(tlon))
    try:
        loc = draw_local(center, fps, cams)
    except Exception:  # noqa: BLE001  (for example a view that misses the Moon)
        loc = None
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), overlay, loc, draw_global(fps, cams), \
        solution_markdown(sol, s, extra + features_markdown(feats)), sol


def random_camera(seed):
    rng = np.random.default_rng()
    lat = math.degrees(math.asin(rng.uniform(-0.97, 0.97)))
    alt = math.exp(rng.uniform(math.log(60), math.log(1200)))
    fov = math.exp(rng.uniform(math.log(8), math.log(45)))
    return (round(lat, 2), round(rng.uniform(0, 360), 2), round(alt, 1), round(fov, 1),
            round(rng.uniform(0, 15), 1), round(rng.uniform(0, 360)), round(rng.uniform(0, 360)),
            round(rng.uniform(5, 80), 1), round(rng.uniform(0, 360)), int(rng.integers(0, 1_000_000)))


# ----------------------------------------------------------------------------- tab 3: multi-camera rig

N_RIG = 3
RIG_COLOURS = [(255, 60, 60), (255, 170, 40), (230, 80, 230)]       # solved footprints, camera 0, 1, 2


def rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mounting(pitch, yaw, roll):
    """Rig -> camera rotation from mounting angles (camera axes: x right, y down, z boresight).

    Positive pitch turns the boresight towards rig image up (forward along the heading), positive yaw
    towards rig image right, roll turns the image about the boresight.
    R_cr = Rz(roll) Ry(-yaw) Rx(-pitch).
    """
    return rot("z", roll) @ rot("y", -yaw) @ rot("x", -pitch)


# Spacecraft presets for the simulated rig: field of view, pitch of the fore/aft cameras, orbit.
# TMC-2 and the Kaguya TC are pushbroom cameras; the demo renders frame images with the same swath.
RIG_PRESETS = {
    "Generic rig: nadir + fore/aft 25°, 20° FOV, 110 km polar orbit":
        dict(fov=20.0, sep=25.0, alt=110, inc=90.0),
    "Chandrayaan-2 TMC-2-like: fore/nadir/aft 25°, 11.4° FOV (20 km swath), 100 km polar orbit":
        dict(fov=11.4, sep=25.0, alt=100, inc=90.0),
    "Kaguya TC-like + nadir: fore/aft 15°, 19.9° FOV (35 km swath), 100 km polar orbit":
        dict(fov=19.9, sep=15.0, alt=100, inc=90.0),
    "LRO-like orbit (50 km polar) with generic 30° cameras":
        dict(fov=30.0, sep=25.0, alt=50, inc=90.0),
    "Wide navigation rig: nadir + fore/aft 35°, 45° FOV, 250 km":
        dict(fov=45.0, sep=35.0, alt=250, inc=90.0),
    "Low-inclination orbit, generic rig, 300 km":
        dict(fov=30.0, sep=25.0, alt=300, inc=10.0),
}


def apply_preset(name):
    p = RIG_PRESETS.get(name)
    if p is None:
        return gr.update(), gr.update(), gr.update(), gr.update()
    return p["fov"], p["sep"], p["alt"], p["inc"]


def ground_track_heading(lat, inc, ascending=True):
    """Direction of flight (degrees clockwise from north) of a circular orbit at a given latitude.

    sin(A) = cos(i) / cos(lat).  Returns (heading, reachable).  The orbit does not reach latitudes
    above its inclination (or above 180° - i for a retrograde orbit); then the heading at the
    highest latitude is used.
    """
    c = math.cos(math.radians(inc)) / max(math.cos(math.radians(lat)), 1e-9)
    a = math.degrees(math.asin(max(-1.0, min(1.0, c))))
    return (a if ascending else 180.0 - a) % 360, abs(c) <= 1.0


def _enu_frame(lat0, lon0):
    """Function: body-frame points (km) -> local east/north/up (km) at a ground point."""
    e, n, u = enu_basis(lat0, lon0)
    O = u * R
    M = np.stack([e, n, u])
    return lambda P: (np.asarray(P, float) - O) @ M.T, M


def _camera_lines(C, Rwc, f, W, H, to_local, lift=0.3):
    """Footprint polyline and the 4 corner rays of one camera, in local coordinates."""
    b = border_pixels(W, H, n=30)
    d = pixel_rays(b[:, 0], b[:, 1], C, Rwc, f, (W - 1) / 2, (H - 1) / 2)
    p, hit = ray_sphere(C, d)
    fp = to_local(p[hit]) + [0, 0, lift] if hit.any() else np.zeros((0, 3))
    corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float)
    dc = pixel_rays(corners[:, 0], corners[:, 1], C, Rwc, f, (W - 1) / 2, (H - 1) / 2)
    pc, hc = ray_sphere(C, dc)
    rays = [np.vstack([to_local(C), to_local(q) + [0, 0, lift]]) for q, h in zip(pc, hc) if h]
    bd = pixel_rays(np.array([(W - 1) / 2]), np.array([(H - 1) / 2]), C, Rwc, f, (W - 1) / 2, (H - 1) / 2)
    pb, hb = ray_sphere(C, bd)
    bore = np.vstack([to_local(C), to_local(pb[0]) + [0, 0, lift]]) if hb[0] else None
    return fp, rays, bore


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def rig_figure(imgs, cams, rig, truth):
    """3-D scene of the rig: terrain with the WAC texture, spacecraft, camera frusta and footprints."""
    import plotly.graph_objects as go
    from lunarsolve.config import LDEM_PPD
    from lunarsolve.terrain import load_ldem
    C_t, Rwr_t = truth
    lat0, lon0 = vec_to_latlon(C_t)
    to_local, _ = _enu_frame(float(lat0), float(lon0))
    names = ("nadir", "fore", "aft")
    cam_lines = []                                     # (label, colour, footprint, rays, boresight, width)
    for k, (img, cam) in enumerate(zip(imgs, cams)):
        H, W = img.shape
        cam_lines.append((f"camera {k} ({names[k]}) truth", "#50a0ff",
                          *_camera_lines(C_t, cam.R_cr @ Rwr_t, cam.focal_px, W, H, to_local), 2))
    if rig is not None:
        for k, sol in enumerate(rig.cameras):
            H, W = sol.shape
            cam_lines.append((f"camera {k} ({names[k]}) solved", _hex(RIG_COLOURS[k]),
                              *_camera_lines(sol.C, sol.Rwc, sol.focal_px, W, H, to_local), 4))
    pts = np.vstack([fp for _, _, fp, *_ in cam_lines if len(fp)] + [np.zeros((1, 3))])
    ext = float(np.max(np.abs(pts[:, :2]))) * 1.15 + 2
    # terrain patch
    n = 140
    xs = np.linspace(-ext, ext, n)
    X, Y = np.meshgrid(xs, xs)
    e, nn, u = enu_basis(float(lat0), float(lon0))
    v = u * R + e * X[..., None] + nn * Y[..., None]
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
    dem = load_ldem()
    r = np.clip(((90 - lat) * LDEM_PPD).astype(int), 0, dem.shape[0] - 1)
    c = ((lon * LDEM_PPD).astype(int)) % dem.shape[1]
    h = dem[r, c].astype(float) * 0.0005
    P = v * (R + h)[..., None]
    L = to_local(P.reshape(-1, 3)).reshape(n, n, 3)
    tex = wac().sample(lat, lon, 2 * ext / n)
    tex = (tex - np.percentile(tex, 1)) / max(np.percentile(tex, 99) - np.percentile(tex, 1), 1e-6)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=L[..., 0], y=L[..., 1], z=L[..., 2], surfacecolor=np.clip(tex, 0, 1),
                             colorscale="gray", cmin=0, cmax=1, showscale=False, name="LROC WAC on LOLA",
                             hoverinfo="skip", lighting=dict(ambient=0.9, diffuse=0.2, specular=0.0)))
    for label, col, fp, rays, bore, w in cam_lines:
        if len(fp):
            fpc = np.vstack([fp, fp[:1]])
            fig.add_trace(go.Scatter3d(x=fpc[:, 0], y=fpc[:, 1], z=fpc[:, 2], mode="lines", name=label,
                                       legendgroup=label, line=dict(color=col, width=w)))
        for ray in rays:
            fig.add_trace(go.Scatter3d(x=ray[:, 0], y=ray[:, 1], z=ray[:, 2], mode="lines", legendgroup=label,
                                       showlegend=False, line=dict(color=col, width=max(1, w - 2)),
                                       hoverinfo="skip"))
        if bore is not None:
            fig.add_trace(go.Scatter3d(x=bore[:, 0], y=bore[:, 1], z=bore[:, 2], mode="lines", legendgroup=label,
                                       showlegend=False, line=dict(color=col, width=1, dash="dash"),
                                       hoverinfo="skip"))
    ct = to_local(C_t)
    fig.add_trace(go.Scatter3d(x=[ct[0]], y=[ct[1]], z=[ct[2]], mode="markers", name="spacecraft (truth)",
                               marker=dict(size=7, color="#50a0ff", symbol="diamond")))
    if rig is not None:
        cr = to_local(rig.C)
        fig.add_trace(go.Scatter3d(x=[cr[0]], y=[cr[1]], z=[cr[2]], mode="markers", name="spacecraft (rig solution)",
                                   marker=dict(size=5, color="#ff3c3c")))
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=30, b=60), paper_bgcolor="#111", font=dict(color="#ddd"),
        title=dict(text="Rig geometry (km; east, north, up from the sub-spacecraft point)", font=dict(size=13)),
        template="plotly_dark", legend=dict(orientation="h", yanchor="top", y=-0.02, font=dict(size=10)),
        scene=dict(aspectmode="data", xaxis_title="east (km)", yaxis_title="north (km)", zaxis_title="up (km)",
                   bgcolor="#111", camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))))
    return fig


def error_figure(rig, singles, truth):
    """3-D scatter of the solved positions around the truth (metres), with the rig 1-sigma ellipsoid."""
    import plotly.graph_objects as go
    C_t, _ = truth
    lat0, lon0 = vec_to_latlon(C_t)
    _, M = _enu_frame(float(lat0), float(lon0))

    def rel(C):
        return M @ (np.asarray(C) - C_t) * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0], mode="markers", name="truth",
                               marker=dict(size=7, color="#50a0ff", symbol="diamond")))
    names = ("nadir", "fore", "aft")
    for k, sg in enumerate(singles or []):
        if sg is None:
            continue
        d = rel(sg.C)
        col = _hex(RIG_COLOURS[k])
        fig.add_trace(go.Scatter3d(x=[d[0]], y=[d[1]], z=[d[2]], mode="markers",
                                   name=f"camera {k} ({names[k]}) alone: {np.linalg.norm(d):.0f} m",
                                   marker=dict(size=5, color=col, symbol="square")))
        sa = sg.altitude_sigma_km * 1000
        fig.add_trace(go.Scatter3d(x=[d[0], d[0]], y=[d[1], d[1]], z=[d[2] - sa, d[2] + sa], mode="lines",
                                   showlegend=False, line=dict(color=col, width=3), hoverinfo="skip"))
    if rig is not None:
        d = rel(rig.C)
        fig.add_trace(go.Scatter3d(x=[d[0]], y=[d[1]], z=[d[2]], mode="markers",
                                   name=f"rig (joint): {np.linalg.norm(d):.0f} m",
                                   marker=dict(size=6, color="#ff3c3c")))
        if rig.cov_C is not None:
            cov = M @ rig.cov_C @ M.T * 1e6
            w, V = np.linalg.eigh(cov)
            a, b = np.meshgrid(np.linspace(0, 2 * np.pi, 30), np.linspace(0, np.pi, 16))
            sph = np.stack([np.cos(a) * np.sin(b), np.sin(a) * np.sin(b), np.cos(b)], -1)
            ell = sph @ (V * np.sqrt(np.maximum(w, 0))).T + d
            fig.add_trace(go.Surface(x=ell[..., 0], y=ell[..., 1], z=ell[..., 2], opacity=0.3, showscale=False,
                                     colorscale=[[0, "#ff3c3c"], [1, "#ff3c3c"]], name="rig 1-sigma",
                                     hoverinfo="skip"))
    fig.update_layout(
        height=680, margin=dict(l=0, r=0, t=30, b=60), paper_bgcolor="#111", font=dict(color="#ddd"),
        title=dict(text="Solved positions around the truth (m). Red: rig with its 1-sigma ellipsoid. "
                        "Squares: single cameras with their altitude 1-sigma", font=dict(size=12)),
        template="plotly_dark", legend=dict(orientation="h", yanchor="top", y=-0.02, font=dict(size=10)),
        scene=dict(aspectmode="data", xaxis_title="east (m)", yaxis_title="north (m)", zaxis_title="up (m)",
                   bgcolor="#111"))
    return fig


def rig_results(imgs, cams, rig, singles, labels, truth=None, t_render=None, head=""):
    """Shared output of the rig tab: overlays, maps, markdown and per-camera solutions."""
    s = solver()
    sols = rig.cameras if rig is not None else [None] * len(imgs)
    overlays, feats_all = [], []
    for img, sol in zip(imgs, sols):
        ovl, feats = overlay_image(img, sol.dets if sol is not None else s.detect(img), sol, labels)
        overlays.append(ovl)
        feats_all += [f for f in feats if f["name"] not in {g["name"] for g in feats_all}]
    fps, marks = [], []
    blue = (80, 160, 255)
    if rig is not None:
        for k, sol in enumerate(sols):
            sl, so = footprint_of_solution(sol)
            fps.append((sl, so, RIG_COLOURS[k % len(RIG_COLOURS)]))
        marks.append((rig.lat, rig.lon, (255, 60, 60)))
    if truth is not None:
        C, Rwr = truth
        for img, cam in zip(imgs, cams):
            H, W = img.shape
            tl, to_ = footprint_of_camera(C, cam.R_cr @ Rwr, cam.focal_px, W, H)
            fps.append((tl, to_, blue))
        tlat, tlon = vec_to_latlon(C)
        marks.append((float(tlat), float(tlon), blue))

    lines = [head] if head else []
    if rig is None:
        lines.append("### No rig solution\nNo camera solved alone and the merged view did not solve. Try a "
                     "longer time budget, wider fields of view, or more cameras pointed at cratered ground.")
    else:
        seed = (f"camera {rig.seed_camera}" if rig.seed_camera >= 0
                else "none (all detections merged into one virtual view)")
        rows = [("Rig latitude, longitude", f"{rig.lat:+.5f}°, {rig.lon:.5f}°"),
                ("Altitude", f"{rig.altitude_km:.3f} km ± {rig.sigma_altitude_km * 1000:.0f} m"),
                ("Horizontal 1-sigma", f"{rig.sigma_horizontal_km * 1000:.0f} m"),
                ("Matched craters per camera", ", ".join(str(n) for n in rig.matches_per_camera)),
                ("RMS residual", f"{rig.rms_px:.2f} px"),
                ("Seed camera (first blind solve)", seed),
                ("Solve time", f"{rig.solve_time_s:.1f} s")]
        lines.append("### Rig solution (joint fit of all cameras)\n| | |\n|---|---|\n"
                     + "\n".join(f"| {a} | {b} |" for a, b in rows))

    if truth is not None:
        C, Rwr = truth
        tlat, tlon = vec_to_latlon(C)
        alt = float(np.linalg.norm(C) - R)
        tab = ["| Method | Position | Altitude | 3-D error | Altitude error |", "|---|---|---|---|---|",
               f"| Truth | {float(tlat):+.5f}°, {float(tlon):.5f}° | {alt:.3f} km | | |"]

        def row(name, c_lat, c_lon, c_alt, c_C):
            return (f"| {name} | {c_lat:+.5f}°, {c_lon:.5f}° | {c_alt:.3f} km | "
                    f"{np.linalg.norm(c_C - C) * 1000:.0f} m | {(c_alt - alt) * 1000:+.0f} m |")
        if rig is not None:
            tab.append(row("**Rig (joint)**", rig.lat, rig.lon, rig.altitude_km, rig.C))
        for k, sg in enumerate(singles or []):
            tab.append(f"| Camera {k} alone | no solution | | | |" if sg is None else
                       row(f"Camera {k} alone", sg.camera_lat, sg.camera_lon, sg.altitude_km, sg.C))
        lines.append("### Comparison with the truth\n" + "\n".join(tab))
    elif singles and rig is not None:
        tab = ["| Method | Position | Altitude | Distance from the rig solution |", "|---|---|---|---|"]
        for k, sg in enumerate(singles):
            tab.append(f"| Camera {k} alone | no solution | | |" if sg is None else
                       f"| Camera {k} alone | {sg.camera_lat:+.5f}°, {sg.camera_lon:.5f}° | {sg.altitude_km:.3f} km "
                       f"| {np.linalg.norm(sg.C - rig.C) * 1000:.0f} m |")
        lines.append("### Single-camera solves\n" + "\n".join(tab))
    note = "*Map colours: blue = truth; red, orange, magenta = solved footprints of cameras 0, 1, 2.*"
    if t_render is not None:
        note += f" *Render time {t_render:.1f} s.*"
    lines.append(note)
    md = "\n\n".join(lines) + features_markdown(feats_all)

    loc = glob = None
    if fps:
        if rig is not None:
            center = (rig.lat, rig.lon)
        else:
            center = (float(vec_to_latlon(truth[0])[0]), float(vec_to_latlon(truth[0])[1]))
        try:
            loc = draw_local(center, fps, marks)
        except Exception:  # noqa: BLE001
            loc = None
        glob = draw_global(fps, marks)
    fig3d = ferr = None
    if truth is not None:
        try:
            fig3d = rig_figure(imgs, cams, rig, truth)
            ferr = error_figure(rig, singles, truth)
        except Exception as ex:  # noqa: BLE001
            print("3-D figure failed:", ex, flush=True)
    while len(overlays) < N_RIG:
        overlays.append(None)
        sols = list(sols) + [None]
    return (*overlays[:N_RIG], loc, glob, fig3d, ferr, md, *list(sols)[:N_RIG])


def single_solves(imgs, cams, max_time, catalog=None):
    s = solver(catalog)
    return [s.solve(img=img, focal_px=cam.focal_px, principal=cam.principal, parity="normal",
                    max_time=max_time / len(imgs), limb=False) for img, cam in zip(imgs, cams)]


def simulate_rig(lat, lon, alt, fov, sep, inc, direction, tilt, tilt_az, source, sun_el, sun_az, size,
                 max_time, compare, labels, catalog=None):
    from lunarsolve.rig import RigCamera, RigSolver
    size = int(size)
    f = (size / 2) / math.tan(math.radians(fov) / 2)
    pitches = (0.0, float(sep), -float(sep))           # nadir, fore, aft
    cams = [RigCamera(mounting(p, 0, 0), f) for p in pitches]
    heading, reach = ground_track_heading(lat, inc, direction.startswith("asc"))
    C, Rwr = nadir_camera(lat, lon % 360, alt, heading, tilt, tilt_az)
    t0 = time.time()
    imgs = []
    rng = np.random.default_rng(0)
    for cam in cams:
        view = dict(C=C, Rwc=cam.R_cr @ Rwr, f=f, W=size, H=size)
        if source.startswith("LROC"):
            from lunarsolve.simulate import render_wac
            imgs.append(render_wac(wac(), view))
        else:
            from lunarsolve.simulate import render_dem
            dem, alb = dem_albedo()
            imgs.append(render_dem(dem, view, rng, sun_el_deg=sun_el, sun_az_deg=sun_az, albedo=alb)[0])
    t_render = time.time() - t0
    rig = RigSolver(solver(catalog)).solve(imgs, cams, max_time=float(max_time))
    singles = single_solves(imgs, cams, float(max_time), catalog) if compare else None
    swath = 2 * alt * math.tan(math.radians(fov) / 2)
    gsd = swath / size * 1000
    head = (f"**Orbit:** {'ascending' if direction.startswith('asc') else 'descending'} pass, inclination "
            f"{inc:.1f}°, flight direction {heading:.1f}° from north"
            + ("" if reach else f" (*this orbit does not reach latitude {lat:.1f}°; the heading at its highest "
                                f"latitude is used*)")
            + f". **Nadir swath** {swath:.1f} km at {gsd:.0f} m/px.")
    if swath < 25:
        head += (" *A swath below about 25 km holds few catalogued craters. Select the fine catalog or a "
                 "larger image size.*")
    out = rig_results(imgs, cams, rig, singles, labels, truth=(C, Rwr), t_render=t_render, head=head)
    meta = dict(renderer=source, rig=dict(lat=lat, lon=lon % 360, altitude_km=alt, inclination_deg=inc,
                                          pass_direction=direction, heading_deg=heading, tilt_deg=tilt,
                                          tilt_azimuth_deg=tilt_az, C_km=C.tolist(), R_world_to_rig=Rwr.tolist()),
                sun=None if source.startswith("LROC") else dict(elevation_deg=sun_el, azimuth_deg=sun_az),
                mounting_convention="R_cr = Rz(roll) Ry(-yaw) Rx(-pitch); +pitch = forward (image up), "
                                    "+yaw = right; camera axes x right, y down, z boresight")
    meta["cameras"] = [dict(file=f"rig_cam{k}.png", pitch_deg=p, yaw_deg=0.0, roll_deg=0.0, fov_deg=fov,
                            focal_px=f, width=size, height=size) for k, p in enumerate(pitches)]
    return out + (save_rig_set(imgs, meta),)


def save_rig_set(imgs, meta):
    """Zip of the rendered images and a truth.json, for download."""
    import json
    import tempfile
    import zipfile
    d = Path(tempfile.mkdtemp(prefix="lunar_rig_"))
    zp = d / "lunar_rig_views.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for k, img in enumerate(imgs):
            _, buf = cv2.imencode(".png", (np.clip(img, 0, 1) * 255).astype(np.uint8))
            z.writestr(f"rig_cam{k}.png", buf.tobytes())
        z.writestr("truth.json", json.dumps(meta, indent=2))
    return str(zp)


# ----------------------------------------------------------------------------- UI

INTRO = """# Lunar plate solver
Give one image of the lunar surface. The solver finds the craters, matches crater "constellations"
against a catalog of 1.3 million craters, and gives the camera latitude, longitude and altitude.
No position hint and no focal length are necessary.

**Overlay colours:** orange = detected craters, green = craters matched to the catalog,
yellow/magenta dots = latitude/longitude lines, cyan outlines = IAU-named craters, orange diamonds = other
named features. **Click** on a solved image to read the latitude, longitude, elevation and nearest named
feature of that pixel.
"""


def build():
    order = ["wac_view.png", "AS16-M-1657.png", "AS16-M-1347.png", "AS17-M-0748.png"]
    examples = [str(EXAMPLES / n) for n in order if (EXAMPLES / n).exists()]
    with gr.Blocks(title="Lunar plate solver") as demo:
        gr.Markdown(INTRO)
        with gr.Tab("Solve an image"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(type="filepath", label="Lunar image", height=360)
                    fov = gr.Number(value=0, label="Horizontal field of view in degrees (0 = unknown)")
                    parity = gr.Dropdown(["both", "normal", "mirrored"], value="both", label="Image parity")
                    tmax = gr.Slider(5, 60, value=30, step=1, label="Time budget (s)")
                    lab = gr.Checkbox(value=True, label="Label named craters and features (IAU names)")
                    catsel = gr.Dropdown(CATALOGS, value=CATALOGS[0], label="Crater catalog (fine: small footprints)")
                    btn = gr.Button("Solve", variant="primary")
                    gr.Examples(examples=[[e] for e in examples], inputs=[inp],
                                label="Examples (left to right): simulated WAC view, Apollo 16 vertical frame, "
                                      "Apollo 16 oblique frame with the limb, Apollo 17 high-sun frame")
                with gr.Column(scale=2):
                    out_overlay = gr.Image(label="Detections and solution (click a pixel for its coordinates)",
                                           height=520, interactive=False)
                    out_click = gr.Markdown("*Solve an image, then click on it to read the coordinates of any pixel.*")
                    out_md = gr.Markdown()
            with gr.Row():
                out_local = gr.Image(label="Footprint on the LROC WAC mosaic (north up)", height=420)
                out_global = gr.Image(label="Position on the Moon (red: footprint, cyan: camera)", height=420)
            sol_state = gr.State(None)
            btn.click(solve_upload, [inp, fov, parity, tmax, lab, catsel],
                      [out_overlay, out_local, out_global, out_md, sol_state])
            out_overlay.select(click_info, [sol_state], [out_click])

        with gr.Tab("Simulate a camera"):
            gr.Markdown("Place a camera, render the view, and solve it blind. The renderer uses the real WAC "
                        "mosaic draped on LOLA terrain, or the LOLA elevation model with Clementine albedo and "
                        "a sun that you choose.")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        s_lat = gr.Slider(-85, 85, value=-23.5, step=0.01, label="Camera latitude (°)")
                        s_lon = gr.Slider(0, 360, value=345.0, step=0.01, label="Camera longitude (°E)")
                    with gr.Row():
                        s_alt = gr.Slider(20, 2000, value=250, step=1, label="Altitude (km)")
                        s_fov = gr.Slider(2, 90, value=20, step=0.1, label="Field of view (°)")
                    with gr.Row():
                        s_tilt = gr.Slider(0, 45, value=12, step=0.5, label="Tilt from nadir (°)")
                        s_taz = gr.Slider(0, 360, value=200, step=1, label="Tilt azimuth (°)")
                        s_head = gr.Slider(0, 360, value=30, step=1, label="Heading (°)")
                    s_src = gr.Radio(["LROC WAC mosaic (fixed lighting)", "LOLA terrain + Clementine albedo"],
                                     value="LROC WAC mosaic (fixed lighting)", label="Renderer")
                    with gr.Row():
                        s_sel = gr.Slider(1, 89, value=30, step=0.5, label="Sun elevation (°), LOLA only")
                        s_saz = gr.Slider(0, 360, value=90, step=1, label="Sun azimuth (°), LOLA only")
                    with gr.Row():
                        s_size = gr.Dropdown(["512", "768", "1024"], value="1024", label="Image size (px)")
                        s_give = gr.Checkbox(value=False, label="Give the FOV to the solver")
                        s_tmax = gr.Slider(5, 60, value=30, step=1, label="Time budget (s)")
                    s_seed = gr.Number(value=0, label="Noise seed", precision=0)
                    with gr.Row():
                        b_rand = gr.Button("Random camera")
                        b_run = gr.Button("Render and solve", variant="primary")
                with gr.Column(scale=2):
                    with gr.Row():
                        o_img = gr.Image(label="Rendered view", height=420)
                        o_ovl = gr.Image(label="Detections, solution and names (click a pixel)", height=420,
                                         interactive=False)
                    o_click = gr.Markdown("*Click on the solved image to read the coordinates of any pixel.*")
                    o_md = gr.Markdown()
            with gr.Row():
                o_loc = gr.Image(label="Footprints (blue: truth, red: solution)", height=420)
                o_glob = gr.Image(label="Position on the Moon", height=420)
            s_state = gr.State(None)
            sliders = [s_lat, s_lon, s_alt, s_fov, s_tilt, s_taz, s_head, s_sel, s_saz, s_seed]
            b_rand.click(random_camera, [s_seed], sliders)
            b_run.click(simulate,
                        [s_lat, s_lon, s_alt, s_fov, s_tilt, s_taz, s_head, s_src, s_sel, s_saz, s_size, s_give,
                         s_tmax, s_seed],
                        [o_img, o_ovl, o_loc, o_glob, o_md, s_state])
            o_ovl.select(click_info, [s_state], [o_click])

        with gr.Tab("Multi-camera rig"):
            gr.Markdown("A spacecraft carries three cameras on one rigid mount: one looks at nadir, one looks "
                        "forward along the flight direction, one looks back. All three take an image at the same "
                        "instant, and their orientations on the mount are known. The solver solves one camera "
                        "blind, predicts the views of the other cameras, matches their craters, and fits one "
                        "spacecraft pose to all matches together. The wide angle between the views fixes the "
                        "altitude much better than one narrow camera.\n\n"
                        "Select a spacecraft, put it on its orbit, and render the three views. "
                        "**Download** the rendered set (PNG images + truth.json) to test other tools.")
            with gr.Row():
                with gr.Column(scale=1):
                    r_pre = gr.Dropdown(list(RIG_PRESETS) + ["Custom"], value=list(RIG_PRESETS)[0],
                                        label="Spacecraft and camera rig")
                    with gr.Row():
                        r_lat = gr.Slider(-85, 85, value=-9.0, step=0.01, label="Sub-spacecraft latitude (°)")
                        r_lon = gr.Slider(0, 360, value=15.5, step=0.01, label="Sub-spacecraft longitude (°E)")
                    with gr.Row():
                        r_alt = gr.Slider(20, 600, value=110, step=1, label="Orbit altitude (km)")
                        r_inc = gr.Slider(0, 180, value=90, step=0.5, label="Orbit inclination (°)")
                    r_dir = gr.Radio(["ascending (northbound)", "descending (southbound)"],
                                     value="ascending (northbound)", label="Pass direction")
                    with gr.Row():
                        r_fov = gr.Slider(5, 60, value=20, step=0.1, label="Field of view per camera (°)")
                        r_sep = gr.Slider(0, 45, value=25, step=0.5, label="Fore/aft camera pitch (± °)")
                    with gr.Row():
                        r_tilt = gr.Slider(0, 20, value=0, step=0.5, label="Attitude error: tilt (°)")
                        r_taz = gr.Slider(0, 360, value=60, step=1, label="Tilt azimuth (°)")
                    r_src = gr.Radio(["LROC WAC mosaic (fixed lighting)", "LOLA terrain + Clementine albedo"],
                                     value="LROC WAC mosaic (fixed lighting)", label="Renderer")
                    with gr.Row():
                        r_sel = gr.Slider(1, 89, value=30, step=0.5, label="Sun elevation (°), LOLA only")
                        r_saz = gr.Slider(0, 360, value=90, step=1, label="Sun azimuth (°), LOLA only")
                    with gr.Row():
                        r_size = gr.Dropdown(["512", "768", "1024"], value="768", label="Image size (px)")
                        r_cat = gr.Dropdown(CATALOGS, value=CATALOGS[0], label="Crater catalog")
                    r_tmax = gr.Slider(10, 90, value=30, step=1, label="Time budget (s)")
                    r_cmp = gr.Checkbox(value=True, label="Also solve each camera alone (for comparison)")
                    r_lab = gr.Checkbox(value=True, label="Label named craters and features (IAU names)")
                    r_run = gr.Button("Render and solve the rig", variant="primary")
                    r_file = gr.File(label="Rendered set (images + truth.json)", interactive=False)
                with gr.Column(scale=2):
                    with gr.Row():
                        r_ovl = [gr.Image(label=f"Camera {k} ({name}; click a pixel)", height=330,
                                          interactive=False)
                                 for k, name in enumerate(("nadir", "fore", "aft"))]
                    r_click = gr.Markdown("*Solve a rig, then click on any camera image to read the coordinates "
                                          "of a pixel.*")
                    r_md = gr.Markdown()
            r_3d = gr.Plot(label="3-D view of the rig (drag to rotate, scroll to zoom)")
            r_err = gr.Plot(label="Uncertainty: rig compared with single cameras")
            with gr.Row():
                r_loc = gr.Image(label="Footprints on the LROC WAC mosaic (north up)", height=420)
                r_glob = gr.Image(label="Position on the Moon", height=420)
            r_states = [gr.State(None) for _ in range(N_RIG)]
            outs = [*r_ovl, r_loc, r_glob, r_3d, r_err, r_md, *r_states, r_file]
            r_pre.change(apply_preset, [r_pre], [r_fov, r_sep, r_alt, r_inc])
            r_run.click(simulate_rig, [r_lat, r_lon, r_alt, r_fov, r_sep, r_inc, r_dir, r_tilt, r_taz, r_src,
                                       r_sel, r_saz, r_size, r_tmax, r_cmp, r_lab, r_cat], outs)
            for ovl, st in zip(r_ovl, r_states):
                ovl.select(click_info, [st], [r_click])
    return demo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    print("loading the catalog, the quad index and CraterNet ...", flush=True)
    solver()
    build().queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port)
