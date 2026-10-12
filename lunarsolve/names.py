"""IAU lunar nomenclature: named craters and other features, projected into a solved image.

Source: IAU / USGS Gazetteer of Planetary Nomenclature, MOON_nomenclature_center_pts
(https://planetarynames.wr.usgs.gov).  9,087 adopted names with centre, diameter and type.
"""
import struct
import zipfile

import cv2
import numpy as np

from .config import DATA_DIR, MOON_RADIUS_KM as R
from .geometry import latlon_to_vec, project

NAMES_DIR = DATA_DIR / "names"
NAMES_ZIP = NAMES_DIR / "moon_names.zip"
NAMES_NPZ = NAMES_DIR / "moon_names.npz"
NAMES_URL = "https://asc-planetarynames-data.s3.us-west-2.amazonaws.com/MOON_nomenclature_center_pts.zip"

# Feature codes drawn as circles (craters); everything else is drawn as a labelled point.
CRATER_CODES = {"AA", "SF"}           # "Crater, craters" and "Satellite Feature" (lettered craters)


def _read_dbf(raw):
    hlen = struct.unpack("<H", raw[8:10])[0]
    rlen = struct.unpack("<H", raw[10:12])[0]
    n = struct.unpack("<I", raw[4:8])[0]
    fields, pos = [], 32
    while raw[pos] != 0x0D:
        d = raw[pos:pos + 32]
        fields.append((d[:11].split(b"\0")[0].decode(), d[16]))
        pos += 32
    rows = []
    for i in range(n):
        r = raw[hlen + i * rlen: hlen + (i + 1) * rlen]
        o, p = {}, 1
        for name, ln in fields:
            o[name] = r[p:p + ln].decode("utf-8", "replace").strip()
            p += ln
        rows.append(o)
    return rows


def build_names():
    """Download (if needed) and convert the nomenclature to a compact npz."""
    NAMES_DIR.mkdir(parents=True, exist_ok=True)
    if not NAMES_ZIP.exists():
        import requests
        r = requests.get(NAMES_URL, timeout=300)
        r.raise_for_status()
        NAMES_ZIP.write_bytes(r.content)
    with zipfile.ZipFile(NAMES_ZIP) as z:
        dbf = next(n for n in z.namelist() if n.endswith(".dbf"))
        rows = _read_dbf(z.read(dbf))
    np.savez(NAMES_NPZ,
             name=np.array([r["name"] for r in rows]),
             ascii=np.array([r["clean_name"] for r in rows]),
             lat=np.array([float(r["center_lat"]) for r in rows]),
             lon=np.array([float(r["center_lon"]) % 360 for r in rows]),
             diam=np.array([float(r["diameter"] or 0) for r in rows]),
             type=np.array([r["type"] for r in rows]),
             code=np.array([r["code"] for r in rows]))


class Gazetteer:
    def __init__(self):
        if not NAMES_NPZ.exists():
            build_names()
        z = np.load(NAMES_NPZ)
        self.name, self.ascii = z["name"], z["ascii"]
        self.lat, self.lon, self.diam = z["lat"], z["lon"], z["diam"]
        self.type, self.code = z["type"], z["code"]
        self.xyz = latlon_to_vec(self.lat, self.lon)
        self.is_crater = np.isin(self.code, list(CRATER_CODES))

    def in_view(self, sol, min_diam_px=14.0, satellite_min_px=40.0):
        """Named features visible in a solved image.

        Craters are kept when their projected diameter is at least min_diam_px (lettered satellite
        craters need satellite_min_px).  Other features are kept when their centre is in the image.
        Returns a list of dicts sorted by size (largest first).
        """
        H, W = sol.shape
        cx, cy = sol.principal()
        P = self.xyz * R
        uv, z = project(P, sol.C, sol.Rwc, sol.focal_px, cx, cy)
        nrm = self.xyz
        facing = np.sum(nrm * (sol.C - P), axis=1) > 0
        rng = np.linalg.norm(P - sol.C, axis=1)
        d_px = self.diam * sol.focal_px / rng
        if sol.mirrored:
            uv = uv.copy()
            uv[:, 0] = W - 1 - uv[:, 0]
        inside = (z > 0) & facing & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        sat = self.code == "SF"
        big = np.where(sat, d_px >= satellite_min_px, np.where(self.is_crater, d_px >= min_diam_px, True))
        out = []
        for i in np.nonzero(inside & big)[0]:
            out.append(dict(name=str(self.name[i]), ascii=str(self.ascii[i]), type=str(self.type[i]),
                            code=str(self.code[i]), lat=float(self.lat[i]), lon=float(self.lon[i]),
                            diam_km=float(self.diam[i]), u=float(uv[i, 0]), v=float(uv[i, 1]),
                            d_px=float(d_px[i]), crater=bool(self.is_crater[i])))
        out.sort(key=lambda f: -f["diam_km"])
        return out

    def nearest(self, lat, lon, k=1):
        """Named features at or near a point: list of (name, type, distance km, inside).

        If the point is inside named features, those come first (smallest first).  Otherwise the
        features are ordered by the distance to their edge (centre distance minus radius).
        """
        p = latlon_to_vec(lat, lon)
        dist = np.arccos(np.clip(self.xyz @ p, -1, 1)) * R
        edge = dist - self.diam / 2
        inside = np.nonzero(edge <= 0)[0]
        inside = inside[np.argsort(self.diam[inside])]
        rest = np.argsort(np.where(edge > 0, edge, np.inf))
        order = list(inside[:k]) + [i for i in rest[:k] if edge[i] > 0][:max(0, k - len(inside))]
        return [(str(self.name[i]), str(self.type[i]), float(max(edge[i], 0.0)), bool(edge[i] <= 0))
                for i in order[:k]]


def crater_outline(sol, lat, lon, diam_km, n=48):
    """Image polyline of a crater rim (a circle on the sphere), for drawing."""
    from .geometry import enu_basis
    e, nn, u = enu_basis(lat, lon)
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = diam_km / 2
    pts = u * R + (np.cos(a)[:, None] * e + np.sin(a)[:, None] * nn) * r
    H, W = sol.shape
    cx, cy = sol.principal()
    uv, z = project(pts, sol.C, sol.Rwc, sol.focal_px, cx, cy)
    if sol.mirrored:
        uv[:, 0] = W - 1 - uv[:, 0]
    return uv[z > 0]


def draw_labels(bgr, sol, features, max_labels=60):
    """Draw named features onto a BGR image (in place) and return it."""
    H, W = bgr.shape[:2]
    scale = max(0.4, min(0.9, W / 1600))
    placed = []
    for f in features[:max_labels]:
        u, v = int(round(f["u"])), int(round(f["v"]))
        col = (255, 255, 120) if f["crater"] else (120, 220, 255)
        if f["crater"]:
            poly = crater_outline(sol, f["lat"], f["lon"], f["diam_km"])
            if len(poly) > 3:
                cv2.polylines(bgr, [poly.round().astype(np.int32)], True, col, 1, cv2.LINE_AA)
        else:
            cv2.drawMarker(bgr, (u, v), col, cv2.MARKER_DIAMOND, 10, 2, cv2.LINE_AA)
        text = f["ascii"]
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        x, y = min(max(u - tw // 2, 2), W - tw - 2), min(max(v - 6, th + 2), H - 4)
        box = (x, y - th, x + tw, y + 2)
        if any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in placed):
            continue                                  # skip labels that would overlap a larger one
        placed.append(box)
        cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
    return bgr


def pixel_info(sol, u, v, gazetteer=None):
    """Latitude, longitude, LOLA elevation and nearest named feature of one pixel."""
    lat, lon = sol.pixel_to_latlon(np.array([float(u)]), np.array([float(v)]))
    lat, lon = float(lat[0]), float(lon[0])
    if not np.isfinite(lat):
        return dict(lat=None, lon=None)
    from .config import LDEM_PPD
    from .terrain import load_ldem
    dem = load_ldem()
    r = int(np.clip((90 - lat) * LDEM_PPD, 0, dem.shape[0] - 1))
    c = int((lon % 360) * LDEM_PPD) % dem.shape[1]
    info = dict(lat=lat, lon=lon, elevation_km=float(dem[r, c]) * 0.0005)
    if gazetteer is not None:
        name, typ, dist, inside = gazetteer.nearest(lat, lon, 1)[0]
        info.update(nearest=name, nearest_type=typ, nearest_km=dist, inside=inside)
    return info
