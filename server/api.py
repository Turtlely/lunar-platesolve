"""HTTP API of the lunar plate solver (used by the website in web/).

    python server/api.py --port 8000          # then open web/index.html, or serve web/ statically

Endpoints
    GET  /api/health                   status, catalog, available features
    POST /api/solve                    one image -> camera position, detections, named features
    GET  /api/lookup?lat=..&lon=..     LOLA elevation and nearest named feature of a ground point

All geometry needed to draw overlays (camera centre, rotation, focal length, principal point) is in
the response, so the client converts pixels to latitude/longitude without more requests.

Protection (environment variables; the defaults are safe for a public deployment):
    LUNAR_ALLOWED_ORIGINS   comma-separated sites that may call the API from a browser (CORS)
                            default: https://lunar.rtang.dev, http://localhost:8000, http://127.0.0.1:8000
    LUNAR_ENABLE_DOCS       1 = serve /docs and /openapi.json (default 0: off)
    LUNAR_RATE_SOLVE        solves per visitor, "count/seconds" (default 6/60)
    LUNAR_RATE_LOOKUP       point lookups per visitor (default 60/60)
    TURNSTILE_SECRET        Cloudflare Turnstile secret key; when set, every solve must carry a valid
                            token (form field "turnstile") from the website: other programs cannot solve
The visitor is identified by the CF-Connecting-IP header (Cloudflare sets it; behind the tunnel every
request passes Cloudflare), else by the connection address.
"""
import argparse
import collections
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from lunarsolve.config import MOON_RADIUS_KM as R  # noqa: E402
from lunarsolve.detect import to_gray_float  # noqa: E402

VERSION = "0.3.0"
MAX_UPLOAD_MB = 30
MAX_SIDE_PX = 4096
BUSY_WAIT_S = 20.0          # a request waits this long for the running solve, then gets 503 (busy)

ALLOWED_ORIGINS = [o.strip() for o in os.environ.get(
    "LUNAR_ALLOWED_ORIGINS", "https://lunar.rtang.dev,http://localhost:8000,http://127.0.0.1:8000").split(",")
    if o.strip()]
ENABLE_DOCS = os.environ.get("LUNAR_ENABLE_DOCS", "0") == "1"
TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET", "").strip()
TURNSTILE_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def _rate(name, default):
    n, sec = os.environ.get(name, default).split("/")
    return int(n), float(sec)


app = FastAPI(title="Lunar plate solver API", version=VERSION,
              docs_url="/docs" if ENABLE_DOCS else None, redoc_url=None,
              openapi_url="/openapi.json" if ENABLE_DOCS else None)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                   allow_headers=["Content-Type"], max_age=600)


class RateLimiter:
    """Sliding-window limit per visitor: at most n requests in any `seconds`."""

    def __init__(self, n, seconds):
        self.n, self.seconds = n, seconds
        self.hits = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()
        self.last_sweep = time.time()

    def check(self, key):
        """Raise HTTP 429 when `key` is over the limit; else record the request."""
        now = time.time()
        with self.lock:
            q = self.hits[key]
            while q and q[0] <= now - self.seconds:
                q.popleft()
            if len(q) >= self.n:
                wait = int(q[0] + self.seconds - now) + 1
                raise HTTPException(429, f"Too many requests. Try again in {wait} s.",
                                    headers={"Retry-After": str(wait)})
            q.append(now)
            if now - self.last_sweep > 300:                 # forget idle visitors
                for k in [k for k, v in self.hits.items() if not v or v[-1] <= now - self.seconds]:
                    del self.hits[k]
                self.last_sweep = now


solve_limit = RateLimiter(*_rate("LUNAR_RATE_SOLVE", "6/60"))
lookup_limit = RateLimiter(*_rate("LUNAR_RATE_LOOKUP", "60/60"))


def visitor(request):
    return request.headers.get("cf-connecting-ip") or (request.client.host if request.client else "unknown")


def check_turnstile(token, ip):
    """Verify a Cloudflare Turnstile token (one use, about 5 minutes valid).  No-op without a secret."""
    if not TURNSTILE_SECRET:
        return
    if not token:
        raise HTTPException(403, "Human check missing. Reload the page and try again.")
    import requests
    try:
        r = requests.post(TURNSTILE_URL, data=dict(secret=TURNSTILE_SECRET, response=token, remoteip=ip), timeout=10)
        ok = bool(r.json().get("success"))
    except Exception as ex:  # noqa: BLE001
        print("turnstile check failed:", ex, flush=True)
        raise HTTPException(503, "The human check is not available. Try again in a minute.")
    if not ok:
        raise HTTPException(403, "Human check failed. Reload the page and try again.")

_state = {}
_lock = threading.Lock()            # one solve at a time (the GPU/CPU model and the solver are shared)


def solver():
    """Robbins first, then the CraterNet catalog when its files are installed (lunarsolve.multi)."""
    if "solver" not in _state:
        from lunarsolve.multi import default_solver
        s = default_solver()
        s.model
        _state["solver"] = s
    return _state["solver"]


def gazetteer():
    if "gaz" not in _state:
        from lunarsolve.names import Gazetteer
        _state["gaz"] = Gazetteer(catalog=solver().cat)
    return _state["gaz"]


def refine_available():
    from lunarsolve.config import WAC_TIF
    return WAC_TIF.exists()


def refiner():
    if "refiner" not in _state:
        from lunarsolve.refine import Refiner
        _state["refiner"] = Refiner()
    return _state["refiner"]


def _f(x, nd=6):
    x = float(x)
    return None if not math.isfinite(x) else round(x, nd)


def _border(W, H, n=24):
    t = np.linspace(0, 1, n)
    return np.vstack([np.column_stack([t * (W - 1), np.zeros(n)]),
                      np.column_stack([np.full(n, W - 1.0), t * (H - 1)]),
                      np.column_stack([(1 - t) * (W - 1), np.full(n, H - 1.0)]),
                      np.column_stack([np.zeros(n), (1 - t) * (H - 1)])])


def geo_grid(sol, img_shape, n=48):
    """Latitude/longitude on an (n+1) x (ny+1) grid of image points, on the LOLA terrain.

    The browser interpolates it for hover readouts and grid lines (the mean sphere is off by
    height x tan(viewing angle), several km in oblique views).  None = the ray misses the Moon."""
    H, W = img_shape
    ny = max(2, round(n * H / W))
    uu, vv = np.meshgrid(np.linspace(0, W - 1, n + 1), np.linspace(0, H - 1, ny + 1))
    lat, lon = sol.pixel_to_latlon(uu.ravel(), vv.ravel())
    return dict(nx=n, ny=ny, lat=[_f(x, 5) for x in lat], lon=[_f(x, 5) for x in lon])


def matched_catalog(sol, cat):
    """Image position and radius (px) of every matched catalog crater, through the solved camera."""
    from lunarsolve.geometry import project
    if not len(sol.matches):
        return []
    H, W = sol.shape
    cx, cy = sol.principal()
    ids = sol.matches[:, 1]
    P = cat.points_km(ids)
    uv, _ = project(P, sol.C, sol.Rwc, sol.focal_px, cx, cy)
    if sol.mirrored:
        uv[:, 0] = W - 1 - uv[:, 0]
    rng = np.linalg.norm(P - sol.C, axis=1)
    r = cat.diam[ids] / 2 * sol.focal_px / rng
    return [[_f(u, 1), _f(v, 1), _f(rr, 1)] for (u, v), rr in zip(uv, r)]


def solution_json(sol, img_shape, labels=True, cat=None):
    from lunarsolve.names import crater_outline
    H, W = img_shape
    cx, cy = sol.principal()
    b = _border(W, H)
    disk = getattr(sol, "disk", None)
    if disk is not None:
        # far view: the footprint is the image rectangle cut by the disk edge
        on = np.hypot(b[:, 0] - disk["cx"], b[:, 1] - disk["cy"]) < disk["r"] * 0.995
        a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        cpts = np.column_stack([disk["cx"] + 0.995 * disk["r"] * np.cos(a), disk["cy"] + 0.995 * disk["r"] * np.sin(a)])
        cin = (cpts[:, 0] >= 0) & (cpts[:, 0] <= W - 1) & (cpts[:, 1] >= 0) & (cpts[:, 1] <= H - 1)
        ring = np.vstack([b[on], cpts[cin]])
        if len(ring) >= 3:
            c0 = ring.mean(0)
            b = ring[np.argsort(np.arctan2(ring[:, 1] - c0[1], ring[:, 0] - c0[0]))]
    lat, lon = sol.pixel_to_latlon(b[:, 0], b[:, 1])
    ok = np.isfinite(lat)
    feats = []
    if labels:
        for f in gazetteer().in_view(sol)[:120]:
            item = dict(name=f["name"], type=f["type"].split(",")[0], lat=_f(f["lat"], 4), lon=_f(f["lon"], 4),
                        diam_km=_f(f["diam_km"], 2), u=_f(f["u"], 1), v=_f(f["v"], 1), crater=f["crater"],
                        d_px=_f(f["d_px"], 1))
            if f["crater"]:
                poly = crater_outline(sol, f["lat"], f["lon"], f["diam_km"], n=40)
                item["outline"] = [[_f(p[0], 1), _f(p[1], 1)] for p in poly]
            feats.append(item)
    out = dict(
        camera_lat=_f(sol.camera_lat), camera_lon=_f(sol.camera_lon), altitude_km=_f(sol.altitude_km, 4),
        altitude_sigma_km=_f(sol.altitude_sigma_km, 4), center_lat=_f(sol.center_lat), center_lon=_f(sol.center_lon),
        gsd_m=_f(sol.gsd_km * 1000, 2), heading_deg=_f(sol.heading_deg, 2), off_nadir_deg=_f(sol.off_nadir_deg, 2),
        fov_deg=_f(sol.fov_deg, 3), focal_px=_f(sol.focal_px, 2), focal_estimated=bool(sol.focal_estimated),
        mirrored=bool(sol.mirrored), n_matches=int(sol.n_matches), log_odds=_f(sol.log_odds, 1),
        rms_px=_f(sol.rms_px, 3), method=sol.method, catalog=getattr(sol, "catalog_name", "Robbins"),
        distance_measured=bool(getattr(sol, "distance_measured", True)),
        disk=({k: _f(v, 1) for k, v in sol.disk.items()} if getattr(sol, "disk", None) else None),
        camera=dict(C=[float(x) for x in sol.C], Rwc=[[float(x) for x in r] for r in sol.Rwc],
                    f=float(sol.focal_px), cx=float(cx), cy=float(cy), width=W, height=H,
                    mirrored=bool(sol.mirrored), moon_radius_km=R),
        footprint=[[_f(a, 5), _f(o, 5)] for a, o in zip(lat[ok], lon[ok])],
        matched=[int(i) for i in sol.matches[:, 0]] if len(sol.matches) else [],
        matched_catalog=matched_catalog(sol, cat) if cat is not None else [],
        geo=geo_grid(sol, img_shape),
        features=feats)
    ref = getattr(sol, "refine", None)
    if getattr(sol, "refine_rejected", None):
        out["refine_rejected"] = sol.refine_rejected
    elif ref is not None:
        out["refined"] = dict(sigma_horizontal_m=_f(ref.sigma_horizontal_km * 1000, 1),
                              sigma_altitude_m=_f(ref.sigma_altitude_km * 1000, 1), patches=int(sum(ref.n_patches)),
                              shift_m=_f(ref.shift_km * 1000, 1), rms_px=_f(ref.rms_px, 3), time_s=_f(ref.time_s, 2))
    return out


@app.get("/api/health")
def health():
    names = solver().names if "solver" in _state else []
    return dict(ok=True, version=VERSION, loaded="solver" in _state, catalogs=names,
                refine=refine_available(), max_upload_mb=MAX_UPLOAD_MB, human_check=bool(TURNSTILE_SECRET))


@app.post("/api/solve")
def solve(request: Request, image: UploadFile = File(...), fov_deg: float = Form(0.0), parity: str = Form("both"),
          max_time: float = Form(30.0), labels: bool = Form(True), refine: bool = Form(False),
          turnstile: str = Form("")):
    ip = visitor(request)
    solve_limit.check(ip)
    check_turnstile(turnstile, ip)
    data = image.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"The image is larger than {MAX_UPLOAD_MB} MB.")
    raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise HTTPException(400, "The file is not an image that the server can read (use PNG, JPEG or TIFF).")
    img = to_gray_float(raw)
    H, W = img.shape
    scale = 1.0
    if max(H, W) > MAX_SIDE_PX:                       # keep memory and time bounded
        scale = MAX_SIDE_PX / max(H, W)
        img = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    if parity not in ("both", "normal", "mirrored"):
        parity = "both"
    max_time = float(np.clip(max_time, 3, 60))
    t0 = time.time()
    if not _lock.acquire(timeout=BUSY_WAIT_S):
        print(f"busy: rejected {W}x{H} image", flush=True)
        raise HTTPException(503, "The solver is busy with another image. Try again in a few seconds.")
    try:
        s = solver()
        sol = s.solve(img=img, fov_deg=fov_deg if fov_deg and fov_deg > 0 else None, parity=parity,
                      max_time=max_time)
        dets = sol.dets if sol is not None else s.detect(img)
        if sol is not None and refine:
            if not refine_available():
                sol.refine_rejected = "this server has no reference map (WAC mosaic)"
            else:
                try:
                    sol = refiner().refine(img, sol)
                except Exception as ex:  # noqa: BLE001
                    print("refine failed:", ex, flush=True)
                    sol.refine_rejected = "the registration failed"
        stats = dict(s.stats)
    finally:
        _lock.release()
    print(f"solve {W}x{H} ({len(data) / 1e6:.1f} MB) fov={fov_deg or '-'} -> "
          + (f"{sol.method} {sol.camera_lat:+.3f} {sol.camera_lon:.3f} alt {sol.altitude_km:.1f} km, "
             f"{sol.n_matches} matches" if sol is not None else f"no solution ({stats.get('n_det', 0)} craters)")
          + f" in {time.time() - t0:.1f} s", flush=True)
    d = np.asarray(dets, float)
    score = d[:, 3] if d.shape[1] > 3 else np.ones(len(d))   # solution detections carry no score column
    res = dict(ok=sol is not None, version=VERSION, time_s=round(time.time() - t0, 2),
               image=dict(width=img.shape[1], height=img.shape[0], scale=scale),
               detections=[[round(float(x), 1), round(float(y), 1), round(float(r), 2), round(float(sc), 3)]
                           for (x, y, r), sc in zip(d[:, :3], score)],
               limb=dict(tilt_deg=_f(stats["limb_tilt"], 2), altitude_km=_f(stats["limb_altitude_km"], 2))
               if "limb_tilt" in stats else None)
    if sol is not None:
        res["solution"] = solution_json(sol, img.shape, labels, getattr(sol, "catalog", s.cat))
    return res


@app.get("/api/lookup")
def lookup(request: Request, lat: float, lon: float):
    lookup_limit.check(visitor(request))
    if not (-90 <= lat <= 90 and math.isfinite(lon)):
        raise HTTPException(400, "lat must be in [-90, 90].")
    from lunarsolve.config import LDEM_PPD
    from lunarsolve.terrain import load_ldem
    dem = load_ldem()
    r = int(np.clip((90 - lat) * LDEM_PPD, 0, dem.shape[0] - 1))
    c = int((lon % 360) * LDEM_PPD) % dem.shape[1]
    name, typ, dist, inside = gazetteer().nearest(lat, lon, 1)[0]
    return dict(lat=lat, lon=lon % 360, elevation_km=round(float(dem[r, c]) * 0.0005, 3), nearest=name,
                nearest_type=typ.split(",")[0], nearest_km=round(dist, 2), inside=inside)


def main():
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--web", action="store_true", help="also serve the website from web/ at /")
    args = ap.parse_args()
    print(f"origins {ALLOWED_ORIGINS}; docs {'on' if ENABLE_DOCS else 'off'}; human check "
          f"{'on' if TURNSTILE_SECRET else 'off'}; rate solve {solve_limit.n}/{solve_limit.seconds:.0f}s, "
          f"lookup {lookup_limit.n}/{lookup_limit.seconds:.0f}s", flush=True)
    print("loading the catalog, the quad index and CraterNet ...", flush=True)
    solver()
    gazetteer()
    if args.web:
        app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
