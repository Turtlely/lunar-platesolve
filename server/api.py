"""HTTP API of the lunar plate solver (used by the website in web/).

    python server/api.py --port 8000          # then open web/index.html, or serve web/ statically

Endpoints
    GET  /api/health                   status, catalog, available features
    POST /api/solve                    one image -> camera position, detections, named features
    GET  /api/lookup?lat=..&lon=..     LOLA elevation and nearest named feature of a ground point

All geometry needed to draw overlays (camera centre, rotation, focal length, principal point) is in
the response, so the client converts pixels to latitude/longitude without more requests.
"""
import argparse
import math
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from lunarsolve.config import MOON_RADIUS_KM as R  # noqa: E402
from lunarsolve.detect import to_gray_float  # noqa: E402

VERSION = "0.3.0"
MAX_UPLOAD_MB = 30
MAX_SIDE_PX = 4096

app = FastAPI(title="Lunar plate solver API", version=VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

_state = {}
_lock = threading.Lock()            # one solve at a time (the GPU/CPU model and the solver are shared)


def solver():
    if "solver" not in _state:
        from lunarsolve.solve import Solver
        s = Solver()
        s.model
        _state["solver"] = s
    return _state["solver"]


def gazetteer():
    if "gaz" not in _state:
        from lunarsolve.names import Gazetteer
        _state["gaz"] = Gazetteer()
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


def solution_json(sol, img_shape, labels=True):
    from lunarsolve.names import crater_outline
    H, W = img_shape
    cx, cy = sol.principal()
    b = _border(W, H)
    lat, lon = sol.pixel_to_latlon(b[:, 0], b[:, 1])
    ok = np.isfinite(lat)
    feats = []
    if labels:
        for f in gazetteer().in_view(sol)[:120]:
            item = dict(name=f["name"], type=f["type"].split(",")[0], lat=_f(f["lat"], 4), lon=_f(f["lon"], 4),
                        diam_km=_f(f["diam_km"], 2), u=_f(f["u"], 1), v=_f(f["v"], 1), crater=f["crater"])
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
        rms_px=_f(sol.rms_px, 3), method=sol.method,
        camera=dict(C=[float(x) for x in sol.C], Rwc=[[float(x) for x in r] for r in sol.Rwc],
                    f=float(sol.focal_px), cx=float(cx), cy=float(cy), width=W, height=H,
                    mirrored=bool(sol.mirrored), moon_radius_km=R),
        footprint=[[_f(a, 5), _f(o, 5)] for a, o in zip(lat[ok], lon[ok])],
        matched=[int(i) for i in sol.matches[:, 0]] if len(sol.matches) else [],
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
    return dict(ok=True, version=VERSION, loaded="solver" in _state, catalog="Robbins (1.3 M craters >= 1 km)",
                refine=refine_available(), max_upload_mb=MAX_UPLOAD_MB)


@app.post("/api/solve")
def solve(image: UploadFile = File(...), fov_deg: float = Form(0.0), parity: str = Form("both"),
          max_time: float = Form(30.0), labels: bool = Form(True), refine: bool = Form(False)):
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
    with _lock:
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
    d = np.asarray(dets, float)
    score = d[:, 3] if d.shape[1] > 3 else np.ones(len(d))   # solution detections carry no score column
    res = dict(ok=sol is not None, version=VERSION, time_s=round(time.time() - t0, 2),
               image=dict(width=img.shape[1], height=img.shape[0], scale=scale),
               detections=[[round(float(x), 1), round(float(y), 1), round(float(r), 2), round(float(sc), 3)]
                           for (x, y, r), sc in zip(d[:, :3], score)],
               limb=dict(tilt_deg=_f(stats["limb_tilt"], 2), altitude_km=_f(stats["limb_altitude_km"], 2))
               if "limb_tilt" in stats else None)
    if sol is not None:
        res["solution"] = solution_json(sol, img.shape, labels)
    return res


@app.get("/api/lookup")
def lookup(lat: float, lon: float):
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
    print("loading the catalog, the quad index and CraterNet ...", flush=True)
    solver()
    gazetteer()
    if args.web:
        app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
