"""Small-footprint benchmark: crops of Apollo Metric Camera scans (independent 1971 film imagery).

    python scripts/eval_apollo_crops.py                 # Robbins catalog
    python scripts/eval_apollo_crops.py --fine          # fine catalog (Robbins + CraterNet on WAC)

For each frame, the full 4048 px scan (about 44 m/px) is solved first with the Robbins catalog
and the known focal length (76.2 mm lens on a 114.3 mm frame).  That solution gives the true
ground position of every crop.  Crops of 256, 512 and 1024 px (about 11, 22 and 45 km) are then
solved with the camera's known intrinsics (focal length and the shifted principal point).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.config import DATA_DIR, MOON_RADIUS_KM  # noqa: E402
from lunarsolve.detect import to_gray_float  # noqa: E402
from lunarsolve.geometry import angular_dist_deg  # noqa: E402

FRAMES = ["AS16-M-1657", "AS16-M-0966", "AS15-M-0906", "AS15-M-0313", "AS17-M-1702", "AS15-M-2238"]
APOLLO = DATA_DIR / "apollo"
TRUTH_PATH = APOLLO / "crop_truth.json"


def fetch_med(name):
    p = APOLLO / f"{name}_MED.png"
    if not p.exists():
        r = requests.get(f"https://apollo.im-ldi.com/data/metric/{name[:4]}/png/{name}_MED.png", timeout=300)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def truth_solutions(solver_robbins):
    """Full-frame solutions (cached): camera centre, rotation, focal length."""
    truth = json.loads(TRUTH_PATH.read_text()) if TRUTH_PATH.exists() else {}
    for name in FRAMES:
        if name in truth:
            continue
        img = to_gray_float(cv2.imread(str(fetch_med(name)), cv2.IMREAD_UNCHANGED))
        f = img.shape[1] * 76.2 / 114.3
        sol = solver_robbins.solve(img=img, focal_px=f, parity="normal", max_time=120)
        if sol is None:
            print(f"{name}: full frame did not solve; skipped")
            continue
        truth[name] = dict(C=sol.C.tolist(), Rwc=sol.Rwc.tolist(), f=f, shape=list(img.shape),
                           matches=int(sol.n_matches), rms=float(sol.rms_px))
        print(f"{name}: full frame solved, {sol.n_matches} matches, rms {sol.rms_px:.2f} px")
        TRUTH_PATH.write_text(json.dumps(truth))
    return truth


def ground_point(t, u, v):
    from lunarsolve.geometry import pixel_rays, ray_sphere, vec_to_latlon
    H, W = t["shape"]
    C, Rwc = np.array(t["C"]), np.array(t["Rwc"])
    d = pixel_rays(np.array([u]), np.array([v]), C, Rwc, t["f"], (W - 1) / 2, (H - 1) / 2)
    p, hit = ray_sphere(C, d)
    lat, lon = vec_to_latlon(p[0])
    return float(lat), float(lon)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fine", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--per-frame", type=int, default=6)
    ap.add_argument("--max-time", type=float, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lunarsolve.catalog import Catalog
    from lunarsolve.index import QuadIndex
    from lunarsolve.solve import Solver
    if not TRUTH_PATH.exists() or len(json.loads(TRUTH_PATH.read_text())) < len(FRAMES):
        truth = truth_solutions(Solver())
    truth = json.loads(TRUTH_PATH.read_text())
    if args.fine:
        from lunarsolve.config import FINE_CATALOG_NPZ, FINE_INDEX_PATH
        solver = Solver(catalog=Catalog(FINE_CATALOG_NPZ), index=QuadIndex(FINE_INDEX_PATH))
    else:
        solver = Solver()
    print(f"catalog: {'fine' if args.fine else 'Robbins'} ({len(solver.cat):,} craters, {len(solver.index):,} quads)")
    rng = np.random.default_rng(args.seed)
    res = {s: [] for s in args.sizes}
    for name, t in truth.items():
        img = to_gray_float(cv2.imread(str(fetch_med(name)), cv2.IMREAD_UNCHANGED))
        H, W = img.shape
        cx, cy = (W - 1) / 2, (H - 1) / 2
        gsd = (np.linalg.norm(t["C"]) - MOON_RADIUS_KM) / t["f"]
        for size in args.sizes:
            for _ in range(args.per_frame):
                x0 = int(rng.integers(500, W - 500 - size))
                y0 = int(rng.integers(500, H - 500 - size))
                crop = img[y0:y0 + size, x0:x0 + size]
                tl, tn = ground_point(t, x0 + (size - 1) / 2, y0 + (size - 1) / 2)
                t0 = time.time()
                sol = solver.solve(img=crop, focal_px=t["f"], principal=(cx - x0, cy - y0), parity="normal",
                                   max_time=args.max_time)
                dt = time.time() - t0
                foot = size * gsd
                if sol is None:
                    res[size].append(dict(ok=False, solved=False, t=dt))
                    print(f"  {name} {size}px ({foot:.0f} km): no solution  {dt:.1f}s", flush=True)
                    continue
                err = angular_dist_deg(sol.center_lat, sol.center_lon, tl, tn) * math.pi / 180 * MOON_RADIUS_KM
                ok = err < max(1.0, 0.1 * foot)
                res[size].append(dict(ok=ok, solved=True, err=err, t=dt, n=sol.n_matches))
                print(f"  {name} {size}px ({foot:.0f} km): {'OK ' if ok else 'BAD'} error {err:.2f} km, "
                      f"{sol.n_matches} matches  {dt:.1f}s", flush=True)
    print()
    for size, rows in res.items():
        ok = [r for r in rows if r["ok"]]
        bad = sum(r["solved"] and not r["ok"] for r in rows)
        med = np.median([r["err"] for r in ok]) if ok else float("nan")
        print(f"{size:5d} px (~{size * 0.044:.0f} km): solved {len(ok)}/{len(rows)}, wrong {bad}, "
              f"median error {med:.2f} km")


if __name__ == "__main__":
    main()
