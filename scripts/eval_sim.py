"""Benchmark the solver on random camera views.

    python scripts/eval_sim.py --mode fake  --n 50          # catalog-derived noisy detections
    python scripts/eval_sim.py --mode wac   --n 30          # WAC mosaic rendered views + CraterNet
    python scripts/eval_sim.py --mode fake  --blind-focal   # do not give the solver the focal length
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.config import MOON_RADIUS_KM  # noqa: E402
from lunarsolve.geometry import angular_dist_deg, pixel_rays, ray_sphere, vec_to_latlon  # noqa: E402
from lunarsolve.simulate import fake_detections, random_view, render_dem, render_wac  # noqa: E402
from lunarsolve.solve import Solver  # noqa: E402


def truth_center(v):
    W, H = v["W"], v["H"]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    d = pixel_rays(np.array([cx]), np.array([cy]), v["C"], v["Rwc"], v["f"], cx, cy)
    p, hit = ray_sphere(v["C"], d)
    return vec_to_latlon(p[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fake", "wac", "dem"], default="fake")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--alt", type=float, nargs=2, default=(30, 1500))
    ap.add_argument("--fov", type=float, nargs=2, default=(5, 40))
    ap.add_argument("--tilt", type=float, default=20)
    ap.add_argument("--weights", default=None, help="CraterNet weights file")
    ap.add_argument("--catalog", default=None, help="crater catalog npz (default: Robbins)")
    ap.add_argument("--index", default=None, help="quad index npz matching --catalog")
    ap.add_argument("--multi", action="store_true", help="Robbins, then the CraterNet catalog (lunarsolve.multi)")
    ap.add_argument("--blind-focal", action="store_true")
    ap.add_argument("--sun", type=float, nargs=2, default=(3, 60), help="dem mode: sun elevation range (deg)")
    ap.add_argument("--no-albedo", action="store_true", help="dem mode: no Clementine albedo")
    ap.add_argument("--max-time", type=float, default=15)
    ap.add_argument("--min-gsd", type=float, default=None,
                    help="km/px; default 0.1 for wac (mosaic resolution) and 0.5 for dem")
    ap.add_argument("--lon", type=float, nargs=2, default=(0, 360), help="longitude range (330-360 is held out)")
    ap.add_argument("--save", default=None, help="directory to save rendered test images")
    args = ap.parse_args()

    if args.multi:
        from lunarsolve.multi import default_solver
        s = default_solver()
    elif args.catalog:
        from lunarsolve.catalog import Catalog
        from lunarsolve.index import QuadIndex
        s = Solver(catalog=Catalog(Path(args.catalog)), index=QuadIndex(Path(args.index)))
    else:
        s = Solver()
    if args.weights:
        from lunarsolve.detect import load_model
        s._model = load_model(args.weights, device="cuda")
    print(f"solver load {s.load_time:.1f}s, {len(s.index)} quads")
    wac = dem = None
    if args.mode == "wac":
        from lunarsolve.terrain import WacMosaic
        wac = WacMosaic()
    albedo = None
    if args.mode == "dem":
        from lunarsolve.terrain import GpuAlbedo, GpuDem
        dem = GpuDem()
        if not args.no_albedo:
            albedo = GpuAlbedo()
    min_gsd = args.min_gsd if args.min_gsd is not None else {"wac": 0.1, "dem": 0.5}.get(args.mode, 0.0)
    rng = np.random.default_rng(args.seed)
    rows = []
    while len(rows) < args.n:
        v = random_view(rng, args.size, args.size, tuple(args.alt), tuple(args.fov), args.tilt, lat_max=80,
                        lon_range=tuple(args.lon))
        if v["alt"] / v["f"] < min_gsd:
            continue
        extra = ""
        if args.mode == "fake":
            d = fake_detections(s.cat, v, rng)
            if d is None or len(d) < 6:
                continue
            t0 = time.time()
            sol = s.solve(dets=d, shape=(v["H"], v["W"]), focal_px=None if args.blind_focal else v["f"],
                          max_time=args.max_time)
            nd = len(d)
        else:
            if args.mode == "wac":
                img = render_wac(wac, v)
            else:
                img, el, az = render_dem(dem, v, rng, sun_el_deg=rng.uniform(*args.sun), albedo=albedo)
                extra = f" sun {el:4.1f}"
            if args.save:
                import cv2
                Path(args.save).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(f"{args.save}/view_{len(rows):03d}.png", (img * 255).clip(0, 255).astype(np.uint8))
            t0 = time.time()
            sol = s.solve(img=img, focal_px=None if args.blind_focal else v["f"], max_time=args.max_time)
            nd = s.stats.get("n_det", 0)
        dt = time.time() - t0
        tlat, tlon = truth_center(v)
        if sol is None:
            ok, cerr, aerr = False, np.nan, np.nan
        else:
            cerr = angular_dist_deg(sol.center_lat, sol.center_lon, tlat, tlon) * math.pi / 180 * MOON_RADIUS_KM
            aerr = (sol.altitude_km - v["alt"]) / v["alt"]
            ok = cerr < 0.02 * v["alt"] * math.tan(math.radians(v["fov"])) * 2 + 1.0
        rows.append(dict(ok=ok, solved=sol is not None, alt=v["alt"], fov=v["fov"], tilt=v["tilt"], nd=nd,
                         cerr=cerr, aerr=aerr, t=dt, nm=sol.n_matches if sol else 0,
                         asig=sol.altitude_sigma_km / v["alt"] if sol else np.nan))
        r = rows[-1]
        tag = "OK  " if ok else ("BAD " if sol is not None else "FAIL")
        print(f"{tag} alt {r['alt']:6.0f} fov {r['fov']:4.1f} tilt {r['tilt']:4.1f} det {nd:5d} match {r['nm']:4d} "
              f"centre err {r['cerr']:7.2f} km  alt err {100 * r['aerr']:+6.1f}% (sigma {100 * r['asig']:5.1f}%)  "
              f"{dt * 1000:6.0f} ms{extra}", flush=True)
    ok = np.array([r["ok"] for r in rows])
    solved = np.array([r["solved"] for r in rows])
    t = np.array([r["t"] for r in rows])
    ae = np.array([abs(r["aerr"]) for r in rows if r["ok"]])
    print(f"\nsuccess {ok.sum()}/{len(rows)}  wrong {(solved & ~ok).sum()}  "
          f"median time {np.median(t[ok]) * 1000 if ok.any() else 0:.0f} ms (solved)  "
          f"median |alt err| {100 * np.median(ae) if len(ae) else 0:.2f}%")


if __name__ == "__main__":
    main()
