"""Benchmark dense registration (lunarsolve.refine) after the crater solve.

    python scripts/eval_refine.py --n 20 --mode wac
    python scripts/eval_refine.py --n 20 --mode dem          # different lighting from the basemap
    python scripts/eval_refine.py --n 12 --rig               # rigid 3-camera rig

wac: views rendered from the WAC mosaic itself (same lighting as the reference; best case).
dem: views rendered from LOLA terrain + Clementine albedo with a random sun (lighting and albedo
     source differ from the reference; the 473 m DEM limits fine detail).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.geometry import nadir_camera  # noqa: E402
from lunarsolve.refine import Refiner  # noqa: E402
from lunarsolve.solve import Solver  # noqa: E402


def errors(C_est, C):
    up = C / np.linalg.norm(C)
    d = C_est - C
    return float(np.linalg.norm(d)), float(np.linalg.norm(d - (d @ up) * up)), float(np.linalg.norm(C_est) - np.linalg.norm(C))


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--mode", default="wac", choices=["wac", "dem"])
    ap.add_argument("--feature", default="bandpass", choices=["bandpass", "gradient"])
    ap.add_argument("--rig", action="store_true")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--alt", type=float, nargs=2, default=(60, 300))
    ap.add_argument("--fov", type=float, nargs=2, default=(15, 45))
    ap.add_argument("--max-time", type=float, default=20)
    args = ap.parse_args()

    from lunarsolve.simulate import render_dem, render_wac
    s = Solver()
    ref = Refiner()
    dem = alb = None
    if args.mode == "dem":
        from lunarsolve.terrain import GpuAlbedo, GpuDem
        dem, alb = GpuDem(), GpuAlbedo()
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.n):
        lat = math.degrees(math.asin(rng.uniform(-0.85, 0.85)))
        lon = rng.uniform(0, 360)
        alt = math.exp(rng.uniform(*np.log(args.alt)))
        fov = 20.0 if args.rig else math.exp(rng.uniform(*np.log(args.fov)))
        W = args.size
        f = (W / 2) / math.tan(math.radians(fov) / 2)
        C, Rwr = nadir_camera(lat, lon, alt, rng.uniform(0, 360), rng.uniform(0, 10), rng.uniform(0, 360))
        sun = (rng.uniform(15, 60), rng.uniform(0, 360))
        sun_txt = f" sun el {sun[0]:4.1f} az {sun[1]:5.1f}" if args.mode == "dem" else ""

        def render(Rwc):
            view = dict(C=C, Rwc=Rwc, f=f, W=W, H=W)
            if args.mode == "wac":
                return render_wac(ref.wac, view)
            return render_dem(dem, view, rng, sun_el_deg=sun[0], sun_az_deg=sun[1], albedo=alb)[0]

        if args.rig:
            from lunarsolve.rig import RigCamera, RigSolver
            cams = [RigCamera(rot_x(p), f) for p in (0, 25, -25)]
            imgs = [render(c.R_cr @ Rwr) for c in cams]
            sol = RigSolver(s).solve(imgs, cams, max_time=args.max_time)
            if sol is None:
                rows.append(None)
                print(f"{k:2d} alt {alt:5.0f} km: no solve", flush=True)
                continue
            t0 = time.time()
            new = ref.refine_rig(imgs, cams, sol, mode=args.feature)
            sig0, sig1 = (sol.sigma_horizontal_km, sol.sigma_altitude_km), (new.sigma_horizontal_km, new.sigma_altitude_km)
        else:
            img = render(Rwr)
            sol = s.solve(img=img, focal_px=f, parity="normal", max_time=args.max_time, limb=False)
            if sol is None:
                rows.append(None)
                print(f"{k:2d} alt {alt:5.0f} km fov {fov:4.1f}: no solve", flush=True)
                continue
            t0 = time.time()
            new = ref.refine(img, sol, mode=args.feature)
            sig0 = (float("nan"), sol.altitude_sigma_km)
            sig1 = ((new.refine.sigma_horizontal_km, new.refine.sigma_altitude_km)
                    if hasattr(new, "refine") else (float("nan"), float("nan")))
        e0, e1 = errors(sol.C, C), errors(new.C, C)
        info = new.refine if hasattr(new, "refine") else None
        gated = getattr(new, "refine_rejected", None)
        rows.append(dict(alt=alt, e0=e0, e1=e1, sig1=sig1, ok=info is not None,
                         patches=info.n_patches if info else None, t=time.time() - t0))
        print(f"{k:2d} alt {alt:5.0f} km fov {fov:4.1f}{sun_txt} | crater solve 3-D {e0[0] * 1000:6.0f} m "
              f"(h {e0[1] * 1000:5.0f}, alt {e0[2] * 1000:+5.0f}) -> dense {e1[0] * 1000:6.0f} m "
              f"(h {e1[1] * 1000:5.0f}, alt {e1[2] * 1000:+5.0f}) sigma h/alt {sig1[0] * 1000:4.0f}/{sig1[1] * 1000:3.0f} m"
              + (f" | patches {info.n_patches} frac {info.patch_frac:.2f} score {info.median_score:.2f} "
                 f"ncc {info.ncc:.2f} passes {info.iterations} {time.time() - t0:.1f}s" if info else " | FAILED")
              + (" | GATE: kept crater solve" if gated else ""),
              flush=True)
    ok = [r for r in rows if r is not None]
    ref_ok = [r for r in ok if r["ok"]]
    print(f"\n{args.mode} {'rig' if args.rig else 'single'} feature={args.feature}: solved {len(ok)}/{len(rows)}, "
          f"refined {len(ref_ok)}/{len(ok)}")
    if ref_ok:
        for name, key in (("crater solve", "e0"), ("dense", "e1")):
            d3 = np.array([r[key][0] for r in ref_ok]) * 1000
            dh = np.array([r[key][1] for r in ref_ok]) * 1000
            da = np.array([abs(r[key][2]) for r in ref_ok]) * 1000
            print(f"{name:13s} 3-D median {np.median(d3):6.0f} m  p90 {np.percentile(d3, 90):6.0f} m  "
                  f"horizontal median {np.median(dh):6.0f} m  |alt| median {np.median(da):5.0f} m")
        worse = sum(r["e1"][0] > r["e0"][0] * 1.5 + 0.05 for r in ref_ok)
        z = [r["e1"][1] / r["sig1"][0] for r in ref_ok if r["sig1"][0] > 0]
        print(f"refinement made {worse} solutions clearly worse; horizontal error < 2 sigma in "
              f"{np.mean(np.array(z) < 2) * 100:.0f}% (median sigma {np.median([r['sig1'][0] for r in ref_ok]) * 1000:.0f} m)")


if __name__ == "__main__":
    main()
