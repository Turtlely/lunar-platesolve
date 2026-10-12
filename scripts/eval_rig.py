"""Benchmark the rigid-rig joint solve against single-camera solves.

    python scripts/eval_rig.py --n 30

A rig of 3 cameras shares one centre: camera 0 looks at nadir, cameras 1 and 2 are pitched by
+/- SEP degrees along the image up axis.  Views are rendered from the real LROC WAC mosaic on LOLA
terrain.  Baselines (none of them uses the truth to choose):

  nadir   camera 0 alone
  pick    the single camera with the most crater matches
  tall    one nadir camera with the same pixel count as the rig (W x 3W) that covers the same
          along-track angle (FOV + 2 SEP); tests whether the gain comes from the geometry or only
          from the larger footprint
  best    the single camera with the smallest true error (optimistic; uses the truth)
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.config import MOON_RADIUS_KM as R  # noqa: E402
from lunarsolve.geometry import nadir_camera  # noqa: E402
from lunarsolve.rig import RigCamera, RigSolver  # noqa: E402
from lunarsolve.simulate import render_wac  # noqa: E402
from lunarsolve.solve import Solver  # noqa: E402


def rot_x(deg):
    a = math.radians(deg)
    return np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])


def errors(C_est, C):
    """(3-D, horizontal, altitude) error in km."""
    up = C / np.linalg.norm(C)
    d = C_est - C
    a = float(np.linalg.norm(C_est) - np.linalg.norm(C))
    return float(np.linalg.norm(d)), float(np.linalg.norm(d - (d @ up) * up)), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--fov", type=float, default=20.0)
    ap.add_argument("--sep", type=float, default=25.0, help="pitch of the side cameras (deg)")
    ap.add_argument("--alt", type=float, nargs=2, default=(40, 300))
    ap.add_argument("--max-time", type=float, default=30)
    ap.add_argument("--json", type=str, default=None, help="write all rows to this file")
    args = ap.parse_args()

    from lunarsolve.terrain import WacMosaic
    wac = WacMosaic()
    s = Solver()
    rs = RigSolver(s)
    W = args.size
    f = (W / 2) / math.tan(math.radians(args.fov) / 2)
    cams = [RigCamera(np.eye(3), f), RigCamera(rot_x(args.sep), f), RigCamera(rot_x(-args.sep), f)]
    # tall camera: same pixel count, same along-track angle
    Ht = 3 * W
    f_tall = (Ht / 2) / math.tan(math.radians(args.fov / 2 + args.sep))
    rng = np.random.default_rng(args.seed)
    rows = []
    for k in range(args.n):
        lat = math.degrees(math.asin(rng.uniform(-0.85, 0.85)))
        lon = rng.uniform(0, 360)
        alt = math.exp(rng.uniform(*np.log(args.alt)))
        C, Rwr = nadir_camera(lat, lon, alt, rng.uniform(0, 360), rng.uniform(0, 10), rng.uniform(0, 360))
        imgs = [render_wac(wac, dict(C=C, Rwc=cam.R_cr @ Rwr, f=f, W=W, H=W)) for cam in cams]
        row = dict(lat=lat, lon=lon, alt=alt)
        single = []
        for img in imgs:
            sol = s.solve(img=img, focal_px=f, parity="normal", max_time=args.max_time / 3, limb=False)
            single.append(None if sol is None else dict(err=errors(sol.C, C), n=sol.n_matches,
                                                        s_alt=sol.altitude_sigma_km))
        row["single"] = single
        ok = [x for x in single if x is not None]
        row["nadir"] = single[0]["err"] if single[0] else None
        row["pick"] = max(ok, key=lambda x: x["n"])["err"] if ok else None
        row["best"] = min(ok, key=lambda x: x["err"][0])["err"] if ok else None
        tall = render_wac(wac, dict(C=C, Rwc=Rwr, f=f_tall, W=W, H=Ht))
        sol = s.solve(img=tall, focal_px=f_tall, parity="normal", max_time=args.max_time, limb=False)
        row["tall"] = errors(sol.C, C) if sol is not None else None
        row["tall_s_alt"] = sol.altitude_sigma_km if sol is not None else None
        t0 = time.time()
        rig = rs.solve(imgs, cams, max_time=args.max_time)
        row["rig"] = errors(rig.C, C) if rig is not None else None
        if rig is not None:
            row.update(s_h=rig.sigma_horizontal_km, s_a=rig.sigma_altitude_km, n=rig.matches_per_camera,
                       t=time.time() - t0)
        rows.append(row)
        fmt = lambda e: "  fail" if e is None else f"{e[0] * 1000:6.0f}"
        print(f"{k:2d} alt {alt:5.0f} km | 3-D error (m): nadir {fmt(row['nadir'])} pick {fmt(row['pick'])} "
              f"best {fmt(row['best'])} tall {fmt(row['tall'])} | rig {fmt(row['rig'])}"
              + (f" (alt {row['rig'][2] * 1000:+.0f} m, sigma h/alt {row['s_h'] * 1000:.0f}/{row['s_a'] * 1000:.0f} m)"
                 if rig else ""), flush=True)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    n = len(rows)
    print(f"\n{n} rigs, FOV {args.fov} deg, side pitch +/-{args.sep} deg, {W} px, altitude {args.alt} km")
    print(f"{'method':8s} {'solved':>7s} {'3-D med':>8s} {'3-D p90':>8s} {'horiz med':>10s} {'|alt| med':>10s} "
          f"{'|alt|/alt med':>14s}")
    for m in ("nadir", "pick", "best", "tall", "rig"):
        e = [(r[m], r["alt"]) for r in rows if r[m] is not None]
        if not e:
            print(f"{m:8s} {0:>4d}/{n}")
            continue
        d3 = np.array([x[0][0] for x in e]) * 1000
        dh = np.array([x[0][1] for x in e]) * 1000
        da = np.array([abs(x[0][2]) for x in e]) * 1000
        rel = np.array([abs(x[0][2]) / x[1] for x in e]) * 100
        print(f"{m:8s} {len(e):>4d}/{n} {np.median(d3):7.0f}m {np.percentile(d3, 90):7.0f}m {np.median(dh):9.0f}m "
              f"{np.median(da):9.0f}m {np.median(rel):12.3f}%")
    # paired comparison on rigs where both the nadir camera and the rig solved
    both = [r for r in rows if r["rig"] is not None and r["nadir"] is not None]
    if both:
        ratio3 = np.median([r["nadir"][0] / max(r["rig"][0], 1e-6) for r in both])
        ratioa = np.median([abs(r["nadir"][2]) / max(abs(r["rig"][2]), 1e-6) for r in both])
        print(f"\npaired over {len(both)} rigs (nadir and rig both solved): median error ratio nadir/rig "
              f"3-D {ratio3:.1f}x, altitude {ratioa:.1f}x")
    ok = [r for r in rows if r["rig"] is not None]
    if ok:
        z_h = np.array([r["rig"][1] / r["s_h"] for r in ok])
        z_a = np.array([abs(r["rig"][2]) / r["s_a"] for r in ok])
        print(f"rig sigma check: horizontal error < 2 sigma in {np.mean(z_h < 2) * 100:.0f}%, "
              f"altitude error < 2 sigma in {np.mean(z_a < 2) * 100:.0f}%; median sigma horizontal "
              f"{np.median([r['s_h'] for r in ok]) * 1000:.0f} m, altitude {np.median([r['s_a'] for r in ok]) * 1000:.0f} m")


if __name__ == "__main__":
    main()
