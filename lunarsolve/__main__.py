"""Command-line blind lunar plate solver.

    python -m lunarsolve image.png
    python -m lunarsolve image.png --fov 20 --overlay solved.png --json result.json
"""
import argparse
import json
import math
import sys
import time

import cv2
import numpy as np


def draw_overlay(img, dets, sol, cat, path):
    out = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    H, W = img.shape
    for x, y, r, *_ in dets:
        cv2.circle(out, (int(round(x)), int(round(y))), max(1, int(round(r))), (0, 160, 255), 1)
    if sol is not None:
        for di, ci in sol.matches:
            x, y, r = dets[di, :3]
            cv2.circle(out, (int(round(x)), int(round(y))), max(1, int(round(r))), (0, 255, 0), 2)
        # latitude / longitude graticule
        vv, uu = np.mgrid[0:H:4, 0:W:4]
        lat, lon = sol.pixel_to_latlon(uu.ravel().astype(float), vv.ravel().astype(float))
        lat = lat.reshape(uu.shape)
        lon = lon.reshape(uu.shape)
        span = max(np.nanmax(lat) - np.nanmin(lat), 1e-3)
        step = 10 ** math.floor(math.log10(span / 2))
        if span / step > 8:
            step *= 2
        for grid, color in ((lat, (255, 200, 0)), (lon, (255, 0, 200))):
            g = np.floor(grid / step)
            edge = np.zeros_like(g, bool)
            edge[:-1] |= g[:-1] != g[1:]
            edge[:, :-1] |= g[:, :-1] != g[:, 1:]
            ys, xs = np.nonzero(edge & np.isfinite(grid))
            for y, x in zip(ys, xs):
                cv2.circle(out, (int(x * 4), int(y * 4)), 1, color, -1)
        txt = f"lat {sol.center_lat:+.3f} lon {sol.center_lon:.3f}  alt {sol.altitude_km:.0f} km"
        cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(path), out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="lunarsolve", description="Blind plate solve of a lunar surface image.")
    ap.add_argument("image")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fov", type=float, help="horizontal field of view in degrees, if known")
    g.add_argument("--focal", type=float, help="focal length in pixels, if known")
    ap.add_argument("--parity", choices=["both", "normal", "mirrored"], default="both")
    ap.add_argument("--max-time", type=float, default=30.0)
    ap.add_argument("--overlay", help="write an annotated PNG here")
    ap.add_argument("--json", help="write the solution as JSON here")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from .detect import to_gray_float
    from .solve import Solver
    raw = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if raw is None:
        sys.exit(f"cannot read {args.image}")
    img = to_gray_float(raw)
    t = time.time()
    solver = Solver(device=args.device)
    print(f"loaded catalog and index in {time.time() - t:.1f}s")
    t = time.time()
    dets = solver.detect(img)
    t_det = time.time() - t
    sol = solver.solve(dets=dets, shape=img.shape, focal_px=args.focal, fov_deg=args.fov, parity=args.parity,
                       max_time=args.max_time)
    print(f"{len(dets)} craters detected in {t_det * 1000:.0f} ms")
    if sol is None:
        print("no solution found")
    else:
        print(sol.summary())
        if sol.focal_estimated and sol.altitude_sigma_km > 0.2 * sol.altitude_km:
            print("note: altitude is poorly constrained without the focal length; pass --fov or --focal")
    if args.overlay:
        draw_overlay(img, sol.dets if sol is not None else dets, sol, solver.cat, args.overlay)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(sol.to_dict() if sol else {"solved": False}, fh, indent=2)
    return 0 if sol else 1


if __name__ == "__main__":
    sys.exit(main())
