"""Benchmark on real spacecraft imagery: Apollo 15/16/17 Metric (mapping) Camera frames.

Browse images and the per-frame state vectors come from the ASU Apollo Digital Image
Archive (https://apollo.im-ldi.com).  The state vectors give the spacecraft position
(selenographic x, y, z in km), the nadir point and the camera-axis intersection, which
serve as ground truth.

    python scripts/eval_apollo.py --n 40                 # random frames, blind solve
    python scripts/eval_apollo.py --frames AS15-M-1000 AS16-M-0500
    python scripts/eval_apollo.py --n 40 --fov 73.7      # give the nominal field of view
"""
import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.config import DATA_DIR, MOON_RADIUS_KM  # noqa: E402
from lunarsolve.detect import to_gray_float  # noqa: E402
from lunarsolve.geometry import angular_dist_deg, latlon_to_vec, vec_to_latlon  # noqa: E402
from lunarsolve.solve import Solver  # noqa: E402

BASE = "https://apollo.im-ldi.com/data/metric"
APOLLO_DIR = DATA_DIR / "apollo"
ALT_FIELD_OFFSET_KM = 0.69    # |selenographic position| - 1737.4 - spacecraft_altitude (median over all rows)
METRIC_FOV_DEG = 2 * math.degrees(math.atan(114.3 / 2 / 76.2))   # 76.2 mm lens, 4.5 in frame


def load_state_vectors():
    rows = {}
    for m in ("AS15", "AS16", "AS17"):
        p = APOLLO_DIR / f"{m}_Metric_SV.csv"
        if not p.exists():
            APOLLO_DIR.mkdir(parents=True, exist_ok=True)
            r = requests.get(f"{BASE}/{m}/state_vectors/{m}_Metric_SV.csv", timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
        with open(p, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    pos = np.array([float(row[k]) for k in ("selenographic_x", "selenographic_y", "selenographic_z")])
                except (ValueError, KeyError):
                    continue
                if not np.isfinite(pos).all() or np.linalg.norm(pos) < MOON_RADIUS_KM:
                    continue
                axis = re.findall(r"-?\d+\.?\d*", row.get("camera_axis_intersect") or "")
                # Truth: the nadir point and altitude fields.  About 4% of rows have a selenographic
                # position vector that disagrees with them by > 5 km; those rows are flagged.
                nad = re.findall(r"-?\d+\.?\d*", row.get("nadir_point") or "")
                try:
                    alt = float(row["spacecraft_altitude"]) + ALT_FIELD_OFFSET_KM
                except ValueError:
                    continue
                if len(nad) != 2:
                    continue
                nlon, nlat = float(nad[0]) % 360, float(nad[1])
                pos_f = latlon_to_vec(nlat, nlon) * (MOON_RADIUS_KM + alt)
                row["suspect"] = bool(np.linalg.norm(pos_f - pos) > 5.0)
                pos = pos_f
                row["pos"] = pos
                row["axis_lonlat"] = (float(axis[0]), float(axis[1])) if len(axis) == 2 else None
                try:
                    row["tilt"] = math.hypot(float(row["x_tilt"]), float(row["y_tilt"]))
                except ValueError:
                    row["tilt"] = float("nan")
                rows[row["image_name"]] = row
    return rows


def fetch_frame(name):
    m = name[:4]
    p = APOLLO_DIR / f"{name}_SML.png"
    if not p.exists():
        r = requests.get(f"{BASE}/{m}/png/{name}_SML.png", timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", nargs="*")
    ap.add_argument("--oblique", action="store_true", help="sample only frames tilted more than 30 degrees")
    ap.add_argument("--fov", type=float, default=None, help="give the solver this field of view (degrees)")
    ap.add_argument("--max-time", type=float, default=30)
    ap.add_argument("--weights", default=None, help="CraterNet weights file")
    ap.add_argument("--overlays", default=None, help="directory for annotated results")
    args = ap.parse_args()

    sv = load_state_vectors()
    names = args.frames or sorted(n for n in sv if (sv[n]["tilt"] > 30) == args.oblique or not args.oblique)
    if not args.frames:
        rng = np.random.default_rng(args.seed)
        names = list(rng.choice(names, size=min(args.n, len(names)), replace=False))
    solver = Solver()
    if args.weights:
        from lunarsolve.detect import load_model
        solver._model = load_model(args.weights, device="cuda")
    print(f"{len(sv)} frames with state vectors; solver loaded in {solver.load_time:.1f}s; nominal FOV {METRIC_FOV_DEG:.1f} deg")
    rows = []
    for name in names:
        row = sv[name]
        try:
            img = to_gray_float(cv2.imread(str(fetch_frame(name)), cv2.IMREAD_UNCHANGED))
        except Exception as e:  # noqa: BLE001
            print(f"{name}: download failed ({e})")
            continue
        t0 = time.time()
        sol = solver.solve(img=img, fov_deg=args.fov, max_time=args.max_time)
        dt = time.time() - t0
        n_det = solver.stats.get("n_det", 0)
        pos = row["pos"]
        tlat, tlon = vec_to_latlon(pos)
        talt = float(np.linalg.norm(pos) - MOON_RADIUS_KM)
        bright = float(np.median(img))
        if sol is None:
            print(f"FAIL {name}  truth {float(tlat):+7.3f} {float(tlon):8.3f} alt {talt:6.1f}  det {n_det:5d}  "
                  f"median brightness {bright:.2f}  {dt:5.1f}s", flush=True)
            rows.append(dict(ok=False, solved=False, t=dt))
            continue
        cam_err = float(np.linalg.norm(sol.C - pos))
        ground = angular_dist_deg(sol.camera_lat, sol.camera_lon, tlat, tlon) * math.pi / 180 * MOON_RADIUS_KM
        ax_err = np.nan
        if row["axis_lonlat"]:
            alon, alat = row["axis_lonlat"]
            ax_err = angular_dist_deg(sol.center_lat, sol.center_lon, alat, alon % 360) * math.pi / 180 * MOON_RADIUS_KM
        ok = ground < 20.0
        rows.append(dict(ok=ok, solved=True, t=dt, ground=ground, cam=cam_err, dalt=sol.altitude_km - talt,
                         asig=sol.altitude_sigma_km, fov=sol.fov_deg, ax=ax_err))
        tag = "OK  " if ok else "BAD "
        print(f"{tag}{name}  truth {float(tlat):+7.3f} {float(tlon):8.3f} alt {talt:6.1f} | solved "
              f"{sol.camera_lat:+7.3f} {sol.camera_lon:8.3f} alt {sol.altitude_km:6.1f} +/- {sol.altitude_sigma_km:5.1f}"
              f" fov {sol.fov_deg:5.1f} | nadir err {ground:5.2f} km  3-D err {cam_err:5.2f} km  axis err {ax_err:5.2f} km"
              f" | {sol.n_matches:4d} matches  {dt:5.1f}s  [{sol.method}, tilt {row['tilt']:.0f}]"
              f"{'  (truth row inconsistent)' if row['suspect'] else ''}", flush=True)
        if args.overlays:
            from lunarsolve.__main__ import draw_overlay
            Path(args.overlays).mkdir(parents=True, exist_ok=True)
            draw_overlay(img, sol.dets, sol, solver.cat, Path(args.overlays) / f"{name}.jpg")
    ok = [r for r in rows if r["ok"]]
    print(f"\nsolved {sum(r['solved'] for r in rows)}/{len(rows)}, correct {len(ok)}, "
          f"wrong {sum(r['solved'] and not r['ok'] for r in rows)}")
    if ok:
        g = np.array([r["ground"] for r in ok])
        da = np.array([r["dalt"] for r in ok])
        print(f"nadir error median {np.median(g):.2f} km (p90 {np.percentile(g, 90):.2f});  altitude error median "
              f"{np.median(da):+.2f} km, median |err| {np.median(np.abs(da)):.2f} km;  "
              f"FOV median {np.median([r['fov'] for r in ok]):.1f} deg;  time median {np.median([r['t'] for r in ok]):.1f}s")


if __name__ == "__main__":
    main()
