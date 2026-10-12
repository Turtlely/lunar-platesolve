"""Stage 1 of the fine catalog: run CraterNet over the global LROC WAC 100 m mosaic.

    python scripts/build_fine_catalog.py detect     # about 30-40 min on an RTX 4060 laptop GPU
    python scripts/build_fine_catalog.py repeat     # precision proxy: re-detection rate vs score
    python scripts/build_fine_catalog.py merge --score 0.5   # merge with Robbins -> fine_catalog.npz
    python -m lunarsolve.index --fine               # index of the fine catalog

The sphere is covered with gnomonic tiles at the mosaic's native 100 m/px.  The tile centres
are a Fibonacci lattice.  A detection belongs to the tile whose centre is nearest, so the tile
overlaps do not give duplicates.  Only small craters (D <= 4 km) are kept: Robbins is complete
above about 2 km.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.config import DATA_DIR, MOON_RADIUS_KM as R  # noqa: E402

RAW_PATH = DATA_DIR / "craternet_wac_detections.npz"
FINE_PATH = DATA_DIR / "fine_catalog.npz"
GSD_KM = 0.1            # native WAC mosaic scale
TILE = 2048             # px
SPACING_KM = 130.0      # Fibonacci lattice spacing; Voronoi cells fit inside the tile
D_MAX_KM = 4.0


def fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


CHUNK = 100                                         # tiles per checkpoint file


def detect_all(limit=None, start=0, out_path=RAW_PATH):
    """Detect over all tiles.  Results are saved every CHUNK tiles under DATA_DIR/fine_chunks, and
    finished chunks are skipped on a restart."""
    import torch
    from scipy.spatial import cKDTree
    from lunarsolve.detect import detect, load_model
    from lunarsolve.geometry import enu_basis, vec_to_latlon
    from lunarsolve.terrain import WacMosaic

    wac = WacMosaic()
    model = load_model()
    n_tiles = int(4 * math.pi * R * R / SPACING_KM ** 2)
    centres = fibonacci(n_tiles)
    owner = cKDTree(centres)
    c = (TILE - 1) / 2
    jj, ii = np.mgrid[0:TILE, 0:TILE].astype(np.float64)
    off_x = (ii - c) * GSD_KM
    off_y = (c - jj) * GSD_KM                       # north up
    chunk_dir = DATA_DIR / ("fine_chunks" if limit is None else "fine_chunks_test")   # tests never mix in
    chunk_dir.mkdir(exist_ok=True)
    stop = n_tiles if limit is None else min(n_tiles, start + limit)
    t0 = time.time()
    done_tiles = 0
    for k0 in range(start - start % CHUNK, stop, CHUNK):
        path = chunk_dir / f"chunk_{k0:05d}.npz"
        if path.exists() and limit is None:
            continue
        out = []
        for k in range(max(k0, start), min(k0 + CHUNK, stop)):
            lat0, lon0 = vec_to_latlon(centres[k])
            lat0, lon0 = float(lat0), float(lon0)
            e, n, u = enu_basis(lat0, lon0)
            v = u * R + e * off_x[..., None] + n * off_y[..., None]
            v /= np.linalg.norm(v, axis=-1, keepdims=True)
            lat = np.degrees(np.arcsin(v[..., 2]))
            lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
            del v
            img = (wac.sample(lat, lon, GSD_KM) / 255.0).astype(np.float32)
            del lat, lon
            done_tiles += 1
            if img.std() < 1e-3:
                continue
            d = detect(model, img)
            d = d[(d[:, 3] >= 0.3) & (2 * d[:, 2] * GSD_KM <= D_MAX_KM)]
            if len(d) == 0:
                continue
            # detection -> sphere (gnomonic inverse at the tile centre)
            x = (d[:, 0] - c) * GSD_KM
            y = (c - d[:, 1]) * GSD_KM
            p = u * R + e * x[:, None] + n * y[:, None]
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            _, own = owner.query(p)
            keep = own == k
            la, lo = vec_to_latlon(p[keep])
            # the gnomonic projection stretches away from the centre: first-order correction
            diam = 2 * d[keep, 2] * GSD_KM * (p[keep] @ u)
            out.append(np.column_stack([la, lo, diam, d[keep, 3], np.full(keep.sum(), k)]))
        a = np.concatenate(out) if out else np.zeros((0, 5))
        np.savez(path, a=a)
        torch.cuda.empty_cache()
        rate = (time.time() - t0) / max(done_tiles, 1)
        left = sum(1 for j in range(k0 + CHUNK, stop, CHUNK) if not (chunk_dir / f"chunk_{j:05d}.npz").exists())
        print(f"chunk {k0 // CHUNK + 1}/{-(-n_tiles // CHUNK)} saved ({len(a):,} craters)  {rate:.2f} s/tile  "
              f"eta {left * CHUNK * rate / 60:.0f} min", flush=True)
    parts = [np.load(chunk_dir / f"chunk_{k0:05d}.npz")["a"] for k0 in range(0, n_tiles, CHUNK)
             if (chunk_dir / f"chunk_{k0:05d}.npz").exists()]
    a = np.concatenate(parts) if parts else np.zeros((0, 5))
    np.savez(out_path, lat=a[:, 0], lon=a[:, 1], diam=a[:, 2].astype(np.float32),
             score=a[:, 3].astype(np.float32), tile=a[:, 4].astype(np.int32))
    print(f"saved {len(a):,} detections from {len(parts)} chunks to {out_path}")


def tile_detect(wac, model, lat0, lon0, rot_deg=0.0, size=TILE):
    """Detect craters in one gnomonic tile (optionally rotated). Returns unit vectors, D km, score."""
    from lunarsolve.detect import detect
    from lunarsolve.geometry import enu_basis
    c = (size - 1) / 2
    jj, ii = np.mgrid[0:size, 0:size].astype(np.float64)
    a = math.radians(rot_deg)
    ex = (ii - c) * GSD_KM
    ny = (c - jj) * GSD_KM
    off_x = math.cos(a) * ex - math.sin(a) * ny
    off_y = math.sin(a) * ex + math.cos(a) * ny
    e, n, u = enu_basis(lat0, lon0)
    v = u * R + e * off_x[..., None] + n * off_y[..., None]
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(v[..., 2]))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
    img = (wac.sample(lat, lon, GSD_KM) / 255.0).astype(np.float32)
    d = detect(model, img)
    d = d[(d[:, 3] >= 0.3) & (2 * d[:, 2] * GSD_KM <= D_MAX_KM)]
    x = (d[:, 0] - c) * GSD_KM
    y = (c - d[:, 1]) * GSD_KM
    xr = math.cos(a) * x - math.sin(a) * y
    yr = math.sin(a) * x + math.cos(a) * y
    p = u * R + e * xr[:, None] + n * yr[:, None]
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    return p, 2 * d[:, 2] * GSD_KM * (p @ u), d[:, 3]


def repeatability(n_tiles=12, seed=0):
    """Precision proxy: detect sample tiles twice (shifted 30 km, rotated 37 deg) and report the
    fraction of craters found again, by diameter and score.  Real craters repeat; noise mostly not."""
    from scipy.spatial import cKDTree
    from lunarsolve.detect import load_model
    from lunarsolve.geometry import enu_basis, vec_to_latlon
    from lunarsolve.terrain import WacMosaic
    wac, model = WacMosaic(), load_model()
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_tiles):
        lat0 = math.degrees(math.asin(rng.uniform(-0.85, 0.85)))
        lon0 = rng.uniform(0, 360)
        p1, d1, s1 = tile_detect(wac, model, lat0, lon0, 0.0)
        e, n, u = enu_basis(lat0, lon0)
        q = u * R + e * 30.0 + n * 20.0
        la2, lo2 = vec_to_latlon(q / np.linalg.norm(q))
        p2, d2, s2 = tile_detect(wac, model, float(la2), float(lo2), 37.0)
        # compare inside a disc both tiles cover well
        core = (p1 @ u) > math.cos(60 / R)
        t2 = cKDTree(p2)
        dist, j = t2.query(p1[core], k=3)
        ok = ((dist * R < d1[core, None] / 2) & (np.abs(np.log(d2[np.minimum(j, len(d2) - 1)] / d1[core, None])) < 0.4)).any(1)
        rows.append(np.column_stack([d1[core], s1[core], ok]))
    a = np.concatenate(rows)
    print("fraction re-detected (rows: diameter km, columns: score)")
    sb = [0.3, 0.4, 0.5, 0.6, 0.8, 1.01]
    print("            " + "  ".join(f"{sb[i]:.1f}-{sb[i + 1]:.1f}" for i in range(len(sb) - 1)))
    for lo, hi in ((0.6, 0.8), (0.8, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0)):
        cells = []
        for i in range(len(sb) - 1):
            m = (a[:, 0] >= lo) & (a[:, 0] < hi) & (a[:, 1] >= sb[i]) & (a[:, 1] < sb[i + 1])
            cells.append(f"{a[m, 2].mean():6.0%} ({m.sum():4d})" if m.sum() >= 10 else "      -      ")
        print(f"  {lo:.1f}-{hi:.1f} km " + "  ".join(cells))


def merge(score_min, d_robbins_km=2.5):
    """Robbins + CraterNet craters (D < d_robbins_km, score >= score_min) that match no Robbins crater."""
    from scipy.spatial import cKDTree
    from lunarsolve.catalog import Catalog
    from lunarsolve.geometry import latlon_to_vec
    raw = np.load(RAW_PATH)
    cat = Catalog()
    P = latlon_to_vec(raw["lat"], raw["lon"])
    D = raw["diam"].astype(np.float64)
    S = raw["score"]
    near = (cat.diam >= 0.5) & (cat.diam <= 2 * D_MAX_KM)
    rid = np.nonzero(near)[0]
    tree = cKDTree(cat.xyz[rid])
    dist, j = tree.query(P, k=4)
    rd = cat.diam[rid[j]]
    match = (dist * R < D[:, None] / 2) & (np.abs(np.log(rd / D[:, None])) < 0.4)
    matched = match.any(1)
    # diameter scale of CraterNet relative to Robbins, from matched craters of 1.5-4 km
    first = np.argmax(match, 1)
    sel = matched & (D >= 1.5)
    scale = float(np.median(D[sel] / rd[np.arange(len(D)), first][sel]))
    print(f"{len(D):,} detections; matched to Robbins {matched.mean():.1%}; D(CraterNet)/D(Robbins) = {scale:.3f}")
    new = (~matched) & (S >= score_min) & (D / scale < d_robbins_km)
    lat = np.concatenate([cat.lat, raw["lat"][new]])
    lon = np.concatenate([cat.lon, raw["lon"][new]])
    diam = np.concatenate([cat.diam, D[new] / scale]).astype(np.float32)
    source = np.concatenate([np.zeros(len(cat), np.int8), np.ones(new.sum(), np.int8)])
    score = np.concatenate([np.ones(len(cat), np.float32), S[new]])
    np.savez(FINE_PATH, lat=lat, lon=lon, diam=diam, source=source, score=score)
    print(f"fine catalog: {len(cat):,} Robbins + {new.sum():,} new = {len(lat):,} craters -> {FINE_PATH}")
    for lo, hi in ((0.5, 1.0), (1.0, 1.5), (1.5, 2.5)):
        print(f"  D {lo}-{hi} km: Robbins {((cat.diam >= lo) & (cat.diam < hi)).sum():,}, "
              f"new {((D[new] / scale >= lo) & (D[new] / scale < hi)).sum():,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["detect", "repeat", "merge"])
    ap.add_argument("--score", type=float, default=0.5, help="merge: minimum detection score for new craters")
    ap.add_argument("--limit", type=int, default=None, help="detect: only N tiles (testing)")
    ap.add_argument("--start", type=int, default=0, help="detect: first tile index")
    ap.add_argument("--out", default=str(RAW_PATH))
    args = ap.parse_args()
    if args.step == "detect":
        detect_all(args.limit, args.start, Path(args.out))
    elif args.step == "repeat":
        repeatability()
    else:
        merge(args.score)


if __name__ == "__main__":
    main()
