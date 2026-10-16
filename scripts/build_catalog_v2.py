"""CraterNet catalog v2: our own crater catalog, built by the same detector that reads the images.

    python scripts/build_catalog_v2.py detect            # all passes (resumable), about 1-1.5 h
    python scripts/build_catalog_v2.py detect --pass B   # one pass
    python scripts/build_catalog_v2.py merge             # clusters + Robbins -> cn_catalog.npz
    python scripts/build_catalog_v2.py index             # quad index of the craters the detector sees

Why.  The solver hashes patterns of the largest craters it detects, as astrometry.net uses the
brightest stars.  When the index is built from a catalog made by other means (Robbins: manual and
semi-automatic mapping), the largest catalog craters of a region are often not the largest craters
the detector finds there (degraded rims, lighting, oblique views), and the patterns do not meet.
A catalog built by the detector itself, under several scales and lightings, has the same "largest
craters" as the images.

Passes (detector: weights/craternet_v3.pt unless --weights):
    A   WAC mosaic,           0.1 km/px, tiles 2048, D 0.5-6 km
    B   WAC mosaic,           0.4 km/px, tiles 1024, D 4-60 km
    C   WAC mosaic,           1.6 km/px, tiles  512, D 40-400 km
    D0..D3  LOLA shaded relief, 0.4 km/px, sun 20 deg from N/E/S/W, D 4-60 km
    E0..E1  LOLA shaded relief, 1.6 km/px, sun 20 deg from N/S,     D 40-400 km
The shaded-relief passes have no albedo and other sun directions than the mosaic, so a crater
found in several passes is a crater, not a lighting or albedo pattern.

Merge.  Detections of all passes are clustered (centres within 0.35 D, |ln D ratio| < 0.35).  Each
pass's diameter scale is calibrated against Robbins.  A cluster is kept when found by >= 2 passes,
or by one pass with score >= 0.5 (small craters exist only in pass A), or >= 0.6.  Clusters that
match a Robbins crater keep the Robbins position and diameter; the rest are new craters.  Every
catalog row has `seen` (found by the detector) and `n_pass`.  The quad index is built from seen
craters only.
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunarsolve.config import DATA_DIR, MOON_RADIUS_KM as R  # noqa: E402

OUT_DIR = DATA_DIR / "catalog_v2"
CN_CATALOG = DATA_DIR / "cn_catalog.npz"
CN_INDEX = DATA_DIR / "quad_index_cn.npz"
CHUNK = 50

PASSES = {
    "A": dict(src="wac", gsd=0.1, tile=2048, dmin=0.5, dmax=6.0),
    "B": dict(src="wac", gsd=0.4, tile=1024, dmin=4.0, dmax=60.0),
    "C": dict(src="wac", gsd=1.6, tile=512, dmin=40.0, dmax=400.0),
    "D0": dict(src="dem", gsd=0.4, tile=1024, dmin=4.0, dmax=60.0, sun_az=0.0, sun_el=20.0),
    "D1": dict(src="dem", gsd=0.4, tile=1024, dmin=4.0, dmax=60.0, sun_az=90.0, sun_el=20.0),
    "D2": dict(src="dem", gsd=0.4, tile=1024, dmin=4.0, dmax=60.0, sun_az=180.0, sun_el=20.0),
    "D3": dict(src="dem", gsd=0.4, tile=1024, dmin=4.0, dmax=60.0, sun_az=270.0, sun_el=20.0),
    "E0": dict(src="dem", gsd=1.6, tile=512, dmin=40.0, dmax=400.0, sun_az=0.0, sun_el=20.0),
    "E1": dict(src="dem", gsd=1.6, tile=512, dmin=40.0, dmax=400.0, sun_az=180.0, sun_el=20.0),
}


def fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


def tile_axes(sun_az_deg=None):
    """Image axes in (east, north) km per pixel: x right, y down.  For shaded relief the x axis
    points away from the sun, so cast shadows follow the rows."""
    if sun_az_deg is None:
        return np.array([1.0, 0.0]), np.array([0.0, -1.0])
    beta = math.radians(sun_az_deg) + math.pi
    return np.array([math.sin(beta), math.cos(beta)]), np.array([math.cos(beta), -math.sin(beta)])


def shaded_relief(dem, lat0, lon0, gsd, size, sun_az, sun_el, dev):
    """LOLA shaded relief of one tile, with cast shadows (no albedo).  Returns a float32 image."""
    import torch
    from lunarsolve.synth import grid_latlon
    ax, ay = tile_axes(sun_az)
    c = (size - 1) / 2
    ii = torch.arange(size, device=dev, dtype=torch.float64) - c
    gi, gj = torch.meshgrid(ii, ii, indexing="xy")
    xk = (gi * ax[0] + gj * ay[0]) * gsd
    yk = (gi * ax[1] + gj * ay[1]) * gsd
    lat, lon = grid_latlon(xk, yk, lat0, lon0)
    h = dem.sample(lat, lon, dem.level_for(gsd)).double()
    z = h - (xk ** 2 + yk ** 2) / (2 * R)
    zx = torch.zeros_like(z)
    zy = torch.zeros_like(z)
    zx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * gsd)
    zy[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2 * gsd)
    e = math.radians(sun_el)
    nrm = torch.rsqrt(1 + zx ** 2 + zy ** 2)
    mu0 = (zx * math.cos(e) + math.sin(e)) * nrm            # sun from -x at elevation e
    Hs = z + (gi + c) * gsd * math.tan(e)
    shadow = torch.cummax(Hs, dim=1).values > Hs + 1e-5
    img = torch.where(shadow, torch.zeros_like(mu0), mu0.clamp_min(0))
    img = img / torch.quantile(img.flatten()[::13].float(), 0.995).clamp_min(1e-6)
    return img.clamp(0, 1).float().cpu().numpy()


def run_pass(name, weights, limit=None):
    import torch
    from scipy.spatial import cKDTree
    from lunarsolve.detect import detect, load_model
    from lunarsolve.geometry import enu_basis, vec_to_latlon
    cfg = PASSES[name]
    gsd, size = cfg["gsd"], cfg["tile"]
    spacing = 0.63 * size * gsd                     # Voronoi cells fit inside the tiles
    n_tiles = int(4 * math.pi * R * R / spacing ** 2)
    centres = fibonacci(n_tiles)
    owner = cKDTree(centres)
    model = load_model(weights)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg["src"] == "wac":
        from lunarsolve.terrain import WacMosaic
        wac = WacMosaic()
        dem = None
    else:
        from lunarsolve.terrain import GpuDem
        dem = GpuDem(dev)
    ax, ay = tile_axes(cfg.get("sun_az"))
    c = (size - 1) / 2
    jj, ii = np.mgrid[0:size, 0:size].astype(np.float64)
    off_e = ((ii - c) * ax[0] + (jj - c) * ay[0]) * gsd
    off_n = ((ii - c) * ax[1] + (jj - c) * ay[1]) * gsd
    out_dir = OUT_DIR / name if limit is None else OUT_DIR / "test" / name     # tests never mix in
    out_dir.mkdir(parents=True, exist_ok=True)
    stop = n_tiles if limit is None else min(n_tiles, limit)
    t0 = time.time()
    done = 0
    for k0 in range(0, stop, CHUNK):
        path = out_dir / f"chunk_{k0:05d}.npz"
        if path.exists() and limit is None:
            continue
        rows = []
        for k in range(k0, min(k0 + CHUNK, stop)):
            lat0, lon0 = (float(x) for x in vec_to_latlon(centres[k]))
            e, n, u = enu_basis(lat0, lon0)
            if dem is None:
                v = u * R + e * off_e[..., None] + n * off_n[..., None]
                v /= np.linalg.norm(v, axis=-1, keepdims=True)
                lat = np.degrees(np.arcsin(v[..., 2]))
                lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
                del v
                img = (wac.sample(lat, lon, gsd) / 255.0).astype(np.float32)
                del lat, lon
            else:
                img = shaded_relief(dem, lat0, lon0, gsd, size, cfg["sun_az"], cfg["sun_el"], dev)
            done += 1
            if img.std() < 1e-3:
                continue
            d = detect(model, img)
            d = d[(d[:, 3] >= 0.3)]
            if not len(d):
                continue
            x = ((d[:, 0] - c) * ax[0] + (d[:, 1] - c) * ay[0]) * gsd
            y = ((d[:, 0] - c) * ax[1] + (d[:, 1] - c) * ay[1]) * gsd
            p = u * R + e * x[:, None] + n * y[:, None]
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            diam = 2 * d[:, 2] * gsd * (p @ u)              # gnomonic stretch, first order
            _, own = owner.query(p)
            keep = (own == k) & (diam >= cfg["dmin"]) & (diam <= cfg["dmax"])
            if keep.any():
                la, lo = vec_to_latlon(p[keep])
                rows.append(np.column_stack([la, lo, diam[keep], d[keep, 3]]))
        a = np.concatenate(rows) if rows else np.zeros((0, 4))
        np.savez(path, a=a)
        if dev == "cuda":
            torch.cuda.empty_cache()
        rate = (time.time() - t0) / max(done, 1)
        left = sum(1 for j in range(k0 + CHUNK, stop, CHUNK) if not (out_dir / f"chunk_{j:05d}.npz").exists())
        print(f"pass {name}: chunk {k0 // CHUNK + 1}/{-(-stop // CHUNK)} ({len(a):,} craters) "
              f"{rate:.2f} s/tile, eta {left * CHUNK * rate / 60:.0f} min", flush=True)
    print(f"pass {name}: done ({n_tiles} tiles)", flush=True)


def load_pass(name):
    files = sorted((OUT_DIR / name).glob("chunk_*.npz"))
    a = np.concatenate([np.load(f)["a"] for f in files]) if files else np.zeros((0, 4))
    return a


def match_by_size(cat, P, D, bin_w=0.3):
    """For each point (unit vector P, diameter D km): the catalog crater of similar size
    (|ln D ratio| < 0.4) whose centre is within D/2, nearest first; -1 if none.

    Searching only among craters of similar size matters: the nearest centres to a large crater
    are often small craters on its floor."""
    from scipy.spatial import cKDTree
    cb = np.floor(np.log(cat.diam) / bin_w).astype(np.int64)
    trees = {int(k): (np.nonzero(cb == k)[0],) for k in np.unique(cb)}
    trees = {k: (ids[0], cKDTree(cat.xyz[ids[0]])) for k, ids in trees.items()}
    qb = np.floor(np.log(D) / bin_w).astype(np.int64)
    best = -np.ones(len(P), np.int64)
    best_d = np.full(len(P), np.inf)
    for k in np.unique(qb):
        q = np.nonzero(qb == k)[0]
        ub = float(D[q].max()) / 2 / R
        for kk in (k - 1, k, k + 1):
            if int(kk) not in trees:
                continue
            ids, tree = trees[int(kk)]
            dist, j = tree.query(P[q], k=1, distance_upper_bound=ub, workers=-1)
            ok = np.isfinite(dist)
            jj = np.where(ok, j, 0)
            cid = ids[np.minimum(jj, len(ids) - 1)]
            good = ok & (dist * R < D[q] / 2) & (np.abs(np.log(cat.diam[cid] / D[q])) < 0.4) & (dist < best_d[q])
            best[q[good]] = cid[good]
            best_d[q[good]] = dist[good]
    return best


def merge(score_single_small=0.5, score_single=0.6):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    from lunarsolve.catalog import Catalog
    from lunarsolve.geometry import latlon_to_vec, vec_to_latlon
    cat = Catalog()
    parts, pid = [], []
    for i, name in enumerate(PASSES):
        a = load_pass(name)
        print(f"pass {name}: {len(a):,} detections")
        parts.append(a)
        pid.append(np.full(len(a), i, np.int16))
    a = np.concatenate(parts)
    pid = np.concatenate(pid)
    P = latlon_to_vec(a[:, 0], a[:, 1])
    D = a[:, 2].astype(np.float64)
    S = a[:, 3]
    # 1. diameter scale of each pass relative to Robbins (matched craters, D >= 2 km)
    mr = match_by_size(cat, P, D)
    for i, name in enumerate(PASSES):
        sel = (pid == i) & (mr >= 0) & (D >= 2.0)
        sc = float(np.median(D[sel] / cat.diam[mr[sel]])) if sel.sum() > 50 else 1.0
        D[pid == i] /= sc
        print(f"pass {name}: D(detector)/D(Robbins) = {sc:.3f} ({sel.sum():,} matched of {(pid == i).sum():,})")
    # 2. cluster detections of all passes
    tree = cKDTree(P)
    pairs = tree.query_ball_point(P, 0.35 * D / R, workers=-1)
    ii = np.repeat(np.arange(len(P)), [len(x) for x in pairs])
    jj = np.concatenate([np.asarray(x, np.int64) for x in pairs])
    ok = (ii < jj) & (np.abs(np.log(D[jj] / D[ii])) < 0.35)
    g = coo_matrix((np.ones(ok.sum()), (ii[ok], jj[ok])), shape=(len(P), len(P)))
    n_cl, lab = connected_components(g, directed=False)
    print(f"{len(P):,} detections -> {n_cl:,} clusters")
    order = np.argsort(lab, kind="stable")
    bounds = np.concatenate([[0], np.nonzero(np.diff(lab[order]))[0] + 1, [len(order)]])
    w = S[order]
    cs = lambda v: np.add.reduceat(v[order] * w[:, None] if v.ndim == 2 else v[order] * w, bounds[:-1])
    wsum = np.add.reduceat(w, bounds[:-1])
    cP = cs(P) / wsum[:, None]
    cP /= np.linalg.norm(cP, axis=1, keepdims=True)
    cD = np.exp(cs(np.log(D)) / wsum)
    cS = np.maximum.reduceat(S[order], bounds[:-1])
    # number of distinct passes per cluster
    key = lab.astype(np.int64) * 64 + pid
    uk = np.unique(key)
    n_pass = np.bincount((uk // 64).astype(np.int64), minlength=n_cl)
    keep = (n_pass >= 2) | ((cD < 4.0) & (cS >= score_single_small)) | (cS >= score_single)
    cP, cD, cS, n_pass = cP[keep], cD[keep], cS[keep], n_pass[keep]
    print(f"kept {keep.sum():,} clusters (>= 2 passes: {(n_pass >= 2).sum():,})")
    # 3. merge with Robbins
    rid = match_by_size(cat, cP, cD)
    hit = rid >= 0
    seen = np.zeros(len(cat.lat), bool)
    npass_r = np.zeros(len(cat.lat), np.int16)
    score_r = np.zeros(len(cat.lat), np.float32)
    seen[rid[hit]] = True
    np.maximum.at(npass_r, rid[hit], n_pass[hit].astype(np.int16))
    np.maximum.at(score_r, rid[hit], cS[hit].astype(np.float32))
    new = ~hit
    nlat, nlon = vec_to_latlon(cP[new])
    lat = np.concatenate([cat.lat, nlat])
    lon = np.concatenate([cat.lon, nlon])
    diam = np.concatenate([cat.diam, cD[new]]).astype(np.float32)
    source = np.concatenate([np.zeros(len(cat.lat), np.int8), np.ones(new.sum(), np.int8)])
    seen_all = np.concatenate([seen, np.ones(new.sum(), bool)])
    n_pass_all = np.concatenate([npass_r, n_pass[new].astype(np.int16)])
    score_all = np.concatenate([score_r, cS[new].astype(np.float32)])
    np.savez(CN_CATALOG, lat=lat, lon=lon, diam=diam, source=source, seen=seen_all, n_pass=n_pass_all,
             score=score_all)
    print(f"catalog: {len(lat):,} craters = {len(cat.lat):,} Robbins ({seen.sum():,} seen by the detector) "
          f"+ {new.sum():,} new -> {CN_CATALOG}")
    for lo, hi in ((0.5, 1), (1, 2), (2, 5), (5, 20), (20, 100), (100, 1000)):
        rb = (cat.diam >= lo) & (cat.diam < hi)
        nw = (cD[new] >= lo) & (cD[new] < hi)
        print(f"  D {lo:>5}-{hi:<5} km: Robbins {rb.sum():>9,} (seen {seen[rb].mean() if rb.any() else 0:5.1%}), new {nw.sum():>9,}")


def build_index():
    from lunarsolve.catalog import Catalog
    from lunarsolve.index import S0, N_LEVELS, build
    cat = Catalog(CN_CATALOG)
    z = np.load(CN_CATALOG)
    seen = z["seen"][cat.order]
    build(cat, s0=S0 / 2 ** 1.5, n_levels=N_LEVELS + 3, dmin_floor=0.6, path=CN_INDEX, use=seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["detect", "merge", "index"])
    ap.add_argument("--pass", dest="passes", nargs="*", default=None, help="detect: only these passes")
    ap.add_argument("--weights", default=str(Path(__file__).resolve().parent.parent / "weights" / "craternet_v3.pt"))
    ap.add_argument("--limit", type=int, default=None, help="detect: only the first N tiles (testing)")
    args = ap.parse_args()
    if args.step == "detect":
        for name in args.passes or list(PASSES):
            run_pass(name, args.weights, args.limit)
    elif args.step == "merge":
        merge()
    else:
        build_index()


if __name__ == "__main__":
    main()
