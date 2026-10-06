"""Multi-scale quad index over the Robbins catalog.

    python -m lunarsolve.index            # builds DATA_DIR/quad_index.npz

Level l covers quads with |AB| in [S0 * sqrt2^l, S0 * sqrt2^(l+1)) km.  At each level a
Fibonacci lattice of sample points covers the sphere; at each sample point the M largest
nearby craters (of a size suited to the level) form up to Q quads.  This mirrors
astrometry.net's "brightest stars per healpix" selection, with crater diameter in place
of brightness.
"""
import math
import time

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from .catalog import Catalog
from .config import INDEX_PATH, MOON_RADIUS_KM as R
from .quads import CODE_DIM, enumerate_quads

S0 = 8.0
N_LEVELS = 15
M_NEAR = 9
Q_PER_CELL = 3


def level_band(l):
    lo = S0 * math.sqrt(2) ** l
    return lo, lo * math.sqrt(2)


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = i * math.pi * (3 - math.sqrt(5))
    r = np.sqrt(1 - z * z)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)


@njit(cache=True)
def _tangent(c, p):
    ex, ey, ez = -c[1], c[0], 0.0
    nn = math.sqrt(ex * ex + ey * ey)
    if nn < 1e-9:
        ex, ey, ez = 0.0, 1.0, 0.0
        nn = 1.0
    ex /= nn
    ey /= nn
    nx = c[1] * ez - c[2] * ey
    ny = c[2] * ex - c[0] * ez
    nz = c[0] * ey - c[1] * ex
    d = p[0] * c[0] + p[1] * c[1] + p[2] * c[2]
    x = R * (p[0] * ex + p[1] * ey + p[2] * ez) / d
    y = R * (p[0] * nx + p[1] * ny + p[2] * nz) / d
    return x, y


@njit(parallel=True)  # no cache: it inlines quads.enumerate_quads, whose constants may change
def _build_level(cells, ptr, nbr, xyz, rad, M, Q, s_lo, s_hi):
    nc = cells.shape[0]
    out_ids = -np.ones((nc, Q, 4), np.int64)
    out_codes = np.zeros((nc, Q, CODE_DIM), np.float64)
    for ci in prange(nc):
        a, b = ptr[ci], ptr[ci + 1]
        n = min(M, b - a)
        if n < 4:
            continue
        px = np.empty(n)
        py = np.empty(n)
        pr = np.empty(n)
        for k in range(n):
            j = nbr[a + k]
            px[k], py[k] = _tangent(cells[ci], xyz[j])
            pr[k] = rad[j]
        codes, idx, cost = enumerate_quads(px, py, pr, n, s_lo, s_hi, 4096)
        if codes.shape[0] == 0:
            continue
        order = np.argsort(cost)
        for q in range(min(Q, codes.shape[0])):
            o = order[q]
            for t in range(4):
                out_ids[ci, q, t] = nbr[a + idx[o, t]]
            out_codes[ci, q] = codes[o]
    return out_ids, out_codes


def build(cat=None, verbose=True):
    cat = cat or Catalog()
    all_ids, all_codes, all_lvl = [], [], []
    for l in range(N_LEVELS):
        t0 = time.time()
        s_lo, s_hi = level_band(l)
        dmin = max(1.0, s_lo / 30)
        dmax = s_hi * 0.7
        pool = np.nonzero((cat.diam >= dmin) & (cat.diam <= dmax))[0]      # largest first
        spacing = s_lo / 1.5
        n_cells = int(4 * math.pi * R * R / spacing ** 2)
        cells = fibonacci_sphere(n_cells)
        tree = cKDTree(cat.xyz[pool])
        rq = 0.6 * s_hi / R
        nb = tree.query_ball_point(cells, 2 * math.sin(rq / 2), workers=-1, return_sorted=True)
        lens = np.fromiter((len(x) for x in nb), np.int64, len(nb))
        ptr = np.concatenate([[0], np.cumsum(lens)])
        nbr = pool[np.concatenate([np.asarray(x, np.int64) for x in nb])] if ptr[-1] else np.zeros(0, np.int64)
        ids, codes = _build_level(cells, ptr, nbr, cat.xyz, cat.diam / 2, M_NEAR, Q_PER_CELL, s_lo, s_hi)
        ids = ids.reshape(-1, 4)
        codes = codes.reshape(-1, CODE_DIM)
        ok = ids[:, 0] >= 0
        ids, codes = ids[ok], codes[ok]
        # de-duplicate quads found from several cells
        key = np.sort(ids, 1)
        _, first = np.unique(key, axis=0, return_index=True)
        ids, codes = ids[first], codes[first]
        all_ids.append(ids.astype(np.int32))
        all_codes.append(codes.astype(np.float32))
        all_lvl.append(np.full(len(ids), l, np.int8))
        if verbose:
            print(f"level {l:2d} |AB| {s_lo:7.1f}-{s_hi:7.1f} km  pool {len(pool):8d}  cells {n_cells:8d}  "
                  f"quads {len(ids):8d}  {time.time() - t0:.1f}s", flush=True)
    np.savez(INDEX_PATH, ids=np.concatenate(all_ids), codes=np.concatenate(all_codes),
             level=np.concatenate(all_lvl))


class QuadIndex:
    def __init__(self, path=INDEX_PATH):
        z = np.load(path)
        self.ids = z["ids"]
        self.codes = z["codes"]
        self.level = z["level"]
        self.tree = cKDTree(self.codes, leafsize=32, balanced_tree=False, compact_nodes=False)

    def __len__(self):
        return len(self.ids)


if __name__ == "__main__":
    build()
