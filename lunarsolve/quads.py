"""Similarity-invariant codes for crater quads (the crater analogue of astrometry.net quads).

For four craters A, B, C, D where A-B is the most distant pair and C, D lie inside the
circle with diameter AB, place A at (0, 0) and B at (1, 0).  The code is

    [xC, yC, xD, yD,  W_R*(ln rA - m), W_R*(ln rB - m), W_R*(ln rC - m),  W_S*(m - ln|AB|)]

with m the mean log radius of the four craters.  The first four numbers are invariant to
translation, rotation and scale of the image; the radius terms add discrimination (the
crater diameter plays the role of stellar brightness, but as a ratio it is a *measured*
invariant, not just an ordering).  Symmetry is broken as in astrometry.net:
xC + xD <= 1 (else swap A/B) and xC <= xD (else swap C/D).
"""
import numpy as np
from numba import njit

W_R = 0.30     # weight on radius ratios: whitens log-radius noise (~0.08) to positional noise (~0.02)
W_S = 0.15     # weight on size/separation term (also absorbs any detector radius bias)
CODE_DIM = 8


@njit(cache=True)
def _code(px, py, pr, ia, ib, ic, id_, out, oidx, k):
    ax, ay = px[ia], py[ia]
    bx, by = px[ib] - ax, py[ib] - ay
    L2 = bx * bx + by * by
    cxr, cyr = px[ic] - ax, py[ic] - ay
    dxr, dyr = px[id_] - ax, py[id_] - ay
    xc = (cxr * bx + cyr * by) / L2
    yc = (bx * cyr - by * cxr) / L2
    xd = (dxr * bx + dyr * by) / L2
    yd = (bx * dyr - by * dxr) / L2
    if xc + xd > 1.0:
        xc, yc, xd, yd = 1.0 - xc, -yc, 1.0 - xd, -yd
        ia, ib = ib, ia
    if xc > xd:
        xc, yc, xd, yd = xd, yd, xc, yc
        ic, id_ = id_, ic
    la, lb, lc, ld = np.log(pr[ia]), np.log(pr[ib]), np.log(pr[ic]), np.log(pr[id_])
    m = 0.25 * (la + lb + lc + ld)
    out[k, 0] = xc
    out[k, 1] = yc
    out[k, 2] = xd
    out[k, 3] = yd
    out[k, 4] = W_R * (la - m)
    out[k, 5] = W_R * (lb - m)
    out[k, 6] = W_R * (lc - m)
    out[k, 7] = W_S * (m - 0.5 * np.log(L2))
    oidx[k, 0] = ia
    oidx[k, 1] = ib
    oidx[k, 2] = ic
    oidx[k, 3] = id_


@njit(cache=True)
def enumerate_quads(px, py, pr, n, dmin, dmax, max_out):
    """All valid quads among the first n points with |AB| in [dmin, dmax).

    Returns codes (K, 8), local indices (K, 4) in canonical order, and a "rank cost"
    (sum of the four indices; points are given largest-first so lower = bigger craters).
    """
    codes = np.empty((max_out, CODE_DIM), np.float64)
    idx = np.empty((max_out, 4), np.int64)
    cost = np.empty(max_out, np.int64)
    k = 0
    for a in range(n):
        for b in range(a + 1, n):
            dx = px[b] - px[a]
            dy = py[b] - py[a]
            L2 = dx * dx + dy * dy
            if L2 < dmin * dmin or L2 >= dmax * dmax:
                continue
            mx = 0.5 * (px[a] + px[b])
            my = 0.5 * (py[a] + py[b])
            r2 = 0.25 * L2
            for c in range(n):
                if c == a or c == b:
                    continue
                if (px[c] - mx) ** 2 + (py[c] - my) ** 2 >= r2:
                    continue
                for d in range(c + 1, n):
                    if d == a or d == b:
                        continue
                    if (px[d] - mx) ** 2 + (py[d] - my) ** 2 >= r2:
                        continue
                    if k >= max_out:
                        return codes[:k], idx[:k], cost[:k]
                    _code(px, py, pr, a, b, c, d, codes, idx, k)
                    cost[k] = a + b + c + d
                    k += 1
    return codes[:k], idx[:k], cost[:k]


def code_variants(code, tol):
    """Codes for every relabelling a noisy measurement could need near a symmetry boundary.

    The canonical order depends on two comparisons (xC + xD vs 1 for A/B, xC vs xD for C/D).
    When either is within 2*tol of its boundary, noise may have flipped it relative to the
    index, so the swapped labelling(s) are queried too.  Returns (code, perm) pairs where
    perm maps canonical (A, B, C, D) positions of `code` to those of the variant.
    """
    xc, yc, xd, yd = code[:4]
    ra, rb, rc = code[4:7]
    rd = -(ra + rb + rc)
    near_ab = abs(xc + xd - 1) < 2 * tol
    near_cd = abs(xc - xd) < 2 * tol
    out = []
    for ab in ((False, True) if near_ab else (False,)):
        for cd in ((False, True) if near_cd else (False,)):
            p = [xc, yc, xd, yd]
            r = [ra, rb, rc, rd]
            perm = [0, 1, 2, 3]
            if ab:
                p = [1 - p[0], -p[1], 1 - p[2], -p[3]]
                r = [r[1], r[0], r[2], r[3]]
                perm = [1, 0, 2, 3]
            if cd:
                p = [p[2], p[3], p[0], p[1]]
                r = [r[0], r[1], r[3], r[2]]
                perm = [perm[0], perm[1], perm[3], perm[2]]
            c = code.copy()
            c[:4] = p
            c[4:7] = r[:3]
            out.append((c, tuple(perm)))
    return out
