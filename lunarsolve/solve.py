"""Blind lunar plate solver.

Pipeline
--------
1. Detect craters (x, y, r) with CraterNet.
2. Take the N largest detections (N grows 8 -> 40), enumerate quads, compute their codes
   for both image parities and look them up in the quad index KD-tree.
3. Each code match gives a similarity transform image -> local tangent plane.  A cheap
   local check against the catalog filters it; survivors are grown outward from the quad
   (similarity -> affine -> homography) and scored over the whole image with a
   log-likelihood ratio (matched detections vs. chance coincidences), astrometry.net style.
4. Fit a full pinhole camera to all matches and report camera latitude, longitude,
   altitude, pointing and focal length.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .catalog import Catalog
from .config import MOON_RADIUS_KM as R
from .geometry import (enu_basis, gnomonic_fwd, gnomonic_inv, nadir_camera, pixel_rays, project,
                       ray_sphere, vec_to_latlon)
from .index import QuadIndex
from .quads import code_variants, enumerate_quads

LOG_R_TOL = 0.4      # radius agreement for a detection-catalog match (ln units)
P_MATCH = 0.6        # P(detection has a catalog counterpart | correct solution)
CAT_COMPLETE_KM = 2.0  # only detections at least this big (implied diameter) can be scored


@dataclass
class Solution:
    camera_lat: float
    camera_lon: float
    altitude_km: float
    altitude_sigma_km: float
    focal_px: float
    focal_estimated: bool
    fov_deg: float
    center_lat: float
    center_lon: float
    boresight_lat: float
    boresight_lon: float
    gsd_km: float
    heading_deg: float
    off_nadir_deg: float
    mirrored: bool
    n_matches: int
    log_odds: float
    rms_px: float
    solve_time_s: float
    C: np.ndarray = field(repr=False)
    Rwc: np.ndarray = field(repr=False)
    shape: tuple = field(repr=False)
    matches: np.ndarray = field(repr=False)        # (K, 2): detection index, catalog index

    def pixel_to_latlon(self, u, v):
        H, W = self.shape
        u = np.asarray(u, float)
        if self.mirrored:
            u = W - 1 - u
        d = pixel_rays(u, np.asarray(v, float), self.C, self.Rwc, self.focal_px, (W - 1) / 2, (H - 1) / 2)
        p, hit = ray_sphere(self.C, d)
        lat, lon = vec_to_latlon(p)
        return np.where(hit, lat, np.nan), np.where(hit, lon, np.nan)

    def to_dict(self):
        keys = ["camera_lat", "camera_lon", "altitude_km", "altitude_sigma_km", "focal_px", "focal_estimated",
                "fov_deg", "center_lat", "center_lon", "boresight_lat", "boresight_lon", "gsd_km", "heading_deg",
                "off_nadir_deg", "mirrored", "n_matches", "log_odds", "rms_px", "solve_time_s"]
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if not isinstance(v, int) else v)
                for k, v in ((k, getattr(self, k)) for k in keys)}

    def summary(self):
        alt = f"{self.altitude_km:.1f} km"
        if self.focal_estimated:
            alt += f" +/- {self.altitude_sigma_km:.1f} (focal length estimated)"
        return (f"camera        lat {self.camera_lat:+.4f}  lon {self.camera_lon:.4f}  altitude {alt}\n"
                f"image centre  lat {self.center_lat:+.4f}  lon {self.center_lon:.4f}  "
                f"GSD {self.gsd_km * 1000:.0f} m/px  heading {self.heading_deg:.1f} deg  "
                f"off-nadir {self.off_nadir_deg:.1f} deg\n"
                f"focal {self.focal_px:.0f} px (FOV {self.fov_deg:.2f} deg)  mirrored={self.mirrored}\n"
                f"{self.n_matches} crater matches, rms {self.rms_px:.2f} px, log-odds {self.log_odds:.0f}, "
                f"{self.solve_time_s * 1000:.0f} ms")


# ----------------------------------------------------------------------------- 2-D models
# All map image pixels (u, v) <-> gnomonic tangent-plane km (x east, y north) at (lat0, lon0).

class Similarity:
    """w = a z + b, with z = u - i*p*v (p=+1 normal, -1 mirrored) and w = x + i y."""

    def __init__(self, a, b, p, lat0, lon0):
        self.a, self.b, self.p, self.lat0, self.lon0 = a, b, p, lat0, lon0

    def img_to_tan(self, uv):
        w = self.a * (uv[:, 0] - 1j * self.p * uv[:, 1]) + self.b
        return np.stack([w.real, w.imag], 1)

    def tan_to_img(self, xy):
        z = ((xy[:, 0] + 1j * xy[:, 1]) - self.b) / self.a
        return np.stack([z.real, -self.p * z.imag], 1)

    @property
    def scale(self):
        return abs(self.a)

    @staticmethod
    def fit(uv, xy, p, lat0, lon0):
        z = uv[:, 0] - 1j * p * uv[:, 1]
        t = xy[:, 0] + 1j * xy[:, 1]
        zm, tm = z.mean(), t.mean()
        a = np.sum((t - tm) * np.conj(z - zm)) / np.sum(np.abs(z - zm) ** 2)
        return Similarity(a, tm - a * zm, p, lat0, lon0)


class Affine:
    def __init__(self, A, lat0, lon0):
        self.A = A                                   # (2, 3): xy = A @ [u, v, 1]
        self.Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
        self.lat0, self.lon0 = lat0, lon0

    def img_to_tan(self, uv):
        return uv @ self.A[:, :2].T + self.A[:, 2]

    def tan_to_img(self, xy):
        return xy @ self.Ai[:, :2].T + self.Ai[:, 2]

    @staticmethod
    def fit(uv, xy, lat0, lon0):
        X = np.column_stack([uv, np.ones(len(uv))])
        A, *_ = np.linalg.lstsq(X, xy, rcond=None)
        return Affine(A.T, lat0, lon0)


class Homography:
    def __init__(self, Hm, lat0, lon0):
        self.H, self.Hi, self.lat0, self.lon0 = Hm, np.linalg.inv(Hm), lat0, lon0

    @staticmethod
    def _apply(M, pts):
        q = np.column_stack([pts, np.ones(len(pts))]) @ M.T
        return q[:, :2] / q[:, 2:3]

    def img_to_tan(self, uv):
        return self._apply(self.H, uv)

    def tan_to_img(self, xy):
        return self._apply(self.Hi, xy)

    @staticmethod
    def fit(uv, xy, lat0, lon0):
        def norm(p):
            m = p.mean(0)
            s = math.sqrt(2) / max(np.mean(np.linalg.norm(p - m, axis=1)), 1e-9)
            return np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
        T1, T2 = norm(uv), norm(xy)
        a = Homography._apply(T1, uv)
        b = Homography._apply(T2, xy)
        n = len(a)
        A = np.zeros((2 * n, 9))
        A[0::2, 0:2] = -a
        A[0::2, 2] = -1
        A[0::2, 6:8] = b[:, :1] * a
        A[0::2, 8] = b[:, 0]
        A[1::2, 3:5] = -a
        A[1::2, 5] = -1
        A[1::2, 6:8] = b[:, 1:] * a
        A[1::2, 8] = b[:, 1]
        _, _, Vt = np.linalg.svd(A)
        Hm = np.linalg.inv(T2) @ Vt[-1].reshape(3, 3) @ T1
        return Homography(Hm / Hm[2, 2], lat0, lon0)


def local_scale(model, uv):
    """km per pixel (area-equivalent) of a 2-D model at an image point."""
    pts = np.array([uv, uv + [1.0, 0], uv + [0, 1.0]], float)
    q = model.img_to_tan(pts)
    d1, d2 = q[1] - q[0], q[2] - q[0]
    return math.sqrt(abs(d1[0] * d2[1] - d1[1] * d2[0]))


def fit_model(uv, xy, p, lat0, lon0):
    """Similarity / affine / homography depending on support, with outlier trimming."""
    keep = np.ones(len(uv), bool)
    for _ in range(3):
        n = keep.sum()
        # a homography needs support spread over the image, not a tight cluster
        spread = np.sqrt(np.linalg.eigvalsh(np.cov(uv[keep].T))[0]) if n >= 3 else 0.0
        if n >= 25 and spread > 80:
            m = Homography.fit(uv[keep], xy[keep], lat0, lon0)
        elif n >= 8:
            m = Affine.fit(uv[keep], xy[keep], lat0, lon0)
        else:
            m = Similarity.fit(uv[keep], xy[keep], p, lat0, lon0)
        res = np.linalg.norm(m.img_to_tan(uv) - xy, axis=1)
        s = 1.4826 * np.median(res[keep]) + 1e-9
        new = res < 3.5 * s
        if new.sum() < 4 or (new == keep).all():
            break
        keep = new
    return m


# ----------------------------------------------------------------------------- solver

class Solver:
    def __init__(self, catalog=None, index=None, model=None, device="cuda"):
        t = time.time()
        self.cat = catalog or Catalog()
        self.index = index or QuadIndex()
        self._model = model
        self.device = device
        _ = self.cat.tree, self.cat.rim_height
        self.load_time = time.time() - t
        self.stats = {}

    @property
    def model(self):
        if self._model is None:
            from .detect import load_model
            self._model = load_model(device=self.device)
        return self._model

    def detect(self, img, thresh=0.3):
        from .detect import detect
        return detect(self.model, img, thresh=thresh, device=self.device)

    # --------------------------------------------------------------------- matching

    def _view_catalog(self, model, shape, rmin_px, km_per_px, center_px=None, radius_px=None):
        """Catalog craters (indices, image x/y/r) predicted inside the image, or inside a
        disc of it when center_px/radius_px are given."""
        H, W = shape
        if center_px is None:
            pts = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [(W - 1) / 2, (H - 1) / 2]], float)
        else:
            c = np.asarray(center_px, float)
            pts = np.array([c + [radius_px, 0], c - [radius_px, 0], c + [0, radius_px], c - [0, radius_px], c])
        xy = model.img_to_tan(pts)
        lat, lon = gnomonic_inv(xy[:, 0], xy[:, 1], model.lat0, model.lon0)
        rad = np.max(np.hypot(xy[:4, 0] - xy[4, 0], xy[:4, 1] - xy[4, 1])) * 1.15
        rad = min(rad, 0.5 * math.pi * R)
        idx = self.cat.query_cap(float(lat[4]), float(lon[4]), rad, dmin=2 * rmin_px * km_per_px * 0.6)
        if len(idx) == 0:
            return idx, np.zeros((0, 3))
        x, y, ok = gnomonic_fwd(self.cat.lat[idx], self.cat.lon[idx], model.lat0, model.lon0)
        idx = idx[ok]
        uv = model.tan_to_img(np.stack([x[ok], y[ok]], 1))
        inside = (uv[:, 0] > -0.5) & (uv[:, 0] < W - 0.5) & (uv[:, 1] > -0.5) & (uv[:, 1] < H - 0.5)
        if center_px is not None:
            inside &= np.hypot(uv[:, 0] - center_px[0], uv[:, 1] - center_px[1]) < radius_px
        r = self.cat.diam[idx] / 2 / km_per_px
        return idx[inside], np.column_stack([uv[inside], r[inside]])

    @staticmethod
    def _match(dets, cat_uvr, dist_frac=0.5, min_px=2.0):
        """Greedy one-to-one matching. Returns (pairs (K,2), tolerance radius per det)."""
        tol = np.maximum(dist_frac * dets[:, 2], min_px)
        if len(cat_uvr) == 0 or len(dets) == 0:
            return np.zeros((0, 2), np.int64), tol
        tree = cKDTree(cat_uvr[:, :2])
        cand = tree.query_ball_point(dets[:, :2], tol)
        used = set()
        pairs = []
        for i, cs in enumerate(cand):
            best, bd = -1, np.inf
            for j in cs:
                if j in used:
                    continue
                lr = abs(math.log(cat_uvr[j, 2] / dets[i, 2]))
                if lr > LOG_R_TOL:
                    continue
                d = math.hypot(cat_uvr[j, 0] - dets[i, 0], cat_uvr[j, 1] - dets[i, 1]) / tol[i] + lr
                if d < bd:
                    best, bd = j, d
            if best >= 0:
                used.add(best)
                pairs.append((i, best))
        return np.array(pairs, np.int64).reshape(-1, 2), tol

    @staticmethod
    def _log_odds(dets, cat_uvr, pairs, tol, area, exclude=()):
        """Sum over detections of log P(data | match model) / P(data | chance)."""
        lr_cat = np.sort(np.log(cat_uvr[:, 2])) if len(cat_uvr) else np.zeros(0)
        matched = np.zeros(len(dets), bool)
        matched[pairs[:, 0]] = True
        lr = np.log(dets[:, 2])
        n_sim = np.searchsorted(lr_cat, lr + LOG_R_TOL) - np.searchsorted(lr_cat, lr - LOG_R_TOL)
        p_bg = np.clip(n_sim * math.pi * tol ** 2 / area, 1e-4, 0.9)
        terms = np.where(matched, np.log(P_MATCH / p_bg), np.log((1 - P_MATCH) / (1 - p_bg)))
        if len(exclude):
            terms[list(exclude)] = 0.0
        return float(terms.sum())

    def _evaluate(self, model, dets, shape, km, center_px=None, radius_px=None):
        # Craters below the catalog's completeness limit carry no evidence either way.
        valid = np.nonzero(dets[:, 2] * 2 * km >= CAT_COMPLETE_KM)[0]
        if len(valid) < 3:
            return -np.inf, np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
        d = dets[valid]
        idx, cat_uvr = self._view_catalog(model, shape, d[:, 2].min(), km, center_px, radius_px)
        pairs, tol = self._match(d, cat_uvr)
        area = float(shape[0] * shape[1]) if radius_px is None else math.pi * radius_px ** 2
        llr = self._log_odds(d, cat_uvr, pairs, tol, area)
        pairs[:, 0] = valid[pairs[:, 0]]
        return llr, pairs, idx

    def _pairs_xy(self, model, idx, pairs):
        c = idx[pairs[:, 1]]
        x, y, _ = gnomonic_fwd(self.cat.lat[c], self.cat.lon[c], model.lat0, model.lon0)
        return np.stack([x, y], 1)

    # --------------------------------------------------------------------- main entry

    def solve(self, img=None, dets=None, shape=None, focal_px=None, fov_deg=None, parity="both",
              max_time=20.0, code_tol=0.08, accept_log_odds=30.0, verbose=False):
        """Blind solve.  Give either an image, or detections (N, >=3: x, y, r[, score]) plus shape.

        Query quads come from two sources, interleaved cheapest-first:
          * the N globally largest detections (N = 8 ... 40), like astrometry.net's brightest stars;
          * a local rule that mirrors how the index was built: at each pixel scale, a grid of cells,
            and in each cell the few largest nearby detections of a suitable size.
        """
        t0 = time.time()
        if dets is None:
            dets = self.detect(img)
            shape = np.asarray(img).shape[:2]
        H, W = shape
        if fov_deg is not None and focal_px is None:
            focal_px = (W / 2) / math.tan(math.radians(fov_deg) / 2)
        if dets.shape[1] > 3:
            dets = dets[dets[:, 3] >= 0.35]
        dets = dets[np.argsort(-dets[:, 2])][:, :3].astype(np.float64)
        self.stats = dict(n_hyp=0, n_pre=0, n_det=len(dets), t_detect=time.time() - t0)
        self._accept = accept_log_odds
        if len(dets) < 6:
            return None
        parities = {"both": (1, -1), "normal": (1,), "mirrored": (-1,)}[parity]
        tried = set()
        seen_quads = set()

        def batches():
            prev = 0
            local = self._local_quads(dets, shape)
            for N in (8, 11, 14, 18, 23, 30, 40):
                n = min(N, len(dets))
                if n > prev:
                    for p in parities:
                        codes, qidx, cost = enumerate_quads(dets[:n, 0].copy(), -p * dets[:n, 1],
                                                            dets[:n, 2].copy(), n, 6.0, 1e9, 500000)
                        sel = qidx.max(1) >= prev
                        order = np.argsort(cost[sel], kind="stable")
                        yield f"top{n}", p, codes[sel][order], qidx[sel][order]
                    prev = n
                if N >= 14 and local:          # interleave one local-rule scale per round
                    qs = local.pop(0)
                    for p in parities:
                        yield "local", p, *self._codes_for(dets, qs, p)
            while local:
                qs = local.pop(0)
                for p in parities:
                    yield "local", p, *self._codes_for(dets, qs, p)

        for name, p, codes, qidx in batches():
            if len(codes) == 0:
                continue
            keep = []
            for k in range(len(qidx)):
                key = (p, tuple(sorted(qidx[k].tolist())))
                if key not in seen_quads:
                    seen_quads.add(key)
                    keep.append(k)
            if not keep:
                continue
            codes, qidx = codes[keep], qidx[keep]
            qcodes, qperm = [], []
            for k in range(len(codes)):
                for c, perm in code_variants(codes[k], code_tol):
                    qcodes.append(c)
                    qperm.append((k, perm))
            qcodes = np.array(qcodes)
            hits = self.index.tree.query_ball_point(qcodes, code_tol, workers=-1)
            hq = np.repeat(np.arange(len(hits)), [len(h) for h in hits])
            if len(hq) == 0:
                continue
            hj = np.concatenate([np.asarray(h, np.int64) for h in hits])
            # Verify the closest code matches first: chance matches fill the tolerance ball
            # uniformly in 8-D, so they pile up near its edge while true matches sit near 0.
            hd = np.linalg.norm(self.index.codes[hj] - qcodes[hq], axis=1)
            for o in np.argsort(hd, kind="stable"):
                k, perm = qperm[hq[o]]
                j = int(hj[o])
                if (p, j) in tried:
                    continue
                tried.add((p, j))
                self.stats["n_hyp"] += 1
                di = qidx[k][list(perm)]
                res = self._try_hypothesis(dets, di, self.index.ids[j], p, shape, accept_log_odds)
                if res is not None:
                    sol = self._finish(dets, res, shape, focal_px, t0)
                    if sol is not None:
                        self.stats["batch"] = name
                        if verbose:
                            print(f"solved: {self.stats}")
                        return sol
                if time.time() - t0 > max_time:
                    if verbose:
                        print(f"timeout: {self.stats}")
                    return None
        if verbose:
            print(f"no solution: {self.stats}")
        return None

    @staticmethod
    def _codes_for(dets, quads, p):
        """Codes (canonical order) for explicit detection quads."""
        codes, idx = [], []
        for q in quads:
            c, qi, _ = enumerate_quads(dets[q, 0].copy(), -p * dets[q, 1], dets[q, 2].copy(), 4, 0.0, 1e12, 4)
            if len(c):
                codes.append(c[0])
                idx.append(np.asarray(q)[qi[0]])
        if not codes:
            return np.zeros((0, 8)), np.zeros((0, 4), np.int64)
        return np.array(codes), np.array(idx)

    @staticmethod
    def _local_quads(dets, shape):
        """Mirror of the index construction in pixel space, one list of quads per scale
        (largest scale first).  Pixel scales replace km scales since the GSD is unknown."""
        from .index import M_NEAR, Q_PER_CELL
        H, W = shape
        tree = cKDTree(dets[:, :2])
        out = []
        s = max(H, W) / 1.5
        while s >= 40:
            s_hi = s * math.sqrt(2)
            ok = (dets[:, 2] >= s / 60) & (dets[:, 2] <= 0.35 * s_hi)
            spacing = s / 1.5
            xs = np.arange(spacing / 2, W, spacing)
            ys = np.arange(spacing / 2, H, spacing)
            quads = []
            seen = set()
            for y in ys:
                for x in xs:
                    nb = np.array(sorted(tree.query_ball_point([x, y], 0.6 * s_hi)), np.int64)
                    nb = nb[ok[nb]][:M_NEAR]                    # dets are sorted largest first
                    if len(nb) < 4:
                        continue
                    codes, qi, cost = enumerate_quads(dets[nb, 0].copy(), -dets[nb, 1], dets[nb, 2].copy(),
                                                      len(nb), s, s_hi, 4096)
                    for o in np.argsort(cost)[:Q_PER_CELL]:
                        q = tuple(sorted(nb[qi[o]].tolist()))
                        if q not in seen:
                            seen.add(q)
                            quads.append(list(q))
            if quads:
                out.append(quads)
            s /= math.sqrt(2)
        return out

    def _try_hypothesis(self, dets, di, ci, p, shape, accept):
        mid = self.cat.xyz[ci[0]] + self.cat.xyz[ci[1]]
        lat0, lon0 = vec_to_latlon(mid)
        lat0, lon0 = float(lat0), float(lon0)
        x, y, ok = gnomonic_fwd(self.cat.lat[ci], self.cat.lon[ci], lat0, lon0)
        if not ok.all():
            return None
        model = Similarity.fit(dets[di, :2], np.stack([x, y], 1), p, lat0, lon0)
        km = model.scale
        lr = np.log(self.cat.diam[ci] / 2 / (km * dets[di, 2]))
        if np.abs(np.median(lr)) > 0.35:
            return None
        # cheap local pre-check: the largest detections near the quad (quad itself excluded).
        # Large craters are used because a 4-point similarity cannot place small ones precisely.
        cen = dets[di, :2].mean(0)
        dist = np.hypot(dets[:, 0] - cen[0], dets[:, 1] - cen[1])
        ab = float(np.hypot(*(dets[di[0], :2] - dets[di[1], :2])))
        rad = 1.5 * ab
        local = np.nonzero(dist < rad)[0]
        local = local[~np.isin(local, di)][:30]          # dets are sorted largest first
        if len(local) < 5:
            return None
        llr, pairs, _ = self._evaluate(model, dets[local], shape, km, cen, rad)
        if len(pairs) < 3 or llr < 3:
            return None
        self.stats["n_pre"] += 1
        # Grow the matched region outward from the quad.  Each stage uses only the K largest
        # detections in its region (big craters tolerate the current model's extrapolation
        # error) and the model is upgraded (similarity -> affine -> homography) as support grows.
        for mult, K in ((2.0, 40), (4.0, 80), (8.0, 120), (None, 200), (None, 400)):
            if mult is None:
                sel = np.arange(min(len(dets), K))
                c_, r_ = None, None
            else:
                sel = np.nonzero(dist < mult * ab)[0][:K]
                c_, r_ = cen, mult * ab
            if len(sel) < 6:
                continue
            llr, pairs, idx = self._evaluate(model, dets[sel], shape, km, c_, r_)
            if len(pairs) < 4:
                return None
            model = fit_model(dets[sel][pairs[:, 0], :2], self._pairs_xy(model, idx, pairs), p, lat0, lon0)
            km = local_scale(model, cen)
        llr, pairs, idx = self._evaluate(model, dets[sel], shape, km)
        if llr < accept or len(pairs) < 8:
            return None
        return dict(model=model, llr=llr, pairs=np.column_stack([sel[pairs[:, 0]], pairs[:, 1]]), idx=idx, p=p)

    def _finish(self, dets, res, shape, focal_px, t0):
        H, W = shape
        model, p, llr = res["model"], res["p"], res["llr"]
        pairs, idx = res["pairs"], res["idx"]
        cat_ids = idx[pairs[:, 1]]
        def rematch(cam, pairs, cat_ids, llr, f_fixed, iters=3):
            """Re-match every detection against the catalog projected through the camera, refit."""
            for _ in range(iters):
                C, Rwc, f = cam[0], cam[1], cam[2]
                gsd = float(np.linalg.norm(C)) - R
                gsd = max(gsd, 1e-3) / f
                sel = np.nonzero(dets[:, 2] * 2 * gsd >= CAT_COMPLETE_KM)[0][:600]
                if len(sel) < 8:
                    break
                idx, cat_uvr = self._camera_view_catalog(C, Rwc, f, p, shape, dets[sel, 2].min())
                pr, tol = self._match(dets[sel], cat_uvr)
                if len(pr) < 8:
                    break
                cam2 = self._fit_camera(dets[sel][pr[:, 0]], idx[pr[:, 1]], model, p, shape, f_fixed,
                                        init=(C, Rwc, f))
                if cam2 is None or cam2[6].sum() < cam[6].sum() - max(2, 0.02 * cam[6].sum()):
                    break
                cam, pairs, cat_ids = cam2, np.column_stack([sel[pr[:, 0]], pr[:, 1]]), idx[pr[:, 1]]
                llr = self._log_odds(dets[sel], cat_uvr, pr, tol, float(H * W))
            return cam, pairs, cat_ids, llr

        # Fixed focal length(s): the given one, or a profile over plausible fields of view.
        if focal_px:
            f_list = [focal_px]
        else:
            f_list = [(W / 2) / math.tan(math.radians(fv) / 2) for fv in np.geomspace(0.5, 120, 10)]
        best = None
        for fc in f_list:
            cam = self._fit_camera(dets[pairs[:, 0]], cat_ids, model, p, shape, fc)
            if cam is None:
                continue
            r = rematch(cam, pairs, cat_ids, llr, fc, iters=2)
            if best is None or (r[0][6].sum(), -r[0][5]) > (best[0][6].sum(), -best[0][5]):
                best = r
        if best is None:
            return None
        cam, pairs, cat_ids, llr = best
        if not focal_px:      # free the focal length, starting from the best profile point
            cam2 = self._fit_camera(dets[pairs[:, 0]], cat_ids, model, p, shape, None, init=cam[:3])
            # the free model nests the fixed one; small inlier-count differences are noise
            if cam2 is not None and cam2[6].sum() >= cam[6].sum() - max(2, 0.02 * cam[6].sum()):
                cam, pairs, cat_ids, llr = rematch(cam2, pairs, cat_ids, llr, None, iters=2)
        C, Rwc, f, f_est, alt_sig, rms, keep = cam
        if len(keep) != len(pairs) or llr < self._accept:
            return None
        pairs, cat_ids = pairs[keep], cat_ids[keep]
        mirrored = p < 0
        cx, cy = (W - 1) / 2, (H - 1) / 2
        clat, clon = vec_to_latlon(C)
        alt = float(np.linalg.norm(C) - R)
        uvc = np.array([[cx, cy], [cx + 1, cy], [cx, cy - 1]])
        rays = pixel_rays(uvc[:, 0], uvc[:, 1], C, Rwc, f, cx, cy)
        pts, okc = ray_sphere(C, rays, R + float(np.median(self.cat.rim_height[cat_ids])))
        down = -C / np.linalg.norm(C)
        off = math.degrees(math.acos(np.clip(rays[0] @ down, -1, 1)))
        if okc.all():
            gsd = float(np.linalg.norm(pts[1] - pts[0]))
            up_vec = pts[2] - pts[0]
            lat_c, lon_c = vec_to_latlon(pts[0])
            e, n_, _u = enu_basis(lat_c, lon_c)
            heading = math.degrees(math.atan2(up_vec @ e, up_vec @ n_)) % 360
        else:
            gsd, heading, lat_c, lon_c = float("nan"), float("nan"), np.nan, np.nan
        return Solution(camera_lat=float(clat), camera_lon=float(clon), altitude_km=alt,
                        altitude_sigma_km=float(alt_sig), focal_px=float(f), focal_estimated=f_est,
                        fov_deg=math.degrees(2 * math.atan(W / 2 / f)),
                        center_lat=float(lat_c), center_lon=float(lon_c),
                        boresight_lat=float(lat_c), boresight_lon=float(lon_c), gsd_km=gsd,
                        heading_deg=heading, off_nadir_deg=off, mirrored=mirrored, n_matches=len(pairs),
                        log_odds=float(llr), rms_px=float(rms), solve_time_s=time.time() - t0, C=C, Rwc=Rwc,
                        shape=(H, W), matches=np.column_stack([pairs[:, 0], cat_ids]))

    # --------------------------------------------------------------------- camera

    def _camera_view_catalog(self, C, Rwc, f, p, shape, rmin_px):
        """Catalog craters seen by a camera: indices and (u, v, r) in the image's own frame."""
        H, W = shape
        cx, cy = (W - 1) / 2, (H - 1) / 2
        us = np.array([0, W - 1, 0, W - 1, cx], float)
        vs = np.array([0, 0, H - 1, H - 1, cy], float)
        pts, hit = ray_sphere(C, pixel_rays(us, vs, C, Rwc, f, cx, cy))
        if not hit[4]:
            return np.zeros(0, np.int64), np.zeros((0, 3))
        cen = pts[4]
        rad = np.max(np.linalg.norm(pts[:4][hit[:4]] - cen, axis=1)) * 1.1 if hit[:4].all() else 0.5 * math.pi * R
        clat, clon = vec_to_latlon(cen)
        gsd = np.linalg.norm(C - cen) / f
        idx = self.cat.query_cap(float(clat), float(clon), min(rad, 0.5 * math.pi * R), dmin=1.2 * rmin_px * gsd)
        if len(idx) == 0:
            return idx, np.zeros((0, 3))
        P = self.cat.points_km(idx)
        uv, z = project(P, C, Rwc, f, cx, cy)
        rng_km = np.linalg.norm(P - C, axis=1)
        nrm = P / np.linalg.norm(P, axis=1, keepdims=True)
        cos_e = np.sum(nrm * (C - P), axis=1) / rng_km
        r = self.cat.diam[idx] / 2 * f / rng_km * np.sqrt(np.clip(cos_e, 0.05, 1))
        ok = (z > 0) & (cos_e > 0) & (uv[:, 0] > -0.5) & (uv[:, 0] < W - 0.5) & (uv[:, 1] > -0.5) & (uv[:, 1] < H - 0.5)
        uv = uv[ok]
        if p < 0:
            uv[:, 0] = W - 1 - uv[:, 0]
        return idx[ok], np.column_stack([uv, r[ok]])

    def _fit_camera(self, dm, cat_ids, model, p, shape, focal_px, init=None):
        """Pinhole camera (position, attitude, optionally focal length) from 2-D/3-D crater matches.

        With unknown focal length the fit is first profiled over a grid of fixed focal lengths
        (each a well-posed 6-parameter problem), then refined with f free from the best one.
        Altitude and focal length are nearly degenerate for narrow, near-nadir views; the
        returned sigma shows how well the data separate them.
        """
        H, W = shape
        cx, cy = (W - 1) / 2, (H - 1) / 2
        uv = dm[:, :2].astype(np.float64).copy()
        if p < 0:
            uv[:, 0] = W - 1 - uv[:, 0]
        P = self.cat.points_km(cat_ids)
        sig = 0.06 * dm[:, 2] + 0.5
        # initial nadir camera above the image centre; image-up azimuth from the 2-D model
        c_img = np.array([[cx, cy], [cx, cy - 1.0]])
        if p < 0:
            c_img[:, 0] = W - 1 - c_img[:, 0]
        xy_c = model.img_to_tan(c_img)
        lat_c, lon_c = gnomonic_inv(xy_c[:, 0], xy_c[:, 1], model.lat0, model.lon0)
        dxy = xy_c[1] - xy_c[0]
        heading = math.degrees(math.atan2(dxy[0], dxy[1]))
        km_c = local_scale(model, np.array([cx, cy]))

        def make_resid(R0, f_fixed):
            def resid(x):
                Rw = Rotation.from_rotvec(x[3:6]).as_matrix() @ R0
                f = f_fixed if f_fixed else math.exp(x[6])
                pr, z = project(P, x[:3], Rw, f, cx, cy)
                r = ((pr - uv) / sig[:, None]).ravel()
                return np.where(np.repeat(z, 2) > 0, r, 1e3)
            return resid

        def inliers(s):
            r = s.fun[: 2 * len(uv)].reshape(-1, 2)
            return int(np.sum(np.hypot(r[:, 0], r[:, 1]) < 3.0))

        def fit_fixed(f0):
            """Robust PnP (RANSAC) at a fixed focal length, then robust least squares."""
            import cv2
            Pm = P.mean(0)
            K = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1]])
            ok, rvec, tvec, inl = cv2.solvePnPRansac((P - Pm).astype(np.float64), uv.astype(np.float64), K, None,
                                                     reprojectionError=max(3.0, float(np.median(sig)) * 3),
                                                     iterationsCount=200, confidence=0.999,
                                                     flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and inl is not None and len(inl) >= 6:
                R0 = cv2.Rodrigues(rvec)[0]
                C0 = -R0.T @ tvec.ravel() + Pm
                if np.linalg.norm(C0) <= R:
                    C0, R0 = nadir_camera(float(lat_c[0]), float(lon_c[0]), max(km_c * f0, 1.0), heading)
            else:
                C0, R0 = nadir_camera(float(lat_c[0]), float(lon_c[0]), max(km_c * f0, 1.0), heading)
            s = least_squares(make_resid(R0, f0), np.concatenate([C0, np.zeros(3)]), loss="soft_l1",
                              f_scale=2.0, x_scale="jac", max_nfev=200)
            return s, R0, f0, inliers(s)

        est_f = focal_px is None
        try:
            if init is not None:
                C0, R0, f0 = init
                if est_f:
                    lo = math.log((W / 2) / math.tan(math.radians(170) / 2))
                    hi = math.log((W / 2) / math.tan(math.radians(0.05) / 2))
                    x0 = np.concatenate([C0, np.zeros(3), [np.clip(math.log(f0), lo, hi)]])
                    s = least_squares(make_resid(R0, None), x0, loss="soft_l1", f_scale=2.0, x_scale="jac",
                                      max_nfev=200, bounds=([-np.inf] * 6 + [lo], [np.inf] * 6 + [hi]))
                    f = math.exp(s.x[6])
                else:
                    s = least_squares(make_resid(R0, focal_px), np.concatenate([C0, np.zeros(3)]),
                                      loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=200)
                    f = focal_px
            elif not est_f:
                s, R0, f, _ = fit_fixed(focal_px)
            else:
                fovs = np.geomspace(0.5, 120, 12)
                cands = [fit_fixed((W / 2) / math.tan(math.radians(fv) / 2)) for fv in fovs]
                s0, R0, f0, inl0 = max(cands, key=lambda c: (c[3], -c[0].cost))
                lo = math.log((W / 2) / math.tan(math.radians(170) / 2))
                hi = math.log((W / 2) / math.tan(math.radians(0.05) / 2))
                x0 = np.concatenate([s0.x, [math.log(f0)]])
                s1 = least_squares(make_resid(R0, None), x0, loss="soft_l1", f_scale=2.0, x_scale="jac",
                                   max_nfev=300, bounds=([-np.inf] * 6 + [lo], [np.inf] * 6 + [hi]))
                if inliers(s1) >= inl0:
                    s, f = s1, math.exp(s1.x[6])
                else:
                    s, f = s0, f0
        except Exception:
            return None
        C = s.x[:3]
        Rwc = Rotation.from_rotvec(s.x[3:6]).as_matrix() @ R0
        if np.linalg.norm(C) < R:
            return None
        pr, z = project(P, C, Rwc, f, cx, cy)
        err = np.linalg.norm(pr - uv, axis=1)
        keep = (err < np.maximum(3 * sig, 3.0)) & (z > 0)
        if keep.sum() < 6:
            return None
        rms = math.sqrt(np.mean(err[keep] ** 2))
        # altitude uncertainty from the Gauss-Newton covariance (includes f when it is free)
        alt_sig = 0.0
        try:
            J = s.jac
            dof = max(1, J.shape[0] - J.shape[1])
            cov = np.linalg.pinv(J.T @ J) * np.sum(s.fun ** 2) / dof
            g = C / np.linalg.norm(C)
            alt_sig = float(math.sqrt(max(0.0, g @ cov[:3, :3] @ g)))
        except Exception:
            pass
        return C, Rwc, f, est_f and len(s.x) == 7, alt_sig, rms, keep
