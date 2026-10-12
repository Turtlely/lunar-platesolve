"""Limb (horizon) geometry for oblique views.

The limb of a sphere seen from a point C is the set of tangent rays, a cone around the
direction a to the sphere's centre with half-angle rho, sin(rho) = R / |C|.  Fitting that
cone to the limb pixels gives, in the camera frame, the nadir direction (so pitch and
roll), the altitude, and, from the arc's curvature, the focal length.  With the nadir
known, the image can be re-projected to a virtual camera at the same point looking
straight down: a pure rotation, so a homography, no surface model needed.  Craters in
the rectified view are round again and the normal solver applies.
"""
import math

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from .config import MOON_RADIUS_KM as R

LIMB_RADIUS_KM = R + 1.5      # the silhouette is formed by the higher terrain along the limb


# ----------------------------------------------------------------------------- detection

def limb_points(img, dark=0.06, min_frac=0.01, margin=4):
    """Sub-pixel points on the boundary between the lit disk and dark space.

    img: float image normalised to ~[0, 1].  Space is the dark region(s) touching the
    image border with a sizeable area.  Returns ((N, 2) x, y points, space mask) or (None, None).
    """
    H, W = img.shape
    img = img.astype(np.float32) / max(float(np.percentile(img, 99.5)), 1e-6)
    sm = cv2.GaussianBlur(img, (0, 0), 1.0)
    space = (sm < dark).astype(np.uint8)
    space = cv2.morphologyEx(space, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(space, connectivity=8)
    keep = np.zeros_like(space)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        touches = x == 0 or y == 0 or x + w == W or y + h == H
        if touches and area >= min_frac * H * W:
            keep[lab == i] = 1
    if not keep.any():
        return None, None
    near_space = cv2.dilate(keep, np.ones((3, 3), np.uint8)).astype(bool)
    edge = near_space & ~keep.astype(bool) & (sm > 2.5 * dark)     # lit pixels touching space
    edge[:margin] = edge[-margin:] = False
    edge[:, :margin] = edge[:, -margin:] = False
    ys, xs = np.nonzero(edge)
    if len(xs) < 30:
        return None, None
    # sub-pixel: along the intensity gradient, find the half-way crossing between disk and space
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)[ys, xs]
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)[ys, xs]
    gn = np.hypot(gx, gy) + 1e-9
    nx, ny = gx / gn, gy / gn                                       # points from space into the disk
    t = np.arange(-5, 6, 0.5)
    px = xs[:, None] + t[None] * nx[:, None]
    py = ys[:, None] + t[None] * ny[:, None]
    prof = map_coordinates(sm, [py.ravel(), px.ravel()], order=1, mode="nearest").reshape(px.shape)
    lo = prof[:, :4].mean(1)
    hi = prof[:, -4:].mean(1)
    mid = (lo + hi) / 2
    above = prof >= mid[:, None]
    k = np.argmax(above, axis=1)                                     # first sample past half-way
    k = np.clip(k, 1, len(t) - 1)
    i = np.arange(len(k))
    p0, p1 = prof[i, k - 1], prof[i, k]
    frac = np.clip((mid - p0) / np.where(np.abs(p1 - p0) < 1e-6, 1e-6, p1 - p0), 0, 1)
    tc = t[k - 1] + frac * (t[1] - t[0])
    good = (hi - lo > 2 * dark) & above[i, -1] & ~above[i, 0]
    out = np.column_stack([xs + tc * nx, ys + tc * ny])[good]
    # The disk's edge is convex: real limb points lie on the convex hull of the lit region,
    # while shadow edges that happen to connect to space (near the terminator) lie inside it.
    lit = (~keep.astype(bool)) & (sm > 2.5 * dark)
    ly, lx = np.nonzero(lit[::2, ::2])
    if len(lx) < 10:
        return None, None
    hull = cv2.convexHull(np.column_stack([lx * 2, ly * 2]).astype(np.float32))
    dist = np.array([cv2.pointPolygonTest(hull, (float(x), float(y)), True) for x, y in out])
    out = out[np.abs(dist) < 4.0]
    return (out, keep.astype(bool)) if len(out) >= 30 else (None, None)


# ----------------------------------------------------------------------------- cone fit

def _rays(pts, f, cx, cy):
    d = np.column_stack([(pts[:, 0] - cx) / f, (pts[:, 1] - cy) / f, np.ones(len(pts))])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _cone_lsq(d):
    """Least-squares cone through unit rays: d . w = 1, a = w/|w|, cos(rho) = 1/|w|."""
    w, *_ = np.linalg.lstsq(d, np.ones(len(d)), rcond=None)
    nw = np.linalg.norm(w)
    return w / nw, math.acos(min(1.0, 1.0 / nw))


def _cone_ransac(d, tol_rad, rng, iters=200):
    best = None
    n = len(d)
    for _ in range(iters):
        s = rng.choice(n, 3, replace=False)
        try:
            w = np.linalg.solve(d[s], np.ones(3))
        except np.linalg.LinAlgError:
            continue
        nw = np.linalg.norm(w)
        if nw <= 1:
            continue
        a = w / nw
        rho = math.acos(1 / nw)
        res = np.abs(np.arccos(np.clip(d @ a, -1, 1)) - rho)
        inl = res < tol_rad
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 20:
        return None
    for _ in range(3):
        a, rho = _cone_lsq(d[best])
        res = np.abs(np.arccos(np.clip(d @ a, -1, 1)) - rho)
        best = res < tol_rad
    a, rho = _cone_lsq(d[best])
    res = np.abs(np.arccos(np.clip(d[best] @ a, -1, 1)) - rho)
    return a, rho, best, float(np.sqrt(np.mean(res ** 2)))


def fit_limb(pts, shape, focal_px=None, tol_px=1.5, seed=0):
    """Fit the limb cone.  With unknown focal length, profile over the field of view.

    Returns dict(a, rho, f, altitude_km, inliers, rms_px, arc_deg, tilt_deg) or None.
    a is the unit direction from the camera to the Moon's centre, in the camera frame
    (x right, y down, z boresight).
    """
    H, W = shape
    cx, cy = (W - 1) / 2, (H - 1) / 2
    rng = np.random.default_rng(seed)

    def fit_at(f):
        d = _rays(pts, f, cx, cy)
        r = _cone_ransac(d, tol_px / f, rng)
        if r is None:
            return None
        a, rho, inl, rms = r
        return dict(a=a, rho=rho, f=f, inliers=inl, rms_px=rms * f, n=int(inl.sum()))

    if focal_px:
        best = fit_at(focal_px)
    else:
        cands = [fit_at((W / 2) / math.tan(math.radians(fv) / 2)) for fv in np.geomspace(15, 130, 30)]
        cands = [c for c in cands if c is not None]
        if not cands:
            return None
        nmax = max(c["n"] for c in cands)
        good = [c for c in cands if c["n"] >= 0.9 * nmax]
        best = min(good, key=lambda c: c["rms_px"])
        # golden-section refinement of f on the inlier set
        sel = pts[best["inliers"]]

        def rms_f(lf):
            a, rho = _cone_lsq(_rays(sel, math.exp(lf), cx, cy))
            d = _rays(sel, math.exp(lf), cx, cy)
            return float(np.sqrt(np.mean((np.arccos(np.clip(d @ a, -1, 1)) - rho) ** 2))) * math.exp(lf)
        lo, hi = math.log(best["f"]) - 0.25, math.log(best["f"]) + 0.25
        g = (math.sqrt(5) - 1) / 2
        x1, x2 = hi - g * (hi - lo), lo + g * (hi - lo)
        f1, f2 = rms_f(x1), rms_f(x2)
        for _ in range(30):
            if f1 < f2:
                hi, x2, f2 = x2, x1, f1
                x1 = hi - g * (hi - lo)
                f1 = rms_f(x1)
            else:
                lo, x1, f1 = x1, x2, f2
                x2 = lo + g * (hi - lo)
                f2 = rms_f(x2)
        best = fit_at(math.exp((lo + hi) / 2)) or best
    if best is None or not (0.05 < best["rho"] < math.pi / 2 - 1e-3):   # degenerate cone (noise, not a limb)
        return None
    a, rho, f = best["a"], best["rho"], best["f"]
    alt = LIMB_RADIUS_KM / math.sin(rho) - R
    d = _rays(pts[best["inliers"]], f, cx, cy)
    # angular extent of the limb arc around the cone axis
    e1 = np.cross(a, [0.0, 0.0, 1.0])
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-6 else np.array([1.0, 0, 0])
    e2 = np.cross(a, e1)
    az = np.unwrap(np.sort(np.arctan2(d @ e2, d @ e1)))
    gaps = np.diff(np.concatenate([az, [az[0] + 2 * math.pi]]))
    arc = math.degrees(2 * math.pi - gaps.max())
    tilt = math.degrees(math.acos(np.clip(a[2], -1, 1)))
    return dict(a=a, rho=rho, f=f, altitude_km=alt, inliers=best["inliers"], rms_px=best["rms_px"],
                arc_deg=arc, tilt_deg=tilt, focal_estimated=focal_px is None)


# ----------------------------------------------------------------------------- rectification

def rectify(img, fit, max_emission_deg=70.0, max_side=2048, space_mask=None):
    """Re-project the image to a virtual camera at the same point looking at nadir.

    Returns (image, info) where info has M (virtual axes in camera frame, columns),
    f_v, pp_v (principal point of the virtual image), H_vr (virtual px -> real px) and valid.
    """
    Hh, Ww = img.shape
    f = fit["f"]
    cx, cy = (Ww - 1) / 2, (Hh - 1) / 2
    a = fit["a"]
    z = a
    x = np.array([1.0, 0, 0]) - a[0] * a
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0, 1.0, 0]) - a[1] * a
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.column_stack([x, y, z])                      # ray_cam = M @ ray_virtual
    sin_rho = math.sin(fit["rho"])
    sin_emax = math.sin(math.radians(max_emission_deg))

    def usable(d):
        cos_t = d @ a
        sin_t = np.sqrt(np.clip(1 - cos_t ** 2, 0, 1))
        return (cos_t > 0) & (sin_t < sin_rho) & (sin_t / sin_rho < sin_emax)

    # extent of the usable part of the real image in the virtual (nadir) image plane
    gy, gx = np.mgrid[0:Hh:4, 0:Ww:4]
    d = _rays(np.column_stack([gx.ravel(), gy.ravel()]).astype(float), f, cx, cy)
    ok = usable(d)
    if space_mask is not None:
        ok &= ~space_mask[gy.ravel(), gx.ravel()]
    if ok.sum() < 100:
        return None, None
    dv = d[ok] @ M                                      # rays in the virtual frame
    q = dv[:, :2] / dv[:, 2:3]
    q0, q1 = q.min(0), q.max(0)
    f_v = min(f, max_side / max(q1 - q0))
    Wv = int(math.ceil((q1[0] - q0[0]) * f_v)) + 1
    Hv = int(math.ceil((q1[1] - q0[1]) * f_v)) + 1
    pp = (-q0[0] * f_v, -q0[1] * f_v)
    Kr = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    Kv_inv = np.array([[1 / f_v, 0, -pp[0] / f_v], [0, 1 / f_v, -pp[1] / f_v], [0, 0, 1]])
    H_vr = Kr @ M @ Kv_inv                              # virtual pixel -> real pixel
    fill = float(np.median(img[img > 0.05])) if (img > 0.05).any() else 0.5
    out = cv2.warpPerspective(img.astype(np.float32), H_vr, (Wv, Hv), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
    # validity of each virtual pixel
    vy, vx = np.mgrid[0:Hv, 0:Wv]
    rv = np.stack([(vx - pp[0]) / f_v, (vy - pp[1]) / f_v, np.ones_like(vx, dtype=float)], -1).reshape(-1, 3)
    rc = rv @ M.T
    rc /= np.linalg.norm(rc, axis=1, keepdims=True)
    uvr = rc[:, :2] / rc[:, 2:3] * f + [cx, cy]
    valid = usable(rc) & (rc[:, 2] > 0) & (uvr[:, 0] >= 0) & (uvr[:, 0] <= Ww - 1) & (uvr[:, 1] >= 0) & (uvr[:, 1] <= Hh - 1)
    valid = valid.reshape(Hv, Wv)
    if space_mask is not None:
        sm = cv2.warpPerspective(space_mask.astype(np.uint8), H_vr, (Wv, Hv),
                                 flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=1).astype(bool)
        valid &= ~sm
    out[~valid] = fill
    return out, dict(M=M, f_v=f_v, pp_v=pp, H_vr=H_vr, valid=valid, f=f)


def map_points(H_vr, uv):
    q = np.column_stack([uv, np.ones(len(uv))]) @ H_vr.T
    return q[:, :2] / q[:, 2:3]


def nadir_scale_prior(alt_km, f_v, pp):
    """Expected ground scale (km per pixel, area-equivalent) in a nadir-looking pinhole view of
    the sphere from altitude alt_km, as a function of pixel position.  Returns a callable."""
    D = R + alt_km

    def km_per_px(uv):
        uv = np.atleast_2d(np.asarray(uv, float))
        rpx = np.hypot(uv[:, 0] - pp[0], uv[:, 1] - pp[1])
        th = np.arctan(rpx / f_v)
        s = np.clip(D * np.sin(th) / R, 0, 0.999999)
        phi = np.arcsin(s) - th
        radial = R * (D * np.cos(th) / (R * np.cos(phi + th)) - 1) * np.cos(th) ** 2 / f_v
        tangential = np.where(rpx > 1e-6, R * np.sin(phi) / np.maximum(rpx, 1e-6), radial)
        return np.sqrt(np.abs(radial * tangential))

    return km_per_px
