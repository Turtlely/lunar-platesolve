"""Far views (telescope from Earth, a spacecraft thousands of km away): the disk edge fixes the geometry.

Seen from far away, the limb is a circle in the image with centre c and radius r (pixels).  The
camera looks at the Moon's centre from distance D; with focal length f,

    sin(rho) = R / D,     r = f tan(rho)        (rho: angular radius of the disk)

Every pixel inside the circle then maps to one point of the sphere: intersect its ray with the
sphere.  In camera axes (x right, y down, z boresight) that point has the unit vector n(x, y).
A catalog crater with body-frame unit vector p appears where n = R_wc p.  With a few matched
craters, the rotation R_wc follows in closed form (Kabsch / Wahba: SVD of sum w_i p_i n_i^T), and
because the geometry is the exact sphere, a local match extends correctly to the whole disk.

Distance: D follows from r and f.  Without the focal length (field of view), a far view does not
determine D (the image changes by < 0.1% between 50,000 and 400,000 km); the solution then uses
D = 384,400 km for drawing and reports the distance as not measured.
"""
import math

import cv2
import numpy as np

from .config import MOON_RADIUS_KM as R

EARTH_MOON_KM = 384_400.0


# ----------------------------------------------------------------------------- disk edge

def disk_circle(img, min_arc_deg=20.0, max_rms=None, rng_seed=0):
    """Fit the lunar limb circle.  Works on dark or bright (blue) sky.

    The background is the border-touching region below an Otsu threshold.  Its boundary with the
    disk contains the sunlit limb (a sharp circle) and, on a partly lit Moon, the terminator
    (soft, not on that circle).  RANSAC keeps the circle with the most sharp boundary points.
    Returns dict(cx, cy, r, inliers, arc_deg, rms) or None.  max_rms default: max(2 px, 0.08% of r)
    (a close-up of the limb shows mountains on it).
    """
    H, W = img.shape
    x = img.astype(np.float32)
    x = (x - np.percentile(x, 0.5)) / max(float(np.percentile(x, 99.5) - np.percentile(x, 0.5)), 1e-6)
    sm = cv2.GaussianBlur(np.clip(x, 0, 1), (0, 0), 1.5)
    u8 = (sm * 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bg = (u8 < thr).astype(np.uint8)
    bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bg, connectivity=8)
    keep = np.zeros_like(bg)
    for i in range(1, n):
        x0, y0, w, h, area = stats[i]
        if (x0 == 0 or y0 == 0 or x0 + w == W or y0 + h == H) and area >= 0.03 * H * W:
            keep[lab == i] = 1
    if not keep.any():
        return None
    # the background must be smooth (sky), not dark mare
    gmag = np.hypot(cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3))
    if float(np.median(gmag[keep.astype(bool)])) > 0.02:
        return None
    edge = cv2.dilate(keep, np.ones((3, 3), np.uint8)).astype(bool) & ~keep.astype(bool)
    edge[:3] = edge[-3:] = False
    edge[:, :3] = edge[:, -3:] = False
    ys, xs = np.nonzero(edge)
    if len(xs) < 60:
        return None
    g = gmag[ys, xs]
    sharp = g >= np.percentile(g, 40)              # the limb is sharper than the terminator
    P = np.column_stack([xs, ys]).astype(np.float64)
    Ps = P[sharp]
    rng = np.random.default_rng(rng_seed)
    best = None
    for _ in range(600):
        a, b, c = Ps[rng.choice(len(Ps), 3, replace=False)]
        circ = _circle3(a, b, c)
        if circ is None:
            continue
        cx, cy, r = circ
        if not (0.1 * min(H, W) < r < 30 * max(H, W)):
            continue
        d = np.abs(np.hypot(Ps[:, 0] - cx, Ps[:, 1] - cy) - r)
        score = int((d < 2.0).sum())
        if best is None or score > best[0]:
            best = (score, cx, cy, r)
    if best is None or best[0] < 60:
        return None
    _, cx, cy, r = best
    for _ in range(3):                              # refine on inliers (algebraic least squares)
        d = np.abs(np.hypot(Ps[:, 0] - cx, Ps[:, 1] - cy) - r)
        inl = Ps[d < max(2.0, 3 * np.median(d[d < 4]) if (d < 4).any() else 2.0)]
        if len(inl) < 30:
            return None
        cx, cy, r = _circle_lsq(inl)
    d = np.hypot(inl[:, 0] - cx, inl[:, 1] - cy) - r
    rms = float(np.sqrt(np.mean(d ** 2)))
    ang = np.sort(np.arctan2(inl[:, 1] - cy, inl[:, 0] - cx))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * np.pi]]))
    arc = math.degrees(2 * np.pi - gaps.max())
    # the disk must be inside the circle: few disk pixels outside it
    yy, xx = np.nonzero(~keep.astype(bool))
    outside = float(np.mean(np.hypot(xx[::7] - cx, yy[::7] - cy) > r + 4))
    if max_rms is None:
        max_rms = max(2.0, 0.0008 * r)
    if rms > max_rms or arc < min_arc_deg or outside > 0.02 or len(inl) < 150:
        return None
    pts = inl[np.linspace(0, len(inl) - 1, min(len(inl), 400)).astype(int)]
    return dict(cx=float(cx), cy=float(cy), r=float(r), inliers=len(inl), arc_deg=arc, rms=rms, pts=pts)


def _circle3(a, b, c):
    ax, ay = a
    bx, by = b
    cx_, cy_ = c
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy_) + (bx * bx + by * by) * (cy_ - ay) + (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx_ - bx) + (bx * bx + by * by) * (ax - cx_) + (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
    return ux, uy, math.hypot(ax - ux, ay - uy)


def _circle_lsq(P):
    """Algebraic circle fit (Kasa): x^2 + y^2 + D x + E y + F = 0."""
    A = np.column_stack([P[:, 0], P[:, 1], np.ones(len(P))])
    b = -(P[:, 0] ** 2 + P[:, 1] ** 2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2, -E / 2
    return cx, cy, math.sqrt(max(cx * cx + cy * cy - F, 1e-9))


# ----------------------------------------------------------------------------- geometry

class FarCamera:
    """Pinhole camera at distance D from the Moon's centre, boresight through the centre.

    The principal point is the disk centre (the actual boresight direction differs only by a
    shift of the image, which this absorbs).  The focal length follows from the disk radius."""

    def __init__(self, disk, D=EARTH_MOON_KM, focal_px=None):
        self.cx, self.cy, self.r = disk["cx"], disk["cy"], disk["r"]
        self._D0, self._f0 = D, focal_px
        if focal_px:                                   # known focal length: the disk size gives D
            self.f = float(focal_px)
            self.D = R / math.sin(math.atan(self.r / self.f))
            self.D_measured = True
        else:
            self.D = float(D)
            self.f = self.r / math.tan(math.asin(R / self.D))
            self.D_measured = False

    def normals(self, uv):
        """Camera-frame unit vectors of the surface points seen at pixels uv (N, 2); NaN off the disk."""
        d = np.column_stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        # ray from the origin; sphere centre at (0, 0, D)
        b = d[:, 2] * self.D
        disc = b * b - (self.D ** 2 - R * R)
        t = b - np.sqrt(np.where(disc >= 0, disc, np.nan))
        q = d * t[:, None]
        q[:, 2] -= self.D
        return q / R

    def project(self, n):
        """Pixels (N, 2) of camera-frame unit vectors n (N, 3) on the sphere; and the cosine of the
        viewing angle (> 0: visible)."""
        q = n * R
        q[:, 2] += self.D
        uv = np.column_stack([self.cx + self.f * q[:, 0] / q[:, 2], self.cy + self.f * q[:, 1] / q[:, 2]])
        view = -q / np.linalg.norm(q, axis=1, keepdims=True)
        cos_e = np.sum(view * n, axis=1)
        return uv, cos_e

    def with_circle(self, cx, cy, r):
        """Same camera model (known or nominal distance) with another disk circle."""
        return FarCamera(dict(cx=cx, cy=cy, r=r), D=self._D0, focal_px=self._f0)

    def pose(self, Rwc):
        """Camera centre C (body frame, km) and world -> camera rotation for a pinhole Solution."""
        z_world = Rwc.T @ np.array([0.0, 0.0, 1.0])     # boresight in body coordinates
        return -self.D * z_world, Rwc


def refine(cam, Rwc, p_cat, uv_det, sig, limb_pts=None, limb_sig=2.0):
    """Joint robust fit of the rotation and the disk circle to crater matches and limb points.

    A short limb arc (a close-up) fixes the circle only to a few percent, which distorts the
    pixel -> sphere mapping near the limb; the craters pin it down.  Returns (camera, Rwc)."""
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    def model(x):
        return Rotation.from_rotvec(x[:3]).as_matrix() @ Rwc, cam.with_circle(x[3], x[4], math.exp(x[5]))

    def resid(x):
        Rr, c = model(x)
        uv, cos_e = c.project(p_cat @ Rr.T)
        r1 = ((uv - uv_det) / sig[:, None]).ravel()
        r1 = np.where(np.repeat(cos_e, 2) > 0, r1, 50.0)
        if limb_pts is None or not len(limb_pts):
            return r1
        r2 = (np.hypot(limb_pts[:, 0] - x[3], limb_pts[:, 1] - x[4]) - math.exp(x[5])) / limb_sig
        return np.concatenate([r1, r2])

    x0 = np.array([0.0, 0.0, 0.0, cam.cx, cam.cy, math.log(cam.r)])
    sol = least_squares(resid, x0, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=200)
    Rr, c = model(sol.x)
    return c, Rr


def kabsch(p, n, w=None):
    """Rotation Rwc minimising sum w |Rwc p - n|^2 (p body-frame, n camera-frame unit vectors)."""
    w = np.ones(len(p)) if w is None else w
    H = (p * w[:, None]).T @ n
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


# ----------------------------------------------------------------------------- straight-down view

def rectify_region(img, cam, max_side=2048, min_tilt_deg=25.0):
    """Resample the visible part of a far view into a straight-down (nadir) view.

    Near the limb a far view sees craters as thin ellipses, which the detector misses.  The
    surface region in the image is re-projected for a virtual camera above its centre, looking
    straight down, with the image's best resolution (at the disk centre).  Craters are round again.

    Works in the far camera's axes with the Moon's centre at the origin (the far camera is at
    (0, 0, -D)).  Returns (rectified image, info) or (None, None) when the region is seen almost
    face-on already (tilt below min_tilt_deg) or too little of it is on the disk.
    info: f_v, size, Cv (virtual camera centre), Rv (rows: virtual camera axes), all in that frame.
    """
    H, W = img.shape
    gx, gy = np.meshgrid(np.linspace(0, W - 1, 33), np.linspace(0, H - 1, 33))
    n = cam.normals(np.column_stack([gx.ravel(), gy.ravel()]))
    n = n[np.isfinite(n).all(1)]
    if len(n) < 50:
        return None, None
    n0 = n.mean(0)
    n0 /= np.linalg.norm(n0)
    tilt = math.degrees(math.acos(np.clip(-n0[2], -1, 1)))         # angle between view and surface normal
    if tilt < min_tilt_deg:
        return None, None
    theta = float(np.max(np.arccos(np.clip(n @ n0, -1, 1))))        # angular radius of the region
    gsd = (cam.D - R) / cam.f                                        # km/px at the disk centre
    h_v = max(0.25 * R, 3 * theta * R)                               # virtual altitude
    half = math.tan(min(theta * R / h_v, math.radians(60)))          # half-width of the view (tan)
    f_v = h_v / gsd
    size = int(min(max_side, 2 * half * f_v))
    if size < 64:
        return None, None
    f_v = size / (2 * half)
    # virtual camera above n0 looking down; image "down" close to the far camera's y axis
    z = -n0
    y = np.array([0.0, 1.0, 0.0]) - z * z[1]
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    Rv = np.stack([x, y, z])
    Cv = (R + h_v) * n0
    c = (size - 1) / 2
    vv, uu = np.mgrid[0:size, 0:size].astype(np.float64)
    d = ((uu - c) / f_v)[..., None] * x + ((vv - c) / f_v)[..., None] * y + z
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    b = d @ Cv
    disc = b * b - (Cv @ Cv - R * R)
    t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
    P = Cv + t[..., None] * d
    uv, cos_e = cam.project((P / R).reshape(-1, 3))
    ok = (cos_e > 0.02) & np.isfinite(uv).all(1)
    mx = np.where(ok, uv[:, 0], -1).reshape(size, size).astype(np.float32)
    my = np.where(ok, uv[:, 1], -1).reshape(size, size).astype(np.float32)
    out = cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    out[~ok.reshape(size, size)] = 0
    return out, dict(f_v=f_v, size=size, Cv=Cv, Rv=Rv, tilt_deg=tilt, theta_deg=math.degrees(theta))


def virtual_to_image(uvr, info, cam):
    """Detections (x, y, r) of the rectified view -> image coordinates (x, y, r) of the far view."""
    c = (info["size"] - 1) / 2
    x, y, z = info["Rv"]
    d = ((uvr[:, 0] - c) / info["f_v"])[:, None] * x + ((uvr[:, 1] - c) / info["f_v"])[:, None] * y + z
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    Cv = info["Cv"]
    b = d @ Cv
    disc = b * b - (Cv @ Cv - R * R)
    t = -b - np.sqrt(np.where(disc >= 0, disc, np.nan))
    P = Cv + t[:, None] * d
    uv, cos_e = cam.project(P / R)
    # radius: geometric mean of the ellipse axes (as the matcher's catalog radii)
    rng_v = np.linalg.norm(P - Cv, axis=1)
    r_km = uvr[:, 2] * rng_v / info["f_v"]
    rng_o = np.linalg.norm(P - np.array([0.0, 0.0, -cam.D]), axis=1)
    r = r_km * cam.f / rng_o * np.sqrt(np.clip(cos_e, 0.05, 1))
    return np.column_stack([uv, r])
