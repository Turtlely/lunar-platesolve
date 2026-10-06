"""Spherical geometry, local tangent planes and the pinhole camera model.

Conventions
-----------
Body-fixed frame: +x toward (lat 0, lon 0), +z toward the north pole.
Longitudes are east-positive degrees in [0, 360).
Local ENU frame at a point: e (east), n (north), u (up).
Gnomonic tangent plane: (x, y) km, x east, y north, touching the sphere at the centre.
Camera frame: +x image right, +y image down, +z boresight.  Pixel coords (u, v)
have u right, v down, origin at the top-left pixel centre.
"""
import numpy as np

from .config import MOON_RADIUS_KM as R


def latlon_to_vec(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)
    c = np.cos(lat)
    return np.stack([c * np.cos(lon), c * np.sin(lon), np.sin(lat)], axis=-1)


def vec_to_latlon(v):
    v = np.asarray(v, dtype=np.float64)
    r = np.linalg.norm(v, axis=-1)
    lat = np.degrees(np.arcsin(np.clip(v[..., 2] / r, -1, 1)))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360.0
    return lat, lon


def enu_basis(lat, lon):
    """Return (e, n, u) unit vectors, each shape (..., 3)."""
    la = np.radians(lat)
    lo = np.radians(lon)
    e = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    n = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], axis=-1)
    u = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    return e, n, u


def gnomonic_fwd(lat, lon, lat0, lon0):
    """Project points onto the gnomonic plane tangent at (lat0, lon0). Returns x, y (km)
    and a mask of points on the visible hemisphere."""
    p = latlon_to_vec(lat, lon)
    e, n, u = enu_basis(lat0, lon0)
    d = p @ u
    ok = d > 1e-6
    d = np.where(ok, d, 1e-6)
    return R * (p @ e) / d, R * (p @ n) / d, ok


def gnomonic_inv(x, y, lat0, lon0):
    e, n, u = enu_basis(lat0, lon0)
    x = np.asarray(x, dtype=np.float64)[..., None]
    y = np.asarray(y, dtype=np.float64)[..., None]
    v = u * R + e * x + n * y
    return vec_to_latlon(v)


def angular_dist_deg(lat1, lon1, lat2, lon2):
    a = latlon_to_vec(lat1, lon1)
    b = latlon_to_vec(lat2, lon2)
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1, 1)))


# ----------------------------------------------------------------------------- camera

def rot_from_euler(yaw, pitch, roll):
    """Rotation matrix from small intrinsic angles (radians), used for camera tilt."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def nadir_camera(lat, lon, alt_km, heading_deg=0.0, tilt_deg=0.0, tilt_az_deg=0.0):
    """Camera centre (km, body frame) and world->camera rotation matrix.

    heading_deg: azimuth of the image "up" direction, clockwise from north.
    tilt_deg: off-nadir angle of the boresight; tilt_az_deg: azimuth it leans toward.
    """
    e, n, u = enu_basis(lat, lon)
    C = u * (R + alt_km)
    h = np.radians(heading_deg)
    up_img = np.cos(h) * n + np.sin(h) * e          # image-up direction on the ground
    right_img = np.cross(up_img, u)                  # image-right = up x (outward normal)... see below
    # For a nadir camera looking down (-u), image right must satisfy x = y_down x z.
    # With y_down = -up_img and z = -u: x = (-up_img) x (-u) = up_img x u.
    x_c = right_img
    y_c = -up_img
    z_c = -u
    Rwc = np.stack([x_c, y_c, z_c])                  # rows: camera axes in world coords
    if tilt_deg:
        t = np.radians(tilt_deg)
        a = np.radians(tilt_az_deg)
        lean = np.cos(a) * n + np.sin(a) * e         # boresight leans toward this ground azimuth
        axis = np.cross(-u, lean)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        Rt = np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
        Rwc = Rwc @ Rt.T                             # rotate camera axes by Rt
    return C, Rwc


def project(points, C, Rwc, f, cx, cy):
    """Project body-frame points (N,3 km) into pixels. Returns uv (N,2) and depth (N,)."""
    pc = (points - C) @ Rwc.T
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = cx + f * pc[:, 0] / zs
    v = cy + f * pc[:, 1] / zs
    return np.stack([u, v], axis=1), z


def visible_from(points, C):
    """True for surface points (on the sphere) not hidden behind the limb."""
    nrm = points / np.linalg.norm(points, axis=1, keepdims=True)
    return np.sum((C - points) * nrm, axis=1) > 0


def pixel_rays(u, v, C, Rwc, f, cx, cy):
    d_cam = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u, dtype=np.float64)], axis=-1)
    d = d_cam @ Rwc                                  # camera->world
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def ray_sphere(C, d, radius=R):
    """First intersection of rays C + t d with a sphere. Returns points and hit mask."""
    b = d @ C
    c = C @ C - radius * radius
    disc = b * b - c
    hit = disc >= 0
    t = -b - np.sqrt(np.where(hit, disc, 0))
    hit &= t > 0
    return C + t[..., None] * d, hit
