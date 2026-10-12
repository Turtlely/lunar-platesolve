"""Random camera views of the Moon for testing: catalog-derived fake detections and
rendered images (WAC mosaic texture or shaded LOLA DEM) through a true pinhole camera."""
import math

import numpy as np

from .config import MOON_RADIUS_KM as R
from .geometry import latlon_to_vec, nadir_camera, pixel_rays, project, ray_sphere, vec_to_latlon, visible_from


def random_view(rng, W=1024, H=1024, alt_range=(20, 2000), fov_range=(3, 60), tilt_max=20.0, lat_max=88.0,
                lon_range=(0, 360)):
    """Random camera; altitude and FOV are drawn log-uniformly."""
    lat = math.degrees(math.asin(rng.uniform(-math.sin(math.radians(lat_max)), math.sin(math.radians(lat_max)))))
    lon = rng.uniform(*lon_range)
    alt = math.exp(rng.uniform(*np.log(alt_range)))
    fov = math.exp(rng.uniform(*np.log(fov_range)))
    f = (W / 2) / math.tan(math.radians(fov) / 2)
    tilt = rng.uniform(0, tilt_max)
    C, Rwc = nadir_camera(lat, lon, alt, rng.uniform(0, 360), tilt, rng.uniform(0, 360))
    return dict(lat=lat, lon=lon, alt=alt, fov=fov, f=f, C=C, Rwc=Rwc, W=W, H=H, tilt=tilt)


def visible_craters(cat, view, rmin_px=3.0):
    """Catalog craters inside the image: (idx, uv, r_px)."""
    C, Rwc, f, W, H = view["C"], view["Rwc"], view["f"], view["W"], view["H"]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    # footprint: rays through the corners and centre
    us = np.array([0, W - 1, 0, W - 1, cx], float)
    vs = np.array([0, 0, H - 1, H - 1, cy], float)
    d = pixel_rays(us, vs, C, Rwc, f, cx, cy)
    pts, hit = ray_sphere(C, d)
    if not hit[4]:
        return None
    if hit.all():
        cen = pts[4]
        rad = np.max(np.linalg.norm(pts[:4] - cen, axis=1)) * 1.1
    else:  # limb in view: whole visible hemisphere
        cen = pts[4]
        rad = math.pi * R / 2
    lat0, lon0 = vec_to_latlon(cen)
    idx = cat.query_cap(float(lat0), float(lon0), rad)
    P = cat.points_km(idx)
    vis = visible_from(P, C)
    idx, P = idx[vis], P[vis]
    uv, z = project(P, C, Rwc, f, cx, cy)
    rng_km = np.linalg.norm(P - C, axis=1)
    nrm = P / np.linalg.norm(P, axis=1, keepdims=True)
    cos_e = np.abs(np.sum(nrm * (C - P), axis=1)) / rng_km
    r_px = cat.diam[idx] / 2 * f / rng_km * np.sqrt(np.clip(cos_e, 0.05, 1))
    ok = (z > 0) & (uv[:, 0] > -0.5) & (uv[:, 0] < W - 0.5) & (uv[:, 1] > -0.5) & (uv[:, 1] < H - 0.5) & (r_px >= rmin_px)
    return idx[ok], uv[ok], r_px[ok]


def fake_detections(cat, view, rng, p_detect=0.75, pos_noise=0.08, rad_noise=0.08, false_frac=0.15,
                    rmax_frac=0.25, dmin_km=1.5):
    res = visible_craters(cat, view)
    if res is None:
        return None
    idx, uv, r = res
    W, H = view["W"], view["H"]
    keep = (rng.random(len(idx)) < p_detect) & (r <= rmax_frac * min(W, H)) & (cat.diam[idx] >= dmin_km)
    uv, r, tid = uv[keep], r[keep], idx[keep]
    uv = uv + rng.normal(0, pos_noise, (len(r), 2)) * r[:, None]
    r = r * np.exp(rng.normal(0, rad_noise, len(r)))
    nf = int(false_frac * len(r))
    if nf:
        fr = np.exp(rng.uniform(np.log(3), np.log(max(4, np.percentile(r, 90) if len(r) else 10)), nf))
        fuv = np.column_stack([rng.uniform(0, W, nf), rng.uniform(0, H, nf)])
        uv = np.vstack([uv, fuv])
        r = np.concatenate([r, fr])
        tid = np.concatenate([tid, -np.ones(nf, np.int64)])
    # columns: x, y, r, score, true catalog index (-1 for false detections)
    return np.column_stack([uv, r, np.full(len(r), 0.9), tid])


_LDEM = None


def terrain_hit(C, d, iters=3):
    """Ray / terrain intersection: sphere hit refined with LOLA heights (fixed-point)."""
    global _LDEM
    from .config import LDEM_PPD
    from .terrain import load_ldem
    if _LDEM is None:
        _LDEM = load_ldem()
    Hd, Wd = _LDEM.shape
    pts, hit = ray_sphere(C, d)
    for _ in range(iters):
        lat, lon = vec_to_latlon(pts)
        r = np.clip(((90 - lat) * LDEM_PPD - 0.5).round().astype(np.int64), 0, Hd - 1)
        c = ((lon % 360) * LDEM_PPD - 0.5).round().astype(np.int64) % Wd
        h = _LDEM[r, c].astype(np.float64) * 0.0005
        pts2, hit2 = ray_sphere(C, d, R + h)
        pts = np.where(hit2[..., None], pts2, pts)
        hit = hit & hit2
    return pts, hit


def render_wac(wac, view):
    """Image (H, W) float of the WAC mosaic seen through the camera, on LOLA terrain."""
    C, Rwc, f, W, H = view["C"], view["Rwc"], view["f"], view["W"], view["H"]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    d = pixel_rays(uu, vv, C, Rwc, f, cx, cy)
    pts, hit = terrain_hit(C, d)
    lat, lon = vec_to_latlon(pts)
    gsd = np.linalg.norm(C) - R
    gsd = gsd / f
    img = wac.sample(lat, lon, gsd).astype(np.float32) / 255.0
    img[~hit] = 0
    return img


def render_dem(dem, view, rng, sun_el_deg=None, sun_az_deg=None, albedo=None):
    """Shaded LOLA DEM seen through the camera, with cast shadows and a random sun.

    Shading is computed on a sun-aligned tangent grid around the footprint (as in training),
    then sampled along the camera rays.  Returns (image, sun elevation, sun azimuth).
    """
    import torch
    import torch.nn.functional as F
    from .geometry import enu_basis, gnomonic_fwd
    from .synth import grid_latlon, lunar_lambert_L, sensor_model
    C, Rwc, f, W, H = view["C"], view["Rwc"], view["f"], view["W"], view["H"]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    pts, hit = terrain_hit(C, pixel_rays(uu, vv, C, Rwc, f, cx, cy))
    lat, lon = vec_to_latlon(pts)
    lat0, lon0 = float(lat[H // 2, W // 2]), float(lon[H // 2, W // 2])
    x, y, ok = gnomonic_fwd(lat, lon, lat0, lon0)
    ext = float(np.max(np.hypot(x[hit & ok], y[hit & ok])))
    gsd = max(np.linalg.norm(C - pts[H // 2, W // 2]) / f, dem.res_km[0])
    S1 = int(min(4096, 2 * ext / gsd + 64)) // 2 * 2
    g = 2 * ext / (S1 - 64)
    sun_el = math.radians(sun_el_deg if sun_el_deg is not None else rng.uniform(3, 60))
    sun_az = math.radians(sun_az_deg if sun_az_deg is not None else rng.uniform(0, 360))
    beta = sun_az + math.pi
    ax = np.array([math.sin(beta), math.cos(beta)])
    ay = np.array([math.cos(beta), -math.sin(beta)])
    dev = dem.device
    c1 = (S1 - 1) / 2
    ii = torch.arange(S1, device=dev, dtype=torch.float64) - c1
    gi, gj = torch.meshgrid(ii, ii, indexing="xy")
    xk = (gi * ax[0] + gj * ay[0]) * g
    yk = (gi * ax[1] + gj * ay[1]) * g
    glat, glon = grid_latlon(xk, yk, lat0, lon0)
    h = dem.sample(glat, glon, dem.level_for(g)).double()
    z = h - (xk ** 2 + yk ** 2) / (2 * R)
    zx = torch.zeros_like(z)
    zy = torch.zeros_like(z)
    zx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * g)
    zy[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2 * g)
    nrm = torch.rsqrt(1 + zx ** 2 + zy ** 2)
    mu0 = ((zx * math.cos(sun_el) + math.sin(sun_el)) * nrm).clamp_min(0)
    L = lunar_lambert_L(90 - math.degrees(sun_el))      # approximate: phase for a nadir viewer
    phot = (1 - L) * mu0 + L * 2 * mu0 / (mu0 + nrm + 1e-6)
    Hs = z + (gi + c1) * g * math.tan(sun_el)
    shadow = torch.cummax(Hs, dim=1).values > Hs + 1e-5
    rad = torch.where(shadow, torch.zeros_like(phot), phot).float() + 0.01
    if albedo is not None:
        rad = rad * albedo.sample(glat, glon, albedo.level_for(g)).clamp_min(0.05)
    # sample along camera rays: tangent km -> sun-aligned grid pixel
    gx = (x * ax[0] + y * ax[1]) / g + c1
    gy = (x * ay[0] + y * ay[1]) / g + c1
    grid = torch.tensor(np.stack([(gx + 0.5) / S1 * 2 - 1, (gy + 0.5) / S1 * 2 - 1], -1)[None],
                        dtype=torch.float32, device=dev)
    img = F.grid_sample(rad[None, None], grid, mode="bilinear", align_corners=False)
    img = img * torch.tensor((hit & ok).astype(np.float32), device=dev)[None, None]
    img = sensor_model(img, rng, dev)
    return img[0, 0].cpu().numpy(), math.degrees(sun_el), math.degrees(sun_az)
