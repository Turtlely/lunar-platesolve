"""Training-sample generation.

Two sources, both labelled from the Robbins catalog:
  * synthetic renders of the LOLA DEM with random sun elevation/azimuth, cast
    shadows, albedo variation and sensor effects (GPU);
  * crops of the real LROC WAC mosaic (CPU).
Each sample is a (S, S) float image plus an (N, 3) array of crater (x, y, r) pixels.
Craters outside the trainable radius band are returned separately as "ignore" craters.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from .config import MOON_RADIUS_KM as R
from .geometry import enu_basis, gnomonic_fwd, latlon_to_vec

R_MIN, R_MAX = 3.0, 48.0          # trainable crater radius band (pixels)
CATALOG_DMIN_KM = 1.5             # Robbins is ~complete above 1-2 km


def random_center(rng, max_abs_lat=88.0):
    while True:
        lat = math.degrees(math.asin(rng.uniform(-1, 1)))
        if abs(lat) <= max_abs_lat:
            return lat, rng.uniform(0, 360)


def tangent_basis_torch(lat0, lon0, device):
    e, n, u = enu_basis(lat0, lon0)
    return [torch.tensor(v, dtype=torch.float64, device=device) for v in (e, n, u)]


def grid_latlon(xk, yk, lat0, lon0):
    """Gnomonic inverse in torch: tangent-plane km (H, W) -> lat, lon degrees."""
    e, n, u = tangent_basis_torch(lat0, lon0, xk.device)
    v = u * R + e * xk[..., None] + n * yk[..., None]
    v = v / v.norm(dim=-1, keepdim=True)
    lat = torch.rad2deg(torch.asin(v[..., 2].clamp(-1, 1)))
    lon = torch.rad2deg(torch.atan2(v[..., 1], v[..., 0])) % 360
    return lat, lon


def catalog_labels(cat, lat0, lon0, A, t, gsd, size, extent_km, rmin, rmax, dmin_km):
    """Project catalog craters into an image.

    The image pixel p (col, row) relates to tangent-plane km q by q = A @ (p - t) * gsd... we
    pass the inverse mapping directly: p = Ainv @ q / gsd + t, implemented by caller via A, t
    where p = A @ q + t (A already includes 1/gsd).
    Returns keep (N,3) and ignore (M,3) arrays of (x, y, r) pixels.
    """
    idx = cat.query_cap(lat0, lon0, extent_km, dmin=0.0)
    if len(idx) == 0:
        z = np.zeros((0, 3), np.float32)
        return z, z
    x, y, ok = gnomonic_fwd(cat.lat[idx], cat.lon[idx], lat0, lon0)
    q = np.stack([x, y], 1)
    p = q @ A.T + t
    # Radius: area-preserving scale of A for a circle of diameter D.
    s = math.sqrt(abs(np.linalg.det(A)))
    r = cat.diam[idx] / 2 * s
    inside = ok & (p[:, 0] > -0.5) & (p[:, 0] < size - 0.5) & (p[:, 1] > -0.5) & (p[:, 1] < size - 0.5)
    good = inside & (r >= rmin) & (r <= rmax) & (cat.diam[idx] >= dmin_km)
    near = inside & ~good & (r >= rmin * 0.5) & (r <= rmax * 1.5)
    out = np.concatenate([p, r[:, None]], 1).astype(np.float32)
    return out[good], out[near]


# ----------------------------------------------------------------------------- DEM renders

def smooth_noise(B, size, device, octaves=(4, 8, 16, 32), gen=None):
    out = torch.zeros(B, 1, size, size, device=device)
    for i, o in enumerate(octaves):
        n = torch.randn(B, 1, o, o, device=device, generator=gen)
        out += F.interpolate(n, size=(size, size), mode="bicubic", align_corners=False) / (1.6 ** i)
    return out / out.flatten(1).std(1).view(B, 1, 1, 1).clamp_min(1e-6)


def gaussian_blur(img, sigma):
    """Per-sample gaussian blur; sigma (B,) tensor in pixels."""
    B = img.shape[0]
    k = 7
    xs = torch.arange(k, device=img.device, dtype=img.dtype) - k // 2
    s = sigma.clamp_min(1e-3).view(B, 1)
    w = torch.exp(-0.5 * (xs[None] / s) ** 2)
    w = w / w.sum(1, keepdim=True)
    x = img.reshape(1, B, *img.shape[-2:])
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), w.view(B, 1, 1, k), groups=B)
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), w.view(B, 1, k, 1), groups=B)
    return x.view_as(img)


def sensor_model(img, rng, device):
    """img: (B,1,S,S) radiance in ~[0, 1.5]. Applies blur, exposure, gamma, noise, 8-bit."""
    B = img.shape[0]
    sig = torch.tensor(rng.uniform(0.0, 1.3, B), device=device, dtype=img.dtype)
    img = gaussian_blur(img, sig)
    # exposure: map a random high percentile to a random white level, allow clipping
    p = torch.quantile(img.flatten(1)[:, ::7], 0.995, dim=1).clamp_min(1e-4)
    white = torch.tensor(rng.uniform(0.6, 1.3, B), device=device, dtype=img.dtype)
    img = img / p.view(B, 1, 1, 1) * white.view(B, 1, 1, 1)
    gamma = torch.tensor(rng.uniform(0.6, 1.4, B), device=device, dtype=img.dtype)
    img = img.clamp(0, 1) ** gamma.view(B, 1, 1, 1)
    shot = torch.tensor(rng.uniform(0.0, 0.03, B), device=device, dtype=img.dtype).view(B, 1, 1, 1)
    read = torch.tensor(rng.uniform(0.002, 0.02, B), device=device, dtype=img.dtype).view(B, 1, 1, 1)
    img = img + torch.randn_like(img) * (shot * img.clamp_min(0).sqrt() + read)
    img = (img.clamp(0, 1) * 255).round() / 255
    return img


class DemRenderer:
    def __init__(self, dem, catalog, size=256, device="cuda"):
        self.dem = dem
        self.cat = catalog
        self.size = size
        self.device = device

    def render(self, B, rng, lon_range=(0, 360), gsd_range=(0.45, 4.0), tilt_max_deg=25.0):
        S = self.size
        S1 = int(math.ceil(S * 1.45)) // 2 * 2
        dev = self.device
        imgs = torch.empty(B, 1, S, S, device=dev)
        keeps, ignores, metas = [], [], []
        albedo = 1.0 + smooth_noise(B, S1, dev) * torch.tensor(rng.uniform(0.0, 0.25, B), device=dev).view(B, 1, 1, 1).float()
        albedo = albedo.clamp(0.3, 2.0)
        for b in range(B):
            while True:
                lat0, lon0 = random_center(rng)
                if lon_range[0] <= lon0 < lon_range[1]:
                    break
            gsd = math.exp(rng.uniform(math.log(gsd_range[0]), math.log(gsd_range[1])))
            sun_el = math.radians(rng.choice([rng.uniform(1.5, 20), rng.uniform(5, 45), rng.uniform(10, 75)]))
            sun_az = rng.uniform(0, 2 * math.pi)   # azimuth of the sun, clockwise from north
            beta = sun_az + math.pi                 # grid +x points away from the sun
            ax = np.array([math.sin(beta), math.cos(beta)])      # (east, north)
            ay = np.array([math.cos(beta), -math.sin(beta)])     # image-down, 90 deg clockwise
            c1 = (S1 - 1) / 2
            ii = torch.arange(S1, device=dev, dtype=torch.float64) - c1
            gi, gj = torch.meshgrid(ii, ii, indexing="xy")       # gi: col offset, gj: row offset
            xk = (gi * ax[0] + gj * ay[0]) * gsd
            yk = (gi * ax[1] + gj * ay[1]) * gsd
            lat, lon = grid_latlon(xk, yk, lat0, lon0)
            h = self.dem.sample(lat, lon, self.dem.level_for(gsd)).double()
            rho2 = xk ** 2 + yk ** 2
            z = h - rho2 / (2 * R)
            # gradients along grid axes (km/km)
            zx = torch.zeros_like(z)
            zy = torch.zeros_like(z)
            zx[:, 1:-1] = (z[:, 2:] - z[:, :-2]) / (2 * gsd)
            zy[1:-1, :] = (z[2:, :] - z[:-2, :]) / (2 * gsd)
            nrm = torch.rsqrt(1 + zx ** 2 + zy ** 2)
            mu0 = (-zx * -math.cos(sun_el) + math.sin(sun_el)) * nrm   # n . s with s=(-cos e,0,sin e)
            mu = nrm                                                   # nadir viewer
            L = rng.uniform(0.0, 1.0)
            mu0c = mu0.clamp_min(0)
            phot = (1 - L) * mu0c + L * 2 * mu0c / (mu0c + mu + 1e-6)
            # cast shadows: lit iff H(x) >= max_{x'<x} H(x'), H = z + x*gsd*tan(e)
            Hs = z + (gi + c1) * gsd * math.tan(sun_el)
            shadow = torch.cummax(Hs, dim=1).values > Hs + 1e-5
            ambient = rng.uniform(0.0, 0.04)
            rad = torch.where(shadow, torch.zeros_like(phot), phot).float()
            rad = rad * albedo[b, 0] + ambient * albedo[b, 0]
            # random final orientation, mild anisotropy (oblique view), crop to S
            rot = rng.uniform(0, 2 * math.pi)
            tilt = math.radians(rng.uniform(0, tilt_max_deg)) if rng.random() < 0.5 else 0.0
            taz = rng.uniform(0, math.pi)
            Rr = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
            Rt = np.array([[math.cos(taz), -math.sin(taz)], [math.sin(taz), math.cos(taz)]])
            Ds = np.diag([math.cos(tilt), 1.0])
            M = Rr @ Rt @ Ds @ Rt.T               # grid1 offsets -> final offsets
            Minv = np.linalg.inv(M)
            c = (S - 1) / 2
            th = torch.tensor(Minv * (S / S1), dtype=torch.float32, device=dev)  # align_corners=False approx
            theta = torch.cat([th, torch.zeros(2, 1, device=dev)], 1)[None]
            grid = F.affine_grid(theta, (1, 1, S, S), align_corners=False)
            imgs[b] = F.grid_sample(rad[None, None], grid, mode="bilinear", align_corners=False)[0]
            # labels: tangent km q -> grid1 offsets o = [ax;ay] q / gsd -> final p = M o + c
            Pg = np.stack([ax, ay]) / gsd
            A = M @ Pg
            keep, ign = catalog_labels(self.cat, lat0, lon0, A, np.array([c, c]), gsd, S,
                                       extent_km=S * gsd * 1.2, rmin=R_MIN, rmax=R_MAX, dmin_km=0.0)
            keeps.append(keep)
            ignores.append(ign)
            metas.append(dict(lat=lat0, lon=lon0, gsd=gsd, sun_el=math.degrees(sun_el)))
        imgs = sensor_model(imgs, rng, dev)
        return imgs, keeps, ignores, metas


# ----------------------------------------------------------------------------- WAC crops

def wac_sample(wac, cat, rng, size=256, lon_range=(0, 360), gsd_range=(0.25, 3.0), tilt_max_deg=25.0,
               max_abs_lat=80.0):
    S = size
    while True:
        lat0, lon0 = random_center(rng, max_abs_lat)
        if lon_range[0] <= lon0 < lon_range[1]:
            break
    gsd = math.exp(rng.uniform(math.log(gsd_range[0]), math.log(gsd_range[1])))
    rot = rng.uniform(0, 2 * math.pi)
    tilt = math.radians(rng.uniform(0, tilt_max_deg)) if rng.random() < 0.5 else 0.0
    taz = rng.uniform(0, math.pi)
    Rr = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    Rt = np.array([[math.cos(taz), -math.sin(taz)], [math.sin(taz), math.cos(taz)]])
    Ds = np.diag([math.cos(tilt), 1.0])
    flipy = np.diag([1.0, -1.0])              # north-up tangent -> image rows go down
    A = Rr @ Rt @ Ds @ Rt.T @ flipy / gsd     # tangent km -> pixel offsets
    Ainv = np.linalg.inv(A)
    c = (S - 1) / 2
    jj, ii = np.mgrid[0:S, 0:S].astype(np.float64)
    off = np.stack([ii - c, jj - c], -1) @ Ainv.T
    e, n, u = enu_basis(lat0, lon0)
    v = u * R + e * off[..., 0:1] + n * off[..., 1:2]
    v /= np.linalg.norm(v, axis=-1, keepdims=True)
    lat = np.degrees(np.arcsin(v[..., 2]))
    lon = np.degrees(np.arctan2(v[..., 1], v[..., 0])) % 360
    img = wac.sample(lat, lon, gsd) / 255.0
    keep, ign = catalog_labels(cat, lat0, lon0, A, np.array([c, c]), gsd, S, extent_km=S * gsd * 1.2,
                               rmin=R_MIN, rmax=R_MAX, dmin_km=CATALOG_DMIN_KM)
    # Craters below the catalog completeness limit are unknown: ignore anything below it.
    return img.astype(np.float32), keep, ign, dict(lat=lat0, lon=lon0, gsd=gsd)
