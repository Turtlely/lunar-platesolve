"""Fast CPU tests for geometry and quad codes (no data files needed).  Run: python -m pytest tests"""
import math

import numpy as np

from lunarsolve.geometry import (gnomonic_fwd, gnomonic_inv, latlon_to_vec, nadir_camera, pixel_rays, project,
                                 ray_sphere, vec_to_latlon)
from lunarsolve.quads import code_variants, enumerate_quads


def test_gnomonic_roundtrip():
    rng = np.random.default_rng(0)
    lat0, lon0 = 23.0, 301.0
    x = rng.uniform(-300, 300, 100)
    y = rng.uniform(-300, 300, 100)
    lat, lon = gnomonic_inv(x, y, lat0, lon0)
    x2, y2, ok = gnomonic_fwd(lat, lon, lat0, lon0)
    assert ok.all()
    assert np.allclose(x, x2, atol=1e-6) and np.allclose(y, y2, atol=1e-6)


def test_nadir_camera_orientation():
    # heading 0: image up = north, image right = east
    C, Rwc = nadir_camera(10.0, 50.0, 100.0)
    f, W = 1000.0, 1001
    c = (W - 1) / 2
    north = latlon_to_vec(10.1, 50.0) * 1737.4
    east = latlon_to_vec(10.0, 50.1) * 1737.4
    uv, z = project(np.stack([north, east]), C, Rwc, f, c, c)
    assert (z > 0).all()
    assert uv[0, 1] < c and abs(uv[0, 0] - c) < 1e-6 * f + 1e-3      # north is up
    assert uv[1, 0] > c and abs(uv[1, 1] - c) < 1.0                  # east is right


def test_ray_projection_roundtrip():
    C, Rwc = nadir_camera(-40.0, 200.0, 300.0, heading_deg=37, tilt_deg=15, tilt_az_deg=80)
    f, W, H = 2000.0, 800, 600
    cx, cy = (W - 1) / 2, (H - 1) / 2
    u = np.array([0.0, 400.0, 799.0])
    v = np.array([0.0, 300.0, 599.0])
    p, hit = ray_sphere(C, pixel_rays(u, v, C, Rwc, f, cx, cy))
    assert hit.all()
    uv, _ = project(p, C, Rwc, f, cx, cy)
    assert np.allclose(uv, np.stack([u, v], 1), atol=1e-6)
    lat, lon = vec_to_latlon(p)
    assert np.all(np.abs(lat) <= 90)


def _codes(pts, r):
    c, idx, _ = enumerate_quads(pts[:, 0].copy(), pts[:, 1].copy(), r.copy(), len(pts), 0.0, 1e12, 10000)
    return {tuple(sorted(i)): (code, i) for code, i in zip(c, idx)}


def test_codes_similarity_invariant():
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 100, (7, 2))
    r = rng.uniform(1, 5, 7)
    a = _codes(pts, r)
    th, s, t = 1.1, 3.7, np.array([40.0, -12.0])
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    b = _codes(pts @ Rm.T * s + t, r * s)
    assert a.keys() == b.keys() and len(a) > 0
    for k in a:
        ca, ia = a[k]
        cb, ib = b[k]
        assert np.allclose(ca, cb, atol=1e-9)
        assert (ia == ib).all()                  # same canonical crater order


def test_code_variants_cover_relabelling():
    # a quad sitting on the xC == xD boundary must still be found after a tiny perturbation
    pts = np.array([[0, 0], [10, 0], [5, 2.0], [5.0001, -2.0]])
    r = np.array([1.0, 1.2, 0.8, 0.9])
    c1, i1, _ = enumerate_quads(pts[:, 0].copy(), pts[:, 1].copy(), r, 4, 0, 1e9, 10)
    pts2 = pts.copy()
    pts2[3, 0] = 4.9999
    c2, i2, _ = enumerate_quads(pts2[:, 0].copy(), pts2[:, 1].copy(), r, 4, 0, 1e9, 10)
    variants = [v for v, _ in code_variants(c2[0], 0.01)]
    assert min(np.linalg.norm(v - c1[0]) for v in variants) < 1e-3


def test_limb_cone_fit_recovers_tilt_and_altitude():
    from lunarsolve.limb import LIMB_RADIUS_KM, fit_limb
    W = H = 1000
    f = 700.0
    alt = 120.0
    rho = math.asin(LIMB_RADIUS_KM / (1737.4 + alt))                  # limb seen from altitude alt
    tilt = math.radians(40)
    a = np.array([0.0, -math.sin(tilt), math.cos(tilt)])              # Moon centre, camera frame
    e1 = np.cross(a, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    phi = np.linspace(-2, 2, 400)
    d = math.cos(rho) * a + math.sin(rho) * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
    d = d[d[:, 2] > 0]
    uv = d[:, :2] / d[:, 2:3] * f + [(W - 1) / 2, (H - 1) / 2]
    uv = uv[(uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H)]
    fit = fit_limb(uv, (H, W), f)
    assert abs(fit["tilt_deg"] - 40) < 0.05
    assert abs(fit["altitude_km"] - alt) < 0.5
