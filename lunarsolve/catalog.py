"""Robbins (2018) lunar crater catalog: ~1.3 M craters with D >= 1 km."""
import numpy as np
from scipy.spatial import cKDTree

from .config import CATALOG_CSV, CATALOG_NPZ, MOON_RADIUS_KM
from .geometry import latlon_to_vec


def build_npz():
    import pandas as pd
    df = pd.read_csv(CATALOG_CSV, usecols=["LAT_CIRC_IMG", "LON_CIRC_IMG", "DIAM_CIRC_IMG",
                                           "DIAM_ELLI_MAJOR_IMG", "DIAM_ELLI_MINOR_IMG",
                                           "DIAM_ELLI_ANGLE_IMG", "ARC_IMG"])
    np.savez(CATALOG_NPZ,
             lat=df.LAT_CIRC_IMG.values.astype(np.float64),
             lon=(df.LON_CIRC_IMG.values % 360).astype(np.float64),
             diam=df.DIAM_CIRC_IMG.values.astype(np.float32),
             arc=df.ARC_IMG.values.astype(np.float32))


def rim_heights(lat, lon, diam, n=12):
    from .terrain import load_ldem
    from .config import LDEM_PPD
    dem = load_ldem()
    H, W = dem.shape
    out = np.zeros(len(lat), np.float32)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    for s in range(0, len(lat), 200000):
        la = lat[s:s + 200000, None]
        lo = lon[s:s + 200000, None]
        rdeg = np.degrees(diam[s:s + 200000, None] / 2 / MOON_RADIUS_KM)
        pla = la + rdeg * np.cos(ang)
        plo = lo + rdeg * np.sin(ang) / np.maximum(np.cos(np.radians(la)), 0.02)
        r = np.clip(((90 - pla) * LDEM_PPD - 0.5).round().astype(np.int64), 0, H - 1)
        c = ((plo % 360) * LDEM_PPD - 0.5).round().astype(np.int64) % W
        out[s:s + 200000] = dem[r, c].astype(np.float32).mean(1) * 0.0005
    return out


class Catalog:
    def __init__(self, path=CATALOG_NPZ):
        if not path.exists():
            build_npz()
        z = np.load(path)
        self.path = path
        # Largest first.  The quad index and the rim-height cache store row numbers, so this order
        # must be the same on every machine.  The default sort orders equal diameters differently on
        # different CPUs (SIMD code paths), and 70% of the catalog rows have tied diameters, so the
        # order is saved once next to the catalog and loaded from there.
        order_path = path.with_name(path.stem + "_order.npy")
        if order_path.exists():
            order = np.load(order_path).astype(np.int64)
            if len(order) != len(z["diam"]):
                raise ValueError(f"{order_path.name} does not match {path.name}; delete it and rebuild the index")
        else:
            order = np.argsort(-z["diam"], kind="stable")
            np.save(order_path, order.astype(np.int32))
        self.order = order                            # catalog row i = file row order[i]
        self.lat = z["lat"][order]
        self.lon = z["lon"][order]
        self.diam = z["diam"][order].astype(np.float64)
        self.arc = z["arc"][order] if "arc" in z else np.zeros(len(order), np.float32)
        self.source = z["source"][order] if "source" in z else np.zeros(len(order), np.int8)  # 0 Robbins, 1 CraterNet
        self.xyz = latlon_to_vec(self.lat, self.lon)
        self._tree = None
        # Completeness model used by the solver's evidence score.
        #   d_min_km:  smaller detections carry no evidence
        #   d_ramp:    P(catalog counterpart) rises from d_ramp[0] to full at d_ramp[1]
        #   top_km:    "large crater" size for the acceptance coverage test
        if (self.source == 1).any():                    # fine catalog: complete to about 1 km
            self.d_min_km, self.d_ramp, self.top_km = 0.6, (0.45, 1.3), 1.0
        else:                                           # Robbins: complete to about 1-2 km
            self.d_min_km, self.d_ramp, self.top_km = 1.0, (0.7, 2.5), 2.0

    @property
    def rim_height(self):
        """Mean LOLA elevation (km) on each crater's rim circle; cached on disk."""
        if getattr(self, "_rim_h", None) is None:
            from .config import DATA_DIR
            name = "robbins_rim_height.npy" if self.path == CATALOG_NPZ else f"{self.path.stem}_rim_height.npy"
            p = DATA_DIR / name
            if p.exists():
                h = np.load(p)
            else:
                h = rim_heights(self.lat, self.lon, self.diam)
                np.save(p, h)
            self._rim_h = h
        return self._rim_h

    def points_km(self, idx):
        """Body-frame crater centres (km) lifted to their rim elevation."""
        return self.xyz[idx] * (MOON_RADIUS_KM + self.rim_height[idx])[:, None]

    def __len__(self):
        return len(self.lat)

    @property
    def tree(self):
        if self._tree is None:
            self._tree = cKDTree(self.xyz)
        return self._tree

    def query_cap(self, lat, lon, radius_km, dmin=0.0, dmax=np.inf):
        """Indices of craters within a spherical cap, filtered by diameter."""
        chord = 2 * np.sin(min(radius_km / MOON_RADIUS_KM, np.pi) / 2)
        idx = np.array(self.tree.query_ball_point(latlon_to_vec(lat, lon), chord), dtype=np.int64)
        if len(idx):
            d = self.diam[idx]
            idx = idx[(d >= dmin) & (d <= dmax)]
        return np.sort(idx)
