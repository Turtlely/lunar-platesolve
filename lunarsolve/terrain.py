"""Access to global rasters: the LOLA DEM (GPU pyramid) and the LROC WAC mosaic (CPU memmap)."""
import struct
from pathlib import Path

import numpy as np

from .config import DATA_DIR, LDEM_IMG, LDEM_PPD, WAC_TIF


# ----------------------------------------------------------------------------- LOLA DEM

def load_ldem():
    """LOLA LDEM_64: (5760*2, 23040) int16, DN*0.5 m above 1737.4 km.
    Pixel (l, s) centre: lat = 90 - (l+0.5)/ppd, lon = (s+0.5)/ppd."""
    h = 180 * LDEM_PPD
    w = 360 * LDEM_PPD
    return np.memmap(LDEM_IMG, dtype="<i2", mode="r", shape=(h, w))


class GpuDem:
    """DEM pyramid on the GPU, heights in km, padded by 2 columns for longitude wrap."""

    def __init__(self, device="cuda", levels=6):
        import torch
        import torch.nn.functional as F
        self.device = device
        a = torch.from_numpy(np.asarray(load_ldem()).astype(np.float32) * 0.0005)
        a = a.to(device)[None, None]
        self.levels = []
        self.ppd = []
        for k in range(levels):
            self.levels.append(self._pad(a))
            self.ppd.append(LDEM_PPD / 2 ** k)
            a = F.avg_pool2d(a, 2)
        self.res_km = [2 * np.pi * 1737.4 / 360 / p for p in self.ppd]

    @staticmethod
    def _pad(a):
        import torch
        return torch.cat([a[..., -2:], a, a[..., :2]], dim=-1).contiguous()

    def level_for(self, gsd_km):
        k = 0
        while k + 1 < len(self.levels) and self.res_km[k + 1] <= gsd_km:
            k += 1
        return k

    def sample(self, lat, lon, level):
        """lat, lon: torch tensors (H, W) degrees. Returns heights (H, W) km (bilinear)."""
        import torch
        import torch.nn.functional as F
        t = self.levels[level]
        H, W = t.shape[-2], t.shape[-1] - 4
        ppd = self.ppd[level]
        x = (lon % 360) * ppd - 0.5 + 2
        y = (90 - lat) * ppd - 0.5
        gx = (x + 0.5) / (W + 4) * 2 - 1
        gy = (y + 0.5) / H * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)[None].float()
        return F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)[0, 0]


# ----------------------------------------------------------------------------- WAC mosaic

def _read_geotiff_meta(path):
    """Minimal (Big)TIFF reader for an uncompressed single-band strip image."""
    with open(path, "rb") as fh:
        head = fh.read(16)
        bo = "<" if head[:2] == b"II" else ">"
        big = struct.unpack(bo + "H", head[2:4])[0] == 43
        if big:
            off = struct.unpack(bo + "Q", head[8:16])[0]
            fh.seek(off)
            n = struct.unpack(bo + "Q", fh.read(8))[0]
            raw = fh.read(20 * n)
            ent = [struct.unpack(bo + "HHQ8s", raw[20 * i:20 * i + 20]) for i in range(n)]
        else:
            off = struct.unpack(bo + "I", head[4:8])[0]
            fh.seek(off)
            n = struct.unpack(bo + "H", fh.read(2))[0]
            raw = fh.read(12 * n)
            ent = [struct.unpack(bo + "HHI4s", raw[12 * i:12 * i + 12]) for i in range(n)]
        sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 11: 4, 12: 8, 16: 8}
        fmts = {1: "B", 2: "c", 3: "H", 4: "I", 5: "II", 11: "f", 12: "d", 16: "Q"}
        tags = {}
        for tag, typ, cnt, val in ent:
            nbytes = sizes[typ] * cnt
            if nbytes <= len(val):
                data = val[:nbytes]
            else:
                p = struct.unpack(bo + ("Q" if big else "I"), val[:8 if big else 4])[0]
                fh.seek(p)
                data = fh.read(nbytes)
            if typ == 2:
                tags[tag] = data.decode("latin1")
            else:
                f = fmts[typ]
                tags[tag] = struct.unpack(bo + f * cnt, data)
    return tags


class WacMosaic:
    """LROC WAC 100 m global mosaic (equirectangular) with a cached 2x pyramid on disk."""

    def __init__(self, path=WAC_TIF, cache_dir=DATA_DIR):
        tags = _read_geotiff_meta(path)
        self.W = tags[256][0]
        self.H = tags[257][0]
        assert tags[258][0] == 8 and tags[259][0] == 1, "expected uncompressed 8-bit"
        offs = tags[273]
        assert offs[-1] - offs[0] == (len(offs) - 1) * self.W * tags[278][0], "strips not contiguous"
        scale = tags[33550]       # (sx, sy, sz) in metres
        tie = tags[33922]         # (i, j, k, x, y, z)
        self.m_per_px = scale[0]
        R = 1737400.0
        deg_per_m = 180 / (np.pi * R)
        # map x,y (metres, simple cylindrical about lon 0) -> degrees
        self.lon0 = tie[3] * deg_per_m - tie[0] * scale[0] * deg_per_m   # lon of pixel-corner 0
        self.lat0 = tie[4] * deg_per_m + tie[1] * scale[1] * deg_per_m   # lat of top edge
        self.ppd = 1.0 / (scale[0] * deg_per_m)
        base = np.memmap(path, dtype=np.uint8, mode="r", offset=offs[0], shape=(self.H, self.W))
        self.levels = [base]
        self.cache_dir = Path(cache_dir)
        self._load_pyramid()

    def _load_pyramid(self, n=5):
        prev = self.levels[0]
        for k in range(1, n):
            p = self.cache_dir / f"wac_l{k}.npy"
            if not p.exists():
                h, w = prev.shape[0] // 2, prev.shape[1] // 2
                out = np.lib.format.open_memmap(p, mode="w+", dtype=np.uint8, shape=(h, w))
                step = 2048
                for r in range(0, h, step):
                    blk = np.asarray(prev[2 * r:2 * min(h, r + step), :2 * w], dtype=np.uint16)
                    blk = (blk[0::2, 0::2] + blk[1::2, 0::2] + blk[0::2, 1::2] + blk[1::2, 1::2] + 2) // 4
                    out[r:r + blk.shape[0]] = blk.astype(np.uint8)
                out.flush()
                del out
            prev = np.load(p, mmap_mode="r")
            self.levels.append(prev)

    def res_km(self, k):
        return self.m_per_px * 2 ** k / 1000.0

    def sample(self, lat, lon, gsd_km):
        """Bilinear sample at lat/lon arrays (H, W) using the pyramid level matched to gsd."""
        import cv2
        k = 0
        while k + 1 < len(self.levels) and self.res_km(k + 1) <= gsd_km:
            k += 1
        lon_c = float(np.median(lon))
        lon_u = lon_c + ((lon - lon_c + 180) % 360 - 180)
        while True:
            img = self.levels[k]
            ppd = self.ppd / 2 ** k
            Hk, Wk = img.shape
            x = (lon_u - self.lon0) * ppd - 0.5
            y = (self.lat0 - lat) * ppd - 0.5
            x0 = int(np.floor(x.min())) - 1
            x1 = int(np.ceil(x.max())) + 2
            y0 = max(0, int(np.floor(y.min())) - 1)
            y1 = min(Hk, int(np.ceil(y.max())) + 2)
            # cv2.remap needs source windows < 32767 px (wide near the poles): go coarser
            if max(x1 - x0, y1 - y0) < 32000 or k + 1 == len(self.levels):
                break
            k += 1
        if y1 <= y0 or x1 - x0 > 4 * Wk:
            return np.zeros(lat.shape, np.float32)
        cols = np.arange(x0, x1) % Wk
        if cols[-1] - cols[0] == len(cols) - 1:
            win = np.asarray(img[y0:y1, cols[0]:cols[-1] + 1])
        else:
            win = np.asarray(img[y0:y1])[:, cols]
        return cv2.remap(win.astype(np.float32), (x - x0).astype(np.float32), (y - y0).astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
