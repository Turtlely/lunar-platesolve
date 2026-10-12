"""Crater detection on arbitrary images with a scale pyramid."""
import contextlib

import cv2
import numpy as np
import torch

from .config import WEIGHTS_PATH
from .model import CraterNet, decode, normalize


def default_device():
    """LUNARSOLVE_DEVICE if set, else "cuda" when a GPU is available, else "cpu"."""
    import os
    dev = os.environ.get("LUNARSOLVE_DEVICE")
    if dev:
        return dev
    return "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"


def load_model(path=WEIGHTS_PATH, device=None):
    device = device or default_device()
    model = CraterNet()
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def to_gray_float(img):
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2GRAY) if img.shape[2] >= 3 else img[..., 0]
    img = img.astype(np.float32)
    hi = np.percentile(img, 99.9)
    return img / max(hi, 1e-6)


def nms_circles(d, dist_frac=0.4, rad_tol=0.35):
    """d: (N,4) x,y,r,score. Keep the highest-score circle among near-duplicates."""
    if len(d) == 0:
        return d
    d = d[np.argsort(-d[:, 3])]
    from scipy.spatial import cKDTree
    tree = cKDTree(d[:, :2])
    keep = np.ones(len(d), bool)
    for i in range(len(d)):
        if not keep[i]:
            continue
        for j in tree.query_ball_point(d[i, :2], dist_frac * d[i, 2]):
            if j > i and keep[j] and abs(np.log(d[j, 2] / d[i, 2])) < rad_tol:
                keep[j] = False
    return d[keep]


@torch.no_grad()
def detect(model, img, thresh=0.3, device=None, min_side=48, rmin=3.0):
    """Detect craters in a grayscale image (H, W) array. Returns (N,4): x, y, r (px), score."""
    device = device or next(model.parameters()).device.type
    on_gpu = str(device).startswith("cuda")
    img = to_gray_float(img)
    out = []
    level = 0
    cur = img
    while min(cur.shape) >= min_side:
        H, W = cur.shape
        ph, pw = (-H) % 32, (-W) % 32
        t = torch.from_numpy(np.pad(cur, ((0, ph), (0, pw)), mode="reflect"))[None, None].to(device)
        t = normalize(t)
        # bf16 on the GPU; plain float32 on the CPU (autocast("cuda") queries CUDA even when disabled)
        with (torch.autocast("cuda", dtype=torch.bfloat16) if on_gpu else contextlib.nullcontext()):
            o = model(t)
        o = {k: v.float() for k, v in o.items()} if isinstance(o, dict) else o.float()
        d = decode(o, thresh)[0]
        d = d[(d[:, 0] < W) & (d[:, 1] < H)]
        last = min(cur.shape[0] // 2, cur.shape[1] // 2) < min_side
        lo = rmin if level == 0 else 14.0
        hi = 48.0 if last else 36.0
        d = d[(d[:, 2] >= lo) & (d[:, 2] <= hi)]
        s = 2.0 ** level
        d[:, :3] = d[:, :3] * s
        d[:, :2] += (s - 1) / 2       # pixel-centre convention of the 2x area pyramid
        out.append(d)
        cur = cv2.resize(cur, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
        level += 1
    d = np.concatenate(out) if out else np.zeros((0, 4), np.float32)
    return nms_circles(d)
