"""Train CraterNet on a mix of DEM renders and real WAC crops.

    python -m lunarsolve.train --iters 30000

Longitudes in [330, 360) are held out for validation.
"""
import argparse
import math
import queue
import threading
import time

import numpy as np
import torch

from .catalog import Catalog
from .config import WEIGHTS_PATH
from .model import CraterNet, build_targets, centernet_loss, decode, normalize
from .synth import DemRenderer, wac_sample

TRAIN_LON = (0.0, 330.0)
VAL_LON = (330.0, 360.0)


class WacStream(torch.utils.data.IterableDataset):
    def __init__(self, size, lon_range, seed):
        self.size = size
        self.lon_range = lon_range
        self.seed = seed

    def __iter__(self):
        from .terrain import WacMosaic
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + 1000 * wid)
        cat = Catalog()
        wac = WacMosaic()
        while True:
            img, keep, ign, meta = wac_sample(wac, cat, rng, self.size, self.lon_range)
            if img.std() < 1e-3:
                continue
            yield torch.from_numpy(img)[None], keep, ign


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch], [b[2] for b in batch]


def match_stats(dets, keeps, dist_frac=0.5, rad_tol=0.25):
    """Greedy match of detections to labels; returns tp, n_det, n_gt, radius errors."""
    tp = nd = ng = 0
    rerr = []
    for d, k in zip(dets, keeps):
        nd += len(d)
        ng += len(k)
        if len(d) == 0 or len(k) == 0:
            continue
        used = np.zeros(len(d), bool)
        for x, y, r in k:
            dd = np.hypot(d[:, 0] - x, d[:, 1] - y)
            ok = (~used) & (dd < dist_frac * r) & (np.abs(np.log(d[:, 2] / r)) < rad_tol)
            if ok.any():
                j = np.argmin(np.where(ok, dd, np.inf))
                used[j] = True
                tp += 1
                rerr.append(math.log(d[j, 2] / r))
    return tp, nd, ng, np.array(rerr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wac-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="start from these weights (fresh LR schedule)")
    ap.add_argument("--no-albedo", action="store_true", help="noise albedo only (no Clementine map)")
    ap.add_argument("--out", default=str(WEIGHTS_PATH))
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    from .terrain import GpuDem
    cat = Catalog()
    dem = GpuDem(dev)
    albedo = None
    if not args.no_albedo:
        from .terrain import GpuAlbedo
        albedo = GpuAlbedo(dev)
    rend = DemRenderer(dem, cat, args.size, dev, albedo=albedo)
    n_wac = int(round(args.batch * args.wac_frac))
    n_dem = args.batch - n_wac
    loader = iter(torch.utils.data.DataLoader(WacStream(args.size, TRAIN_LON, 0), batch_size=n_wac,
                                              num_workers=args.workers, collate_fn=collate,
                                              persistent_workers=True, prefetch_factor=4))
    model = CraterNet().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.iters, pct_start=0.05)
    scaler = torch.amp.GradScaler()
    start = 0
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
    if args.resume:
        ck = torch.load(args.out, map_location=dev)
        model.load_state_dict(ck["model"])
        start = ck.get("iter", 0)
    rng = np.random.default_rng(12345)
    vrng = np.random.default_rng(999)
    # fixed validation set
    val = []
    vloader = iter(torch.utils.data.DataLoader(WacStream(args.size, VAL_LON, 77), batch_size=16, num_workers=0,
                                               collate_fn=collate))
    for _ in range(4):
        imgs, k, ig, _m = rend.render(16, vrng, lon_range=VAL_LON, gsd_range=(0.6, 4.0))
        val.append(("dem", imgs, k))
        wi, wk, _ = next(vloader)
        val.append(("wac", wi.to(dev), wk))

    dem_q = queue.Queue(maxsize=4)

    def dem_worker():
        while True:
            dem_q.put(rend.render(n_dem, rng, lon_range=TRAIN_LON, gsd_range=(0.6, 4.0)))

    threading.Thread(target=dem_worker, daemon=True).start()
    t0 = time.time()
    hist = []
    for it in range(start, args.iters):
        model.train()
        ims_d, kd, igd, _ = dem_q.get()
        ims_w, kw, igw = next(loader)
        imgs = torch.cat([ims_d, ims_w.to(dev, non_blocking=True)])
        # random flips/rot90 are free augmentation
        keeps, ignores = kd + kw, igd + igw
        if it % 2:
            imgs = imgs.flip(-1)
            S = imgs.shape[-1]
            keeps = [np.column_stack([S - 1 - k[:, 0], k[:, 1], k[:, 2]]) if len(k) else k for k in keeps]
            ignores = [np.column_stack([S - 1 - k[:, 0], k[:, 1], k[:, 2]]) if len(k) else k for k in ignores]
        # random global contrast inversion is unphysical; skip. Random brightness offset:
        x = normalize(imgs).contiguous(memory_format=torch.channels_last)
        heat, weight, pos, reg = build_targets(keeps, ignores, args.size, args.size, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        l_hm, l_off, l_rad = centernet_loss(out, heat, weight, pos, reg)
        loss = l_hm + l_off + 2.0 * l_rad
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        hist.append([l_hm.item(), l_off.item(), l_rad.item()])
        if (it + 1) % 100 == 0:
            h = np.mean(hist, 0)
            hist = []
            print(f"it {it + 1} loss hm {h[0]:.3f} off {h[1]:.3f} rad {h[2]:.3f} "
                  f"lr {sched.get_last_lr()[0]:.2e} {(time.time() - t0) / (it + 1 - start):.3f}s/it", flush=True)
        if (it + 1) % 2000 == 0 or it + 1 == args.iters:
            model.eval()
            for src in ("dem", "wac"):
                tp = nd = ng = 0
                rerr = []
                for s, im, k in val:
                    if s != src:
                        continue
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        o = model(normalize(im).contiguous(memory_format=torch.channels_last))
                    dets = decode(o, 0.3)
                    a, b, c, e = match_stats([d[(d[:, 2] >= 3.5) & (d[:, 2] <= 44)] for d in dets],
                                             [kk[(kk[:, 2] >= 3) & (kk[:, 2] <= 48)] for kk in k])
                    tp += a; nd += b; ng += c; rerr.append(e)
                rerr = np.concatenate(rerr)
                print(f"  val {src}: precision {tp / max(nd, 1):.3f} recall {tp / max(ng, 1):.3f} "
                      f"rad err {np.std(rerr) if len(rerr) else 0:.3f}", flush=True)
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "iter": it + 1}, args.out)


if __name__ == "__main__":
    main()
