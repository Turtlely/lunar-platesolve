"""CraterNet: a small U-Net with CenterNet-style heads at stride 2.

Heads: crater-centre heatmap (1), sub-cell offset (2), log radius in input pixels (1).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRIDE = 2


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None
        if stride != 1 or cin != cout:
            self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.silu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.silu(y + (x if self.skip is None else self.skip(x)))


class Up(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.block = ResBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], 1))


class CraterNet(nn.Module):
    def __init__(self, ch=(24, 32, 64, 128, 192, 256)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, ch[0], 3, 1, 1, bias=False), nn.BatchNorm2d(ch[0]), nn.SiLU())
        self.downs = nn.ModuleList()
        for i in range(1, len(ch)):
            self.downs.append(nn.Sequential(ResBlock(ch[i - 1], ch[i], 2), ResBlock(ch[i], ch[i])))
        # decode back to stride 2 (feature index 1)
        self.ups = nn.ModuleList()
        c = ch[-1]
        for i in range(len(ch) - 2, 0, -1):
            self.ups.append(Up(c, ch[i], ch[i]))
            c = ch[i]
        self.head = nn.Sequential(nn.Conv2d(c, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 4, 1))
        nn.init.constant_(self.head[-1].bias[:1], -4.0)
        nn.init.constant_(self.head[-1].bias[3:], 2.0)

    def forward(self, x):
        feats = [self.stem(x)]
        for d in self.downs:
            feats.append(d(feats[-1]))
        y = feats[-1]
        for k, up in enumerate(self.ups):
            y = up(y, feats[-2 - k])
        return self.head(y)       # (B, 4, H/2, W/2): logit, dx, dy, log r


def normalize(img):
    """Per-image standardisation; img (B,1,H,W)."""
    m = img.mean(dim=(2, 3), keepdim=True)
    s = img.std(dim=(2, 3), keepdim=True)
    return (img - m) / (s + 1e-3)


# ----------------------------------------------------------------------------- targets / loss

def build_targets(keeps, ignores, H, W, device):
    """Heatmap (B,1,h,w), ignore weight (B,1,h,w), and regression lists for positive cells."""
    B = len(keeps)
    h, w = H // STRIDE, W // STRIDE
    heat = torch.zeros(B, 1, h, w, device=device)
    weight = torch.ones(B, 1, h, w, device=device)
    ys = torch.arange(h, device=device, dtype=torch.float32).view(1, h, 1)
    xs = torch.arange(w, device=device, dtype=torch.float32).view(1, 1, w)
    pos_b, pos_y, pos_x, reg = [], [], [], []
    for b in range(B):
        k = torch.as_tensor(keeps[b], device=device)
        if len(k):
            c = (k[:, :2] + 0.5) / STRIDE - 0.5
            r_out = k[:, 2] / STRIDE
            sig = (r_out / 4).clamp_min(0.6).view(-1, 1, 1)
            d2 = (xs - c[:, 0].view(-1, 1, 1)) ** 2 + (ys - c[:, 1].view(-1, 1, 1)) ** 2
            heat[b, 0] = torch.exp(-d2 / (2 * sig ** 2)).amax(0)
            cell = c.round().long()
            cell[:, 0].clamp_(0, w - 1)
            cell[:, 1].clamp_(0, h - 1)
            heat[b, 0, cell[:, 1], cell[:, 0]] = 1.0
            pos_b.append(torch.full((len(k),), b, device=device, dtype=torch.long))
            pos_y.append(cell[:, 1])
            pos_x.append(cell[:, 0])
            reg.append(torch.stack([c[:, 0] - cell[:, 0], c[:, 1] - cell[:, 1], k[:, 2].log()], 1))
        ig = torch.as_tensor(ignores[b], device=device)
        if len(ig):
            c = (ig[:, :2] + 0.5) / STRIDE - 0.5
            rad = (ig[:, 2] / STRIDE * 0.5).clamp_min(1.5).view(-1, 1, 1)
            d2 = (xs - c[:, 0].view(-1, 1, 1)) ** 2 + (ys - c[:, 1].view(-1, 1, 1)) ** 2
            weight[b, 0] = torch.where((d2 < rad ** 2).any(0), 0.0, 1.0)
    if pos_b:
        pos = (torch.cat(pos_b), torch.cat(pos_y), torch.cat(pos_x))
        reg = torch.cat(reg)
    else:
        pos, reg = None, None
    return heat, weight, pos, reg


def centernet_loss(out, heat, weight, pos, reg):
    logit = out[:, :1].float()
    p = torch.sigmoid(logit).clamp(1e-4, 1 - 1e-4)
    is_pos = heat == 1.0
    neg_w = (1 - heat) ** 4 * weight * (~is_pos)
    lpos = -((1 - p) ** 2 * torch.log(p))[is_pos].sum()
    lneg = -(neg_w * p ** 2 * torch.log(1 - p)).sum()
    npos = max(1, int(is_pos.sum()))
    l_hm = (lpos + lneg) / npos
    if pos is None:
        return l_hm, torch.zeros((), device=out.device), torch.zeros((), device=out.device)
    b, y, x = pos
    pr = out[b, 1:, y, x].float()
    l_off = F.l1_loss(pr[:, :2], reg[:, :2])
    l_rad = F.l1_loss(pr[:, 2], reg[:, 2])
    return l_hm, l_off, l_rad


@torch.no_grad()
def decode(out, thresh=0.3, max_det=2000):
    """out (B,4,h,w) -> list of (N,4) arrays: x, y, r (input px), score."""
    p = torch.sigmoid(out[:, :1].float())
    peak = (F.max_pool2d(p, 3, 1, 1) == p) & (p > thresh)
    res = []
    for b in range(out.shape[0]):
        yy, xx = torch.nonzero(peak[b, 0], as_tuple=True)
        s = p[b, 0, yy, xx]
        if len(s) > max_det:
            top = torch.topk(s, max_det).indices
            yy, xx, s = yy[top], xx[top], s[top]
        dx = out[b, 1, yy, xx].float()
        dy = out[b, 2, yy, xx].float()
        r = out[b, 3, yy, xx].float().exp()
        cx = (xx.float() + dx) * STRIDE + (STRIDE - 1) / 2
        cy = (yy.float() + dy) * STRIDE + (STRIDE - 1) / 2
        res.append(torch.stack([cx, cy, r, s], 1).cpu().numpy())
    return res
