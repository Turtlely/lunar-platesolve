"""Dense registration: refine a solved camera (or rig) pose against the WAC basemap on LOLA terrain.

The crater solve matches the image to a catalog.  Catalog positions and the basemap disagree by
hundreds of metres in places, and only a few dozen craters constrain the fit.  Here the image is
matched pixel by pixel to a rendering of the LROC WAC mosaic draped on LOLA terrain, seen through
the current camera.  Every rendered pixel has a known 3-D ground point, so each matched patch gives
one 2-D/3-D correspondence.  Hundreds of patches then constrain the pose.

    ref = Refiner()                                   # loads the WAC mosaic
    sol2 = ref.refine(img, sol)                       # single camera
    rig2 = ref.refine_rig(images, cameras, rig_sol)   # rigid rig (lunarsolve.rig)

Steps (repeated 3-4 times):
  1. Bring the image to a working scale near the basemap pixel (100 m); never upsample it.
  2. Render the basemap through the current camera at the working scale, with the 3-D point of
     every pixel (ray / LOLA terrain intersection).
  3. Filter both images to a lighting-tolerant feature (band-pass intensity or gradient magnitude).
  4. Match square patches on a grid by normalized cross-correlation inside a search window; keep
     distinct peaks; refine the peak to sub-pixel.
  5. Fit the 6 pose parameters (camera centre and rotation) to all correspondences of all cameras
     with a robust least-squares fit.  The search window shrinks every pass.
"""
import math
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import MOON_RADIUS_KM as R
from .geometry import pixel_rays, project, vec_to_latlon

BASEMAP_KM = 0.1                 # WAC mosaic pixel
MODEL_FLOOR = 0.12               # sigma floor in working pixels (see refine_views)
# Quality gate.  Shadows move with the sun, so when the image lighting differs from the basemap the
# patch matches are biased and the "refined" pose can be worse than the crater solve.  Accept the
# refinement only when the image looks like the basemap at the final pose.  Thresholds from
# scripts/eval_refine.py: same-lighting views all have ncc >= 0.96 and patch_frac >= 0.74; every
# different-lighting view that the refinement made worse had ncc <= 0.87.
GATE_NCC = 0.90
GATE_FRAC = 0.60


def gate(res):
    """(accepted, reason) for a RefineResult."""
    if res is None:
        return False, "too few patches matched the reference map"
    if not (res.ncc >= GATE_NCC):
        return False, (f"the image does not look like the reference map (correlation {res.ncc:.2f} < {GATE_NCC}); "
                       f"usually a different sun angle")
    if not (res.patch_frac >= GATE_FRAC):
        return False, f"only {res.patch_frac:.0%} of the image patches matched the reference map"
    return True, ""


@dataclass
class View:
    """One camera for the refinement: image (camera frame, not mirrored), mounting and intrinsics."""
    img: np.ndarray
    R_cr: np.ndarray
    f: float
    cx: float
    cy: float


@dataclass
class RefineResult:
    C: np.ndarray
    Rwr: np.ndarray                       # world -> rig (for a single camera: world -> camera)
    cov_C: np.ndarray                     # km^2
    sigma_horizontal_km: float
    sigma_altitude_km: float
    n_patches: list                       # accepted patches per camera
    rms_px: float                         # at the working scale
    work_gsd_km: list
    iterations: int
    shift_km: float                       # 3-D change of the camera centre
    time_s: float
    ncc: float = float("nan")             # image / reference correlation at the final pose (feature images)
    patch_frac: float = float("nan")      # accepted fraction of the candidate patches (last pass)
    median_score: float = float("nan")    # median NCC of the accepted patches (last pass)
    accepted: bool = True                 # passed the quality gate (see Refiner.refine)
    history: list = field(default_factory=list, repr=False)


def _bilinear(arr, u, v):
    """Sample arr (H, W, ...) at float pixel positions."""
    H, W = arr.shape[:2]
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    a = (u - u0)[:, None]
    b = (v - v0)[:, None]
    return ((1 - a) * (1 - b) * arr[v0, u0] + a * (1 - b) * arr[v0, u0 + 1]
            + (1 - a) * b * arr[v0 + 1, u0] + a * b * arr[v0 + 1, u0 + 1])


def feature(img, mode="bandpass", valid=None):
    """Lighting-tolerant feature image.  Invalid pixels (space, outside the map) become 0."""
    x = np.asarray(img, np.float32).copy()
    if valid is None:
        valid = np.isfinite(x)
    x[~valid] = float(np.nanmean(x[valid])) if valid.any() else 0.0
    x = cv2.GaussianBlur(x, (0, 0), 0.8)
    if mode == "gradient":
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        y = np.sqrt(gx * gx + gy * gy)
        y = y - cv2.GaussianBlur(y, (0, 0), 6.0)
    else:
        y = x - cv2.GaussianBlur(x, (0, 0), 6.0)       # remove brightness gradients and broad albedo
    y[~valid] = 0
    return y


def match_patches(F_img, F_ref, valid_img, valid_ref, patch=32, step=32, search=12, min_score=0.5,
                  min_margin=0.08):
    """Grid of patch matches.  Returns uv in the image, uv in the reference, and the NCC score.

    A match is kept when its NCC peak is at least min_score and higher than the best peak outside
    a 3-px neighbourhood by min_margin (distinct), and the peak is not on the border of the window.
    """
    H, W = F_img.shape
    h = patch // 2
    s = int(search)
    out_i, out_r, out_s = [], [], []
    for v in range(h + s, H - h - s, step):
        for u in range(h + s, W - h - s, step):
            if not valid_img[v - h:v + h, u - h:u + h].all():
                continue
            win_valid = valid_ref[v - h - s:v + h + s, u - h - s:u + h + s]
            if win_valid.mean() < 0.98:
                continue
            t = F_img[v - h:v + h, u - h:u + h]
            if t.std() < 1e-4:
                continue
            win = F_ref[v - h - s:v + h + s, u - h - s:u + h + s]
            res = cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED)
            _, best, _, (x, y) = cv2.minMaxLoc(res)
            if best < min_score or x in (0, res.shape[1] - 1) or y in (0, res.shape[0] - 1):
                continue
            r2 = res.copy()
            r2[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = -1
            if best - r2.max() < min_margin:
                continue
            # sub-pixel peak (1-D parabolas)
            dx = 0.5 * (res[y, x - 1] - res[y, x + 1]) / (res[y, x - 1] - 2 * best + res[y, x + 1] - 1e-12)
            dy = 0.5 * (res[y - 1, x] - res[y + 1, x]) / (res[y - 1, x] - 2 * best + res[y + 1, x] - 1e-12)
            out_i.append((u - 0.5, v - 0.5))            # patch centre (even-sized patch)
            out_r.append((u - 0.5 + x + np.clip(dx, -0.5, 0.5) - s, v - 0.5 + y + np.clip(dy, -0.5, 0.5) - s))
            out_s.append(best)
    return np.array(out_i).reshape(-1, 2), np.array(out_r).reshape(-1, 2), np.array(out_s)


class Refiner:
    def __init__(self, wac=None):
        if wac is None:
            from .terrain import WacMosaic
            wac = WacMosaic()
        self.wac = wac

    # ------------------------------------------------------------------ rendering
    def render(self, C, Rwc, f, cx, cy, W, H):
        """Basemap through the camera: image (H, W) float32 (NaN off the Moon) and 3-D points (H, W, 3)."""
        from .simulate import terrain_hit
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        d = pixel_rays(uu, vv, C, Rwc, f, cx, cy)
        pts, hit = terrain_hit(C, d)
        lat, lon = vec_to_latlon(pts)
        rng = np.linalg.norm(pts - C, axis=-1)
        cosang = np.abs(np.sum(-d * pts / np.linalg.norm(pts, axis=-1, keepdims=True), axis=-1))
        gsd = float(np.median((rng / f)[hit] / np.maximum(cosang[hit], 0.2))) if hit.any() else BASEMAP_KM
        img = self.wac.sample(lat, lon, gsd).astype(np.float32)
        img[~hit] = np.nan
        return img, pts, hit

    # ------------------------------------------------------------------ core
    @staticmethod
    def _work_scale(view, C, Rwr):
        """Scale factor that brings the image near the basemap pixel (never above 1)."""
        Rwc = view.R_cr @ Rwr
        H, W = view.img.shape
        d = pixel_rays(np.array([view.cx]), np.array([view.cy]), C, Rwc, view.f, view.cx, view.cy)
        from .geometry import ray_sphere
        p, hit = ray_sphere(C, d)
        dist = float(np.linalg.norm(p[0] - C)) if hit[0] else float(np.linalg.norm(C) - R)
        gsd = dist / view.f
        return min(1.0, gsd / BASEMAP_KM), gsd

    def _fit(self, C0, Rwr0, blocks):
        """Robust fit of the rig pose to 2-D/3-D correspondences.  blocks: (view, s, uv_work, X)."""
        def resid(x):
            Rwr = Rotation.from_rotvec(x[3:]).as_matrix() @ Rwr0
            out = []
            for view, s, uv, X in blocks:
                Rwc = view.R_cr @ Rwr
                pr, z = project(X, x[:3], Rwc, view.f * s, (view.cx + 0.5) * s - 0.5, (view.cy + 0.5) * s - 0.5)
                out.append(np.where(np.repeat(z, 2) > 0, (pr - uv).ravel(), 1e3))
            return np.concatenate(out)

        x0 = np.concatenate([C0, np.zeros(3)])
        sol = least_squares(resid, x0, loss="huber", f_scale=1.0, x_scale="jac", max_nfev=200)
        Rwr = Rotation.from_rotvec(sol.x[3:]).as_matrix() @ Rwr0
        r = sol.fun.reshape(-1, 2)
        e = np.hypot(r[:, 0], r[:, 1])
        J = sol.jac
        dof = max(1, J.shape[0] - J.shape[1])
        # robust scale of the residuals (1.4826 MAD per axis)
        s_px = 1.4826 * float(np.median(np.abs(sol.fun))) + 1e-3
        cov = np.linalg.pinv(J.T @ J) * s_px ** 2 * (J.shape[0] / dof)
        return sol.x[:3], Rwr, cov[:3, :3], e

    def refine_views(self, views, C, Rwr, init_err_km=1.5, iters=4, mode="bandpass", patch=32, verbose=False):
        """Refine a rig pose (C, Rwr) with all views.  Returns RefineResult (or None)."""
        t0 = time.time()
        C0 = np.array(C, float)
        C = C0.copy()
        Rwr = np.array(Rwr, float)
        scales = [self._work_scale(v, C, Rwr) for v in views]
        works = []
        for v, (s, _) in zip(views, scales):
            if s < 1.0:
                wimg = cv2.resize(v.img.astype(np.float32), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            else:
                wimg = v.img.astype(np.float32)
            works.append(wimg)
        gsd_w = [g / s if s < 1 else g for s, g in scales]
        hist = []
        cov = np.full((3, 3), np.nan)
        n_pp, rms = [0] * len(views), float("nan")
        searches = []
        for k in range(iters):
            if k == 0:
                srch = [int(np.clip(math.ceil(2.5 * init_err_km / g) + 3, 6, 48)) for g in gsd_w]
            else:
                srch = [max(4, int(round(x * 0.5))) for x in searches[-1]]
            searches.append(srch)
            blocks = []
            n_pp, n_cand, scores = [], 0, []
            for v, (s, _), wimg, sr in zip(views, scales, works, srch):
                H, W = wimg.shape
                Rwc = v.R_cr @ Rwr
                cx, cy = (v.cx + 0.5) * s - 0.5, (v.cy + 0.5) * s - 0.5
                ref, pts, hit = self.render(C, Rwc, v.f * s, cx, cy, W, H)
                valid_i = np.isfinite(wimg) & (wimg > 0.01 * np.nanmax(wimg))
                vi = cv2.erode(valid_i.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                Fi = feature(wimg, mode, vi)
                Fr = feature(ref, mode, hit)
                ui, ur, sc = match_patches(Fi, Fr, vi, hit, patch=patch, step=patch, search=sr)
                n_cand += max(1, int(vi.sum()) // (patch * patch))
                scores.extend(sc.tolist())
                if len(ui) < 6:
                    n_pp.append(0)
                    continue
                X = _bilinear(pts, ur[:, 0], ur[:, 1])
                blocks.append((v, s, ui, X))
                n_pp.append(len(ui))
            if sum(n_pp) < 12:
                if verbose:
                    print(f"refine pass {k}: only {sum(n_pp)} patches")
                if k == 0:
                    return None
                break
            C_new, Rwr_new, cov, e = self._fit(C, Rwr, blocks)
            step = float(np.linalg.norm(C_new - C))
            C, Rwr = C_new, Rwr_new
            rms = float(np.sqrt(np.mean(np.minimum(e, 5) ** 2)))
            hist.append(dict(patches=list(n_pp), search=srch, step_km=step, rms_px=rms,
                             patch_frac=sum(n_pp) / n_cand, median_score=float(np.median(scores))))
            if verbose:
                print(f"refine pass {k}: patches {n_pp}, search {srch} px, step {step * 1000:.0f} m, "
                      f"rms {rms:.2f} px")
            if k >= 2 and step < 0.2 * min(gsd_w):
                break
        # quality at the final pose: correlation of the feature images over all views
        num = den_i = den_r = 0.0
        for v, (s, _), wimg in zip(views, scales, works):
            H, W = wimg.shape
            cx, cy = (v.cx + 0.5) * s - 0.5, (v.cy + 0.5) * s - 0.5
            ref, _, hit = self.render(C, v.R_cr @ Rwr, v.f * s, cx, cy, W, H)
            valid_i = np.isfinite(wimg) & (wimg > 0.01 * np.nanmax(wimg))
            m = valid_i & hit
            if m.sum() < 100:
                continue
            a = feature(wimg, mode, valid_i)[m]
            b = feature(ref, mode, hit)[m]
            a = a - a.mean()
            b = b - b.mean()
            num += float(a @ b)
            den_i += float(a @ a)
            den_r += float(b @ b)
        ncc = num / math.sqrt(den_i * den_r) if den_i > 0 and den_r > 0 else float("nan")
        up = C / np.linalg.norm(C)
        # The fit covariance assumes independent patch errors.  Rendering and resampling errors are
        # correlated over the image, so add a floor of MODEL_FLOOR working pixels.  The absolute
        # registration of the basemap itself is not included.
        floor = MODEL_FLOOR * min(gsd_w) / math.sqrt(len(views))
        s_alt = float(math.sqrt(max(0.0, up @ cov @ up) + (0.5 * floor) ** 2))
        P = np.eye(3) - np.outer(up, up)
        s_h = float(math.sqrt(max(0.0, np.trace(P @ cov @ P)) + floor ** 2))
        return RefineResult(C=C, Rwr=Rwr, cov_C=cov, sigma_horizontal_km=s_h, sigma_altitude_km=s_alt,
                            n_patches=n_pp, rms_px=rms, work_gsd_km=gsd_w, iterations=len(hist),
                            shift_km=float(np.linalg.norm(C - C0)), time_s=time.time() - t0, history=hist,
                            ncc=ncc, patch_frac=hist[-1]["patch_frac"] if hist else float("nan"),
                            median_score=hist[-1]["median_score"] if hist else float("nan"))

    # ------------------------------------------------------------------ public API
    def refine(self, img, sol, init_err_km=None, **kw):
        """Refine a single-camera Solution.  Returns a new Solution (method "<method>+dense").  If the
        registration fails or does not pass the quality gate, returns a copy of the input solution
        with `.refine_rejected` (the reason).  The refinement result is in `.refine`."""
        from .solve import make_solution
        work = img[:, ::-1] if sol.mirrored else img
        cx, cy = sol.principal()
        v = View(np.ascontiguousarray(work), np.eye(3), sol.focal_px, cx, cy)
        if init_err_km is None:
            init_err_km = max(0.5, 2.0 * sol.altitude_sigma_km, 0.02 * sol.altitude_km)
        res = self.refine_views([v], sol.C, sol.Rwc, init_err_km=init_err_km, **kw)
        ok, why = gate(res)
        if not ok:
            if res is not None:
                res.accepted = False
            out = replace(sol)
            out.refine = res
            out.refine_rejected = why
            return out
        pairs = np.zeros((0, 2), np.int64)
        new = make_solution(res.C, res.Rwr, sol.focal_px, sol.shape, (cx, cy), sol.mirrored, sol.focal_estimated,
                            res.sigma_altitude_km, res.rms_px, sol.log_odds, pairs, np.zeros(0, np.int64),
                            sol.dets, time.time() - sol.solve_time_s - res.time_s, R, top_frac=sol.top_frac,
                            method=sol.method + "+dense")
        new = replace(new, matches=sol.matches, n_matches=sol.n_matches)
        new.refine = res
        return new

    def refine_rig(self, images, cameras, rig, **kw):
        """Refine a RigSolution (lunarsolve.rig).  Returns a new RigSolution, or the input on failure."""
        views = []
        for img, cam in zip(images, cameras):
            H, W = img.shape
            cx, cy = cam.principal if cam.principal is not None else ((W - 1) / 2, (H - 1) / 2)
            views.append(View(img, cam.R_cr, cam.focal_px, cx, cy))
        init = max(0.3, 3.0 * rig.sigma_horizontal_km)
        res = self.refine_views(views, rig.C, rig.Rwr, init_err_km=init, **kw)
        ok, why = gate(res)
        if not ok:
            if res is not None:
                res.accepted = False
            out = replace(rig)
            out.refine = res
            out.refine_rejected = why
            return out
        from .solve import make_solution
        lat, lon = vec_to_latlon(res.C)
        cams = []
        for v, sol in zip(views, rig.cameras or [None] * len(views)):
            Rwc = v.R_cr @ res.Rwr
            H, W = v.img.shape
            dets = sol.dets if sol is not None else np.zeros((0, 3))
            new = make_solution(res.C, Rwc, v.f, (H, W), (v.cx, v.cy), False, False, res.sigma_altitude_km,
                                res.rms_px, float("nan"), np.zeros((0, 2), np.int64), np.zeros(0, np.int64), dets,
                                time.time(), R, method="rig+dense")
            if sol is not None:
                new = replace(new, matches=sol.matches, n_matches=sol.n_matches)
            cams.append(new)
        out = replace(rig, lat=float(lat), lon=float(lon), altitude_km=float(np.linalg.norm(res.C) - R),
                      sigma_horizontal_km=res.sigma_horizontal_km, sigma_altitude_km=res.sigma_altitude_km,
                      C=res.C, Rwr=res.Rwr, cameras=cams, cov_C=res.cov_C,
                      solve_time_s=rig.solve_time_s + res.time_s)
        out.refine = res
        return out
