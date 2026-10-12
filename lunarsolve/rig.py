"""Joint solve of a rigid multi-camera rig: several images taken at the same instant by cameras with
known mounting (rotation and offset in a common rig frame) and known intrinsics.

One image that solves blind fixes the rig pose.  The known mounting then predicts the pose of every
other camera, so the other images only need matching near a prediction, not a blind search.  A joint
robust least-squares fit of the single 6-DoF rig pose to the matches of all cameras gives the final
pose and its covariance.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .config import MOON_RADIUS_KM as R
from .geometry import project, vec_to_latlon


@dataclass
class RigCamera:
    """One camera of the rig.

    R_cr: rotation rig frame -> camera frame (x_cam = R_cr @ x_rig).  Camera axes: x right, y down,
          z boresight.
    t_r:  camera centre in the rig frame (km; usually metres, so about 0).
    """
    R_cr: np.ndarray
    focal_px: float
    t_r: np.ndarray = field(default_factory=lambda: np.zeros(3))
    principal: tuple = None


@dataclass
class RigSolution:
    lat: float
    lon: float
    altitude_km: float
    sigma_horizontal_km: float
    sigma_altitude_km: float
    C: np.ndarray
    Rwr: np.ndarray                     # world -> rig rotation
    n_matches: int
    matches_per_camera: list
    rms_px: float
    seed_camera: int
    solve_time_s: float
    cameras: list = field(repr=False, default=None)   # per-camera Solution objects (or None)
    cov_C: np.ndarray = field(repr=False, default=None)  # covariance of C (km^2, body frame)

    def summary(self):
        per = ", ".join(str(n) for n in self.matches_per_camera)
        return (f"rig centre    lat {self.lat:+.5f}  lon {self.lon:.5f}  altitude {self.altitude_km:.3f} km\n"
                f"1-sigma       horizontal {self.sigma_horizontal_km * 1000:.0f} m, "
                f"altitude {self.sigma_altitude_km * 1000:.0f} m\n"
                f"{self.n_matches} crater matches ({per} per camera), rms {self.rms_px:.2f} px, "
                f"seed camera {self.seed_camera}, {self.solve_time_s:.1f} s")


class RigSolver:
    def __init__(self, solver):
        self.s = solver

    def _camera_pose(self, C_rig, Rwr, cam):
        Rwc = cam.R_cr @ Rwr
        C = C_rig + Rwr.T @ np.asarray(cam.t_r, float)
        return C, Rwc

    def _match_camera(self, img_dets, shape, cam, C, Rwc, search_frac=0.0):
        """Match one camera's detections against the catalog projected through a predicted pose.

        search_frac > 0: first find the image offset of the prediction by voting.  Every pair of a large
        detection and a projected catalog crater of similar size votes for the shift between them; the
        histogram peak is the offset (up to search_frac of the image width).
        """
        s = self.s
        H, W = shape
        pp = cam.principal if cam.principal is not None else ((W - 1) / 2, (H - 1) / 2)
        gsd = max(np.linalg.norm(C) - R, 1e-3) / cam.focal_px
        sel = np.nonzero(img_dets[:, 2] * 2 * gsd >= s.cat.d_min_km)[0][:800]
        if len(sel) < 3:
            return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64)
        d = img_dets[sel, :3].copy()
        m = int(search_frac * max(H, W))
        s._pp = (pp[0] + m, pp[1] + m)                 # enlarged view: catalog craters just outside count too
        idx, cu = s._camera_view_catalog(C, Rwc, cam.focal_px, 1, (H + 2 * m, W + 2 * m), d[:, 2].min())
        cu = cu.copy()
        cu[:, :2] -= m
        s._pp = pp
        if m > 0 and len(cu):
            big = d[:150]
            dr = np.abs(np.log(cu[None, :, 2] / big[:, None, 2])) < 0.3
            ii, jj = np.nonzero(dr)
            off = cu[jj, :2] - big[ii, :2]
            ok = np.all(np.abs(off) < m, 1)
            if ok.sum() >= 5:
                off = off[ok]
                b = max(4.0, float(np.median(big[:, 2])) * 0.5)
                hist, xe, ye = np.histogram2d(off[:, 0], off[:, 1], bins=int(2 * m / b) + 1, range=[[-m, m], [-m, m]])
                k = np.unravel_index(np.argmax(hist), hist.shape)
                c0 = np.array([(xe[k[0]] + xe[k[0] + 1]) / 2, (ye[k[1]] + ye[k[1] + 1]) / 2])
                near = np.linalg.norm(off - c0, axis=1) < 1.5 * b
                shift = off[near].mean(0) if near.any() else c0
                d[:, :2] += shift                       # move detections onto the predicted catalog frame
        pr, _ = s._match(d, cu)
        if len(pr) == 0:
            return np.zeros((0, 3)), np.zeros(0, np.int64), np.zeros(0, np.int64)
        return img_dets[sel][pr[:, 0], :3], idx[pr[:, 1]], sel[pr[:, 0]]

    def _joint_fit(self, C0, Rwr0, obs):
        """obs: list of (cam, shape, dets (K,3), cat_ids). Fit C (3) and a rotation update (3)."""
        s = self.s
        blocks = []
        for cam, shape, d, ids, *_ in obs:
            if len(ids) == 0:
                continue
            H, W = shape
            pp = cam.principal if cam.principal is not None else ((W - 1) / 2, (H - 1) / 2)
            blocks.append((cam, pp, d[:, :2], 0.06 * d[:, 2] + 0.5, s.cat.points_km(ids)))

        def resid(x):
            Rwr = Rotation.from_rotvec(x[3:]).as_matrix() @ Rwr0
            out = []
            for cam, pp, uv, sig, P in blocks:
                C, Rwc = self._camera_pose(x[:3], Rwr, cam)
                pr, z = project(P, C, Rwc, cam.focal_px, pp[0], pp[1])
                r = ((pr - uv) / sig[:, None]).ravel()
                out.append(np.where(np.repeat(z, 2) > 0, r, 1e3))
            return np.concatenate(out)

        x0 = np.concatenate([C0, np.zeros(3)])
        sol = least_squares(resid, x0, loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=300)
        Rwr = Rotation.from_rotvec(sol.x[3:]).as_matrix() @ Rwr0
        J = sol.jac
        dof = max(1, J.shape[0] - J.shape[1])
        cov = np.linalg.pinv(J.T @ J) * np.sum(sol.fun ** 2) / dof
        return sol.x[:3], Rwr, cov[:3, :3], sol

    def _solve_merged(self, images, cameras, dets, max_time, verbose=False):
        """Blind solve of all cameras' detections in one virtual pinhole view along the rig axis.

        The camera centres coincide (offsets of metres), so each detection's ray maps into the rig
        frame by a pure rotation.  Radii are scaled by the local area magnification of that map.
        """
        f_v = float(np.median([c.focal_px for c in cameras]))
        merged = []
        for img, cam, d in zip(images, cameras, dets):
            if len(d) == 0:
                continue
            H, W = img.shape
            pp = cam.principal if cam.principal is not None else ((W - 1) / 2, (H - 1) / 2)

            def to_virtual(uv):
                ray = np.column_stack([(uv[:, 0] - pp[0]) / cam.focal_px, (uv[:, 1] - pp[1]) / cam.focal_px,
                                       np.ones(len(uv))]) @ cam.R_cr        # camera frame -> rig frame
                ok = ray[:, 2] > 0.05
                return f_v * ray[:, :2] / np.where(ok, ray[:, 2], 1)[:, None], ok

            q, ok = to_virtual(d[:, :2])
            qx, _ = to_virtual(d[:, :2] + [1.0, 0.0])
            qy, _ = to_virtual(d[:, :2] + [0.0, 1.0])
            mag = np.sqrt(np.abs((qx[:, 0] - q[:, 0]) * (qy[:, 1] - q[:, 1]) - (qx[:, 1] - q[:, 1]) * (qy[:, 0] - q[:, 0])))
            merged.append(np.column_stack([q, d[:, 2] * mag, d[:, 3]])[ok])
        if not merged:
            return None
        m = np.concatenate(merged)
        lo = m[:, :2].min(0) - 2
        hi = m[:, :2].max(0) + 2
        shape = (int(hi[1] - lo[1]) + 1, int(hi[0] - lo[0]) + 1)
        m[:, :2] -= lo
        pp_v = (-lo[0], -lo[1])
        if verbose:
            print(f"merged view: {len(m)} detections, {shape[1]}x{shape[0]} px")
        return self.s.solve(dets=m, shape=shape, focal_px=f_v, principal=pp_v, parity="normal",
                            max_time=max_time, limb=False)

    def solve(self, images, cameras, max_time=40.0, verbose=False):
        """images: list of grayscale float images; cameras: list of RigCamera (same order)."""
        t0 = time.time()
        s = self.s
        dets = []
        for img in images:
            d = s.detect(img)
            d = d[d[:, 3] >= 0.35]
            dets.append(d[np.argsort(-d[:, 2], kind="stable")])
        order = np.argsort([-len(d) for d in dets])
        seed, sol0 = None, None
        single_budget = 0.5 * max_time                   # the rest is for the merged-view fallback
        for k, i in enumerate(order):
            left = single_budget - (time.time() - t0)
            if left < 2:
                break
            cam = cameras[i]
            sol0 = s.solve(dets=dets[i], shape=images[i].shape, focal_px=cam.focal_px, principal=cam.principal,
                           parity="normal", max_time=left / (len(order) - k), limb=False)
            if verbose:
                print(f"camera {i}: {'solved' if sol0 else 'no blind solution'}")
            if sol0 is not None:
                seed = int(i)
                break
        if sol0 is None:
            # No camera solves alone: merge all detections into one wide virtual view and search that.
            left = max_time - (time.time() - t0)
            sol_v = self._solve_merged(images, cameras, dets, left, verbose) if left > 3 else None
            if sol_v is None:
                return None
            seed = -1
            Rwr = sol_v.Rwc                              # virtual camera axes = rig axes
            C = sol_v.C
        else:
            # rig pose from the seed camera
            cam = cameras[seed]
            Rwr = cam.R_cr.T @ sol0.Rwc
            C = sol0.C - Rwr.T @ np.asarray(cam.t_r, float)
        # match all cameras near their predicted poses, fit jointly, repeat
        # A single narrow camera fixes its ground footprint well but trades camera position against tilt,
        # so the first predictions of the other cameras can be off by a large fraction of their view:
        # search for their offset on the first pass, then refine.
        obs = None
        counts = [0] * len(images)
        for it, search in enumerate((0.35, 0.25, 0.15, 0.05, 0.0, 0.0)):
            obs = []
            for i, (img, cam) in enumerate(zip(images, cameras)):
                Ci, Rwci = self._camera_pose(C, Rwr, cam)
                # keep searching for cameras that are not locked yet (the pose improves every pass)
                srch = 0.0 if (i == seed or (it > 0 and counts[i] >= 25)) else search
                d, ids, rows = self._match_camera(dets[i], img.shape, cam, Ci, Rwci, srch)
                obs.append((cam, img.shape, d, ids, rows))
            counts = [len(o[3]) for o in obs]
            C, Rwr, covC, fit = self._joint_fit(C, Rwr, obs)
            if verbose:
                print(f"pass {it}: matches per camera {[len(o[3]) for o in obs]}")
        n_per = [len(o[3]) for o in obs]
        r = fit.fun.reshape(-1, 2)
        # rms in pixels: residuals are normalised by sigma; undo that for reporting
        sig = np.concatenate([0.06 * o[2][:, 2] + 0.5 for o in obs if len(o[3])])
        rms = float(np.sqrt(np.mean((np.hypot(r[:, 0], r[:, 1]) * sig) ** 2)))
        up = C / np.linalg.norm(C)
        s_alt = float(math.sqrt(max(0.0, up @ covC @ up)))
        horiz = covC - np.outer(covC @ up, up) - np.outer(up, up @ covC) + (up @ covC @ up) * np.outer(up, up)
        s_h = float(math.sqrt(max(0.0, np.trace(horiz))))
        lat, lon = vec_to_latlon(C)
        # per-camera solutions at the final rig pose (for overlays, labels and pixel lookup)
        from .solve import make_solution
        per_cam = []
        for i, (img, cam) in enumerate(zip(images, cameras)):
            Ci, Rwci = self._camera_pose(C, Rwr, cam)
            d, ids, rows = self._match_camera(dets[i], img.shape, cam, Ci, Rwci, 0.0)
            H, W = img.shape
            pp = cam.principal if cam.principal is not None else ((W - 1) / 2, (H - 1) / 2)
            pairs = np.column_stack([rows, np.arange(len(rows))])
            ground = R + (float(np.median(s.cat.rim_height[ids])) if len(ids) else 0.0)
            per_cam.append(make_solution(Ci, Rwci, cam.focal_px, (H, W), pp, False, False, s_alt, rms,
                                         float("nan"), pairs, ids, dets[i], t0, ground,
                                         method="rig"))
        return RigSolution(lat=float(lat), lon=float(lon), altitude_km=float(np.linalg.norm(C) - R),
                           sigma_horizontal_km=s_h, sigma_altitude_km=s_alt, C=C, Rwr=Rwr,
                           n_matches=int(sum(n_per)), matches_per_camera=n_per, rms_px=rms, seed_camera=seed,
                           solve_time_s=time.time() - t0, cameras=per_cam, cov_C=covC)
