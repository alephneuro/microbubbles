from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates, maximum_filter
from scipy.optimize import linear_sum_assignment

from .h5_io import acq_keys, grid_arrays, load_compound, open_h5, select_acquisitions
from .runtime import dump_pickle, load_pickle
from .svd import doppler_velocity_to_freq, filtered_magnitude


@dataclass(frozen=True)
class TrackingOptions:
    beamformed_path: Path
    tracks_path: Path
    num_acqs: int | None = None
    acq_start: int | None = None
    acq_step: int = 1
    output_per_acq: bool = False
    sigma_threshold: float = 2.0
    svd_low_cutoff: float = 0.1
    svd_high_cutoff: float | None = None
    svd_n_components: int | None = None
    svd_method: str = "fast"
    knee_filter: bool = True
    tissue_freq_hz: float = 100.0
    # Carrier-invariant alternative to tissue_freq_hz; converted per acquisition
    # from the acquisition's own tx frequency and speed of sound. Takes
    # precedence over tissue_freq_hz when set.
    tissue_velocity_mm_s: float | None = None
    temporal_sigma: float = 0.0
    filter_method: str = "svd"
    min_distance: int = 2
    smoothing_sigma: float = 1.0
    subpixel: str = "centroid"
    window_size: int = 5
    patch_radius: int = 3
    tracking: str = "kalman"
    frame_rate_hz: float | None = None
    max_dist: tuple[float, float, float] | None = None
    max_dist_mms: tuple[float, float, float] | None = None
    max_gap: int = 3
    min_track_length: int = 15
    reversal_penalty: float = 10.0
    max_cost: float = 10.0
    smooth_sigma: float = 2.0
    smooth_method: str = "gaussian"
    smooth_window: int = 5
    smooth_interp_factor: float = 0.1
    export_dir: Path | None = None
    export_stem: str | None = None
    export_min_lengths: tuple[int, ...] = (5, 20, 50)


def _acq_id_from_path(path: Path) -> int:
    match = re.search(r"_acq(\d+)\.pkl$", path.name)
    if not match:
        raise ValueError(f"Cannot parse acquisition id from {path}")
    return int(match.group(1))


def _parabolic_offset(y_minus: float, y_center: float, y_plus: float) -> float:
    denom = 2.0 * (y_minus + y_plus - 2.0 * y_center)
    if abs(denom) < 1e-10:
        return 0.0
    return float(np.clip((y_minus - y_plus) / denom, -0.5, 0.5))


def subpixel_localize_3d(
    frame: np.ndarray,
    pixel_coords: np.ndarray,
    method: str,
    window_size: int,
) -> np.ndarray:
    if len(pixel_coords) == 0:
        return np.empty((0, 3), dtype=np.float32)

    n_elev, n_z, n_x = frame.shape
    out: list[list[float]] = []
    for coord in pixel_coords:
        e, z, x = (int(coord[0]), int(coord[1]), int(coord[2]))
        if method == "parabolic":
            offsets = [0.0, 0.0, 0.0]
            if 0 < e < n_elev - 1:
                offsets[0] = _parabolic_offset(frame[e - 1, z, x], frame[e, z, x], frame[e + 1, z, x])
            if 0 < z < n_z - 1:
                offsets[1] = _parabolic_offset(frame[e, z - 1, x], frame[e, z, x], frame[e, z + 1, x])
            if 0 < x < n_x - 1:
                offsets[2] = _parabolic_offset(frame[e, z, x - 1], frame[e, z, x], frame[e, z, x + 1])
            out.append([e + offsets[0], z + offsets[1], x + offsets[2]])
            continue

        half = max(1, int(window_size) // 2)
        e0, e1 = max(0, e - half), min(n_elev, e + half + 1)
        z0, z1 = max(0, z - half), min(n_z, z + half + 1)
        x0, x1 = max(0, x - half), min(n_x, x + half + 1)
        window = frame[e0:e1, z0:z1, x0:x1]
        weights = window - float(window.min())
        total = float(weights.sum())
        if total <= 0:
            out.append([float(e), float(z), float(x)])
            continue
        ee, zz, xx = np.meshgrid(
            np.arange(e0, e1),
            np.arange(z0, z1),
            np.arange(x0, x1),
            indexing="ij",
        )
        out.append(
            [
                float((ee * weights).sum() / total),
                float((zz * weights).sum() / total),
                float((xx * weights).sum() / total),
            ]
        )
    return np.asarray(out, dtype=np.float32)


def indices_to_mm(
    indices: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, 3), dtype=np.float32)
    coords = indices.T
    x = map_coordinates(grid_x, coords, order=1, mode="nearest")
    y = map_coordinates(grid_y, coords, order=1, mode="nearest")
    z = map_coordinates(grid_z, coords, order=1, mode="nearest")
    return np.stack([x, y, z], axis=1).astype(np.float32, copy=False)


def _slice_stats(volume: np.ndarray, smoothing_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(volume, dtype=np.float32)
    if smoothing_sigma > 0:
        data = gaussian_filter(data, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
    n_elev = data.shape[1]
    means = np.zeros(n_elev, dtype=np.float32)
    stds = np.ones(n_elev, dtype=np.float32)
    for elev in range(n_elev):
        values = data[:, elev]
        mask = values > 0
        if np.any(mask):
            means[elev] = float(values[mask].mean())
            std = float(values[mask].std())
            stds[elev] = std if std > 1e-10 else 1.0
    return means, stds


def detect_batch(
    volume: np.ndarray,
    sigma_threshold: float,
    min_distance: int,
    smoothing_sigma: float,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if volume.ndim != 4:
        raise ValueError(f"Expected volume (frames,elev,z,x), got {volume.shape}")

    smoothed = np.asarray(volume, dtype=np.float32)
    if smoothing_sigma > 0:
        smoothed = gaussian_filter(smoothed, sigma=(0, 0, smoothing_sigma, smoothing_sigma))
    means, stds = _slice_stats(volume, smoothing_sigma)
    zscore = (smoothed - means[None, :, None, None]) / (stds[None, :, None, None] + 1e-10)

    filter_size = 2 * int(min_distance) + 1
    max_filtered = maximum_filter(zscore, size=(1, filter_size, filter_size, filter_size))
    is_peak = (zscore == max_filtered) & (zscore > float(sigma_threshold))
    coords = np.array(np.where(is_peak))
    empty = (np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
    n_frames = int(volume.shape[0])
    if coords.size == 0:
        return [empty for _ in range(n_frames)]

    frame_ids = coords[0]
    spatial = coords[1:].T.astype(np.int32, copy=False)
    intensities = volume[coords[0], coords[1], coords[2], coords[3]].astype(np.float32, copy=False)
    zscores = zscore[coords[0], coords[1], coords[2], coords[3]].astype(np.float32, copy=False)
    bounds = np.searchsorted(frame_ids, np.arange(n_frames + 1))
    out = []
    for frame in range(n_frames):
        lo, hi = int(bounds[frame]), int(bounds[frame + 1])
        if lo == hi:
            out.append(empty)
        else:
            out.append((spatial[lo:hi], intensities[lo:hi], zscores[lo:hi]))
    return out


def _spacing(values: np.ndarray) -> float:
    unique = np.unique(np.round(np.asarray(values, dtype=np.float64).reshape(-1), 6))
    diffs = np.diff(unique)
    diffs = np.abs(diffs[diffs > 1e-6])
    return float(np.median(diffs)) if diffs.size else 0.0


def _grid_spacing(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray) -> dict[str, float]:
    return {
        "dx": _spacing(grid_x[0, 0, :]),
        "dy": _spacing(grid_y[:, 0, 0]),
        "dz": _spacing(grid_z[0, :, 0]),
    }


def _tracking_gate(opts: TrackingOptions, spacing: dict[str, float]) -> tuple[float, float, float]:
    if opts.max_dist is not None:
        return tuple(float(v) for v in opts.max_dist)
    if opts.max_dist_mms is not None:
        if not opts.frame_rate_hz or opts.frame_rate_hz <= 0:
            raise ValueError("--max-dist-mms requires --frame-rate")
        return tuple(float(v) / float(opts.frame_rate_hz) for v in opts.max_dist_mms)
    dx = spacing.get("dx") or 0.5
    dy = spacing.get("dy") or max(dx, spacing.get("dz") or dx)
    dz = spacing.get("dz") or dx
    return (max(dx * 2.0, 0.25), max(dy * 2.0, 0.25), max(dz * 2.0, 0.25))


def kalman_tracking_3d(
    detections: List[np.ndarray],
    max_distance_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_gap: int = 3,
    min_track_length: int = 5,
    process_noise: float = 0.5,
    measurement_noise: float = 0.5,
    reversal_penalty: float = 10.0,
    max_cost: float = 1e5,
    intensities: List[np.ndarray] = None,
) -> List[Dict]:
    """
    Track bubbles in 3D using Kalman filter + Hungarian assignment.

    Args:
        detections: List of (N_i, 3) arrays, one per frame, in mm coordinates (x, y, z)
        max_distance_mm: Maximum distance in mm from CURRENT position for each dimension (x, y, z).
                        This is a hard constraint that prevents large jumps.
        max_gap: Maximum frames to bridge gaps
        min_track_length: Minimum track length to keep
        process_noise: Kalman process noise
        measurement_noise: Kalman measurement noise
        reversal_penalty: Cost penalty for velocity reversals
        max_cost: Maximum assignment cost to accept a match. Pairs above this are
                 rejected even if Hungarian assigns them. Lower values make the
                 reversal penalty actually cause rejection (default: 1e5).
        intensities: Optional list of (N_i,) arrays, one per frame, with detection intensities

    Returns:
        List of track dicts with 'positions', 'frames', 'length' (and 'intensities' if provided)
    """
    has_intensities = intensities is not None
    max_dist_mm = np.array(max_distance_mm)
    tracks = []
    track_id_counter = 0

    # Pre-compute shared Kalman matrices
    Q_template = np.diag([process_noise] * 3 + [process_noise * 2] * 3)
    R_template = np.eye(3) * measurement_noise
    P_init = np.eye(6) * 100.0
    H = np.array(
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=float
    )
    I6 = np.eye(6)

    # Active track state in parallel arrays for vectorized operations
    _pos_last = np.empty((0, 3))  # last observed position
    _frame_last = np.empty(0, dtype=np.int64)
    _ages = np.empty(0, dtype=np.int64)
    _lengths = np.empty(0, dtype=np.int64)
    _states = np.empty((0, 6))  # Kalman state [x,y,z,vx,vy,vz]
    _covs = np.empty((0, 6, 6))  # Kalman covariance P

    # Variable-length history per track (can't vectorize)
    _hist_pos = []  # list of lists of (3,) arrays
    _hist_fr = []  # list of lists of ints
    _hist_int = []  # list of lists of floats

    reject_cost = max_cost * 10
    _n_dead = 0  # count of tombstoned tracks (age == -1)

    def _retire(mask):
        """Mark tracks as dead (tombstone); emit completed tracks."""
        nonlocal _n_dead
        idxs = np.where(mask)[0]
        for i in idxs:
            if _lengths[i] >= min_track_length:
                t = {
                    "positions": np.array(_hist_pos[i]),
                    "frames": np.array(_hist_fr[i]),
                    "length": int(_lengths[i]),
                }
                if has_intensities:
                    t["intensities"] = np.array(_hist_int[i])
                tracks.append(t)
            _hist_pos[i] = None
            _hist_fr[i] = None
            _hist_int[i] = None
        _ages[mask] = -1  # tombstone marker
        _n_dead += int(mask.sum())

    def _compact():
        """Remove tombstoned tracks, compacting arrays and lists."""
        nonlocal _pos_last, _frame_last, _ages, _lengths, _states, _covs
        nonlocal _hist_pos, _hist_fr, _hist_int, _n_dead
        if _n_dead == 0:
            return
        keep = _ages >= 0
        _pos_last = _pos_last[keep]
        _frame_last = _frame_last[keep]
        _ages = _ages[keep]
        _lengths = _lengths[keep]
        _states = _states[keep]
        _covs = _covs[keep]
        _hist_pos[:] = [h for h in _hist_pos if h is not None]
        _hist_fr[:] = [h for h in _hist_fr if h is not None]
        _hist_int[:] = [h for h in _hist_int if h is not None]
        _n_dead = 0

    def _spawn(positions, frame_idx, det_int=None):
        """Create new tracks from unmatched detections."""
        nonlocal _pos_last, _frame_last, _ages, _lengths, _states, _covs
        nonlocal track_id_counter
        n = len(positions)
        if n == 0:
            return
        new_states = np.zeros((n, 6))
        new_states[:, :3] = positions
        _pos_last = np.concatenate([_pos_last, positions])
        _frame_last = np.concatenate(
            [_frame_last, np.full(n, frame_idx, dtype=np.int64)]
        )
        _ages = np.concatenate([_ages, np.zeros(n, dtype=np.int64)])
        _lengths = np.concatenate([_lengths, np.ones(n, dtype=np.int64)])
        _states = np.concatenate([_states, new_states])
        _covs = np.concatenate([_covs, np.tile(P_init, (n, 1, 1))])
        for di in range(n):
            _hist_pos.append([positions[di].copy()])
            _hist_fr.append([frame_idx])
            if has_intensities and det_int is not None:
                _hist_int.append([float(det_int[di])])
            else:
                _hist_int.append([])
        track_id_counter += n

    for frame_idx, dets in enumerate(detections):
        cur_int = intensities[frame_idx] if has_intensities else None
        n_tracks = len(_ages)

        if dets is None or len(dets) == 0:
            if n_tracks > 0:
                _ages += 1
                _retire(_ages > max_gap)
                _compact()
            continue

        cur_dets = np.asarray(dets)
        n_dets = len(cur_dets)

        if n_tracks == 0:
            _spawn(cur_dets, frame_idx, cur_int)
            continue

        # ---- Vectorized distance gating ----
        gaps = frame_idx - _frame_last  # (T,)
        diffs = np.abs(_pos_last[:, None, :] - cur_dets[None, :, :])  # (T, D, 3)
        valid_mask = np.all(diffs <= max_dist_mm, axis=2) & (gaps[:, None] <= max_gap)
        valid_indices = np.argwhere(valid_mask)

        if len(valid_indices) == 0:
            _ages += 1
            _retire(_ages > max_gap)
            _compact()
            _spawn(cur_dets, frame_idx, cur_int)
            continue

        # ---- Batch Kalman predict for tracks with valid pairs ----
        twv = np.unique(valid_indices[:, 0])
        n_vt = len(twv)

        tv_s = _states[twv]  # (V, 6)
        tv_c = _covs[twv]  # (V, 6, 6)
        tv_g = gaps[twv].astype(float)  # (V,)

        # x_pred[:3] = x[:3] + dt * x[3:], x_pred[3:] = x[3:]
        pred_pos = tv_s[:, :3] + tv_g[:, None] * tv_s[:, 3:]  # (V, 3)

        # P_pred = F @ P @ F.T + Q * dt  (batch, using F = I + dt*E structure)
        FP = tv_c.copy()
        FP[:, :3, :] += tv_g[:, None, None] * tv_c[:, 3:, :]
        P_pred_v = FP.copy()
        P_pred_v[:, :, :3] += tv_g[:, None, None] * FP[:, :, 3:]
        P_pred_v += Q_template[None, :, :] * tv_g[:, None, None]

        pred_cov_33 = P_pred_v[:, :3, :3]  # (V, 3, 3)
        pred_cov_inv = np.linalg.inv(pred_cov_33)  # (V, 3, 3)

        tv_vel = tv_s[:, 3:]  # (V, 3)
        tv_len = _lengths[twv]  # (V,)

        # Map global track idx -> local valid-track idx
        t2l = np.empty(n_tracks, dtype=np.int64)
        t2l[twv] = np.arange(n_vt)

        # ---- Vectorized cost computation ----
        vi = valid_indices[:, 0]  # track indices
        vj = valid_indices[:, 1]  # detection indices
        li = t2l[vi]  # local indices

        # Mahalanobis: diff @ cov_inv @ diff
        dm = cur_dets[vj] - pred_pos[li]  # (K, 3)
        cidm = np.einsum("ki,kij->kj", dm, pred_cov_inv[li])
        maha = np.sqrt(np.maximum(np.einsum("ki,ki->k", dm, cidm), 0.0))

        # Momentum cost (vectorized)
        pair_gaps = gaps[vi].astype(float)
        implied_vel = (cur_dets[vj] - _pos_last[vi]) / pair_gaps[:, None]
        v_old = tv_vel[li]
        sp_old = np.linalg.norm(v_old, axis=1)
        sp_new = np.linalg.norm(implied_vel, axis=1)

        denom = sp_old * sp_new
        denom[denom < 1e-30] = 1e-30
        dot = np.sum(v_old * implied_vel, axis=1) / denom
        momentum = np.where(dot < 0, reversal_penalty * np.abs(dot), -0.5 * dot)
        momentum[~((sp_old >= 0.1) & (sp_new >= 0.1) & (tv_len[li] >= 2))] = 0.0

        pair_costs = maha + momentum

        # ---- Reduced Hungarian: only include tracks/dets with valid pairs ----
        involved_tracks = np.unique(vi)
        involved_dets = np.unique(vj)
        nt_r = len(involved_tracks)
        nd_r = len(involved_dets)

        # Map global -> reduced indices
        t2r = np.empty(n_tracks, dtype=np.int64)
        t2r[involved_tracks] = np.arange(nt_r)
        d2r = np.empty(n_dets, dtype=np.int64)
        d2r[involved_dets] = np.arange(nd_r)

        cost_reduced = np.full((nt_r, nd_r), reject_cost)
        cost_reduced[t2r[vi], d2r[vj]] = pair_costs

        row_r, col_r = linear_sum_assignment(cost_reduced)
        costs_r = cost_reduced[row_r, col_r]
        good = costs_r < max_cost
        m_rows = involved_tracks[row_r[good]]
        m_cols = involved_dets[col_r[good]]

        # ---- Batch Kalman update for matched tracks ----
        if len(m_rows) > 0:
            mg = gaps[m_rows].astype(float)
            md = cur_dets[m_cols]
            ms = _states[m_rows]
            mc = _covs[m_rows]

            # Predict
            xp = ms.copy()
            xp[:, :3] += mg[:, None] * ms[:, 3:]
            FPm = mc.copy()
            FPm[:, :3, :] += mg[:, None, None] * mc[:, 3:, :]
            Pp = FPm.copy()
            Pp[:, :, :3] += mg[:, None, None] * FPm[:, :, 3:]
            Pp += Q_template[None, :, :] * mg[:, None, None]

            # S = P_pred[:3,:3] + R;  K = P_pred[:,:3] @ S_inv
            S = Pp[:, :3, :3] + R_template
            S_inv = np.linalg.inv(S)
            K = np.einsum("mij,mjk->mik", Pp[:, :, :3], S_inv)
            y = md - xp[:, :3]
            x_new = xp + np.einsum("mij,mj->mi", K, y)
            KH = np.einsum("mij,jk->mik", K, H)
            P_new = np.einsum("mij,mjk->mik", I6[None, :, :] - KH, Pp)

            _states[m_rows] = x_new
            _covs[m_rows] = P_new
            _pos_last[m_rows] = md
            _frame_last[m_rows] = frame_idx
            _ages[m_rows] = 0
            _lengths[m_rows] += 1

            for mi, (row, col) in enumerate(zip(m_rows, m_cols)):
                _hist_pos[row].append(md[mi].copy())
                _hist_fr[row].append(frame_idx)
                if has_intensities:
                    _hist_int[row].append(float(cur_int[col]))

        # Age unmatched tracks
        unmatched_mask = np.ones(n_tracks, dtype=bool)
        unmatched_mask[m_rows] = False
        _ages[unmatched_mask] += 1

        # New tracks for unmatched detections
        matched_det_mask = np.zeros(n_dets, dtype=bool)
        matched_det_mask[m_cols] = True
        um_dets = np.where(~matched_det_mask)[0]
        if len(um_dets) > 0:
            _spawn(
                cur_dets[um_dets],
                frame_idx,
                cur_int[um_dets] if has_intensities else None,
            )

        # Retire old tracks
        _retire(_ages > max_gap)
        _compact()

    # Finalize all remaining
    if len(_ages) > 0:
        _retire(np.ones(len(_ages), dtype=bool))

    return tracks


def _track_detections(
    detections: list[np.ndarray],
    intensities: list[np.ndarray],
    opts: TrackingOptions,
    spacing: dict[str, float],
) -> list[dict]:
    max_dist = np.asarray(_tracking_gate(opts, spacing), dtype=np.float32)
    if opts.tracking == "kalman":
        return kalman_tracking_3d(
            detections,
            max_distance_mm=tuple(float(v) for v in max_dist),
            max_gap=opts.max_gap,
            min_track_length=opts.min_track_length,
            reversal_penalty=opts.reversal_penalty,
            max_cost=opts.max_cost,
            intensities=intensities,
        )
    active: list[dict] = []
    done: list[dict] = []
    next_id = 0

    for frame_idx, dets in enumerate(detections):
        dets = np.asarray(dets, dtype=np.float32)
        det_int = np.asarray(intensities[frame_idx], dtype=np.float32)
        if len(active) == 0:
            for idx, det in enumerate(dets):
                active.append(
                    {
                        "id": next_id,
                        "positions": [det.copy()],
                        "frames": [frame_idx],
                        "intensities": [float(det_int[idx])] if len(det_int) else [],
                        "age": 0,
                    }
                )
                next_id += 1
            continue

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        if len(dets):
            pred = []
            gaps = []
            velocities = []
            for track in active:
                gap = max(1, frame_idx - int(track["frames"][-1]))
                gaps.append(gap)
                positions = track["positions"]
                if opts.tracking == "kalman" and len(positions) >= 2:
                    prev_gap = max(1, int(track["frames"][-1]) - int(track["frames"][-2]))
                    velocity = (positions[-1] - positions[-2]) / float(prev_gap)
                else:
                    velocity = np.zeros(3, dtype=np.float32)
                velocities.append(velocity)
                pred.append(positions[-1] + velocity * gap)

            pred_arr = np.asarray(pred, dtype=np.float32)
            gaps_arr = np.asarray(gaps, dtype=np.float32)
            velocities_arr = np.asarray(velocities, dtype=np.float32)
            diffs = dets[None, :, :] - pred_arr[:, None, :]
            allowed = max_dist[None, None, :] * gaps_arr[:, None, None]
            norm_sq = np.sum((diffs / np.maximum(allowed, 1e-6)) ** 2, axis=2)
            cost = np.linalg.norm(diffs, axis=2)
            valid = (norm_sq <= 1.0) & (gaps_arr[:, None] <= opts.max_gap)
            cost_matrix = np.full(cost.shape, opts.max_cost * 10.0, dtype=np.float32)
            cost_matrix[valid] = cost[valid]

            if opts.tracking == "kalman":
                implied = (dets[None, :, :] - pred_arr[:, None, :]) / gaps_arr[:, None, None]
                old_speed = np.linalg.norm(velocities_arr, axis=1)[:, None]
                new_speed = np.linalg.norm(implied, axis=2)
                denom = np.maximum(old_speed * new_speed, 1e-12)
                dot = np.sum(velocities_arr[:, None, :] * implied, axis=2) / denom
                reversal = np.where(dot < 0, opts.reversal_penalty * np.abs(dot), 0.0)
                reversal[(old_speed < 0.1) | (new_speed < 0.1)] = 0.0
                cost_matrix[valid] += reversal[valid]

            rows, cols = linear_sum_assignment(cost_matrix)
            for row, col in zip(rows, cols):
                if cost_matrix[row, col] >= opts.max_cost:
                    continue
                track = active[int(row)]
                track["positions"].append(dets[int(col)].copy())
                track["frames"].append(frame_idx)
                if len(det_int):
                    track["intensities"].append(float(det_int[int(col)]))
                track["age"] = 0
                matched_tracks.add(int(row))
                matched_dets.add(int(col))

        for idx, track in enumerate(active):
            if idx not in matched_tracks:
                track["age"] += 1
        for idx, det in enumerate(dets):
            if idx not in matched_dets:
                active.append(
                    {
                        "id": next_id,
                        "positions": [det.copy()],
                        "frames": [frame_idx],
                        "intensities": [float(det_int[idx])] if len(det_int) else [],
                        "age": 0,
                    }
                )
                next_id += 1

        kept = []
        for track in active:
            if int(track["age"]) > opts.max_gap:
                if len(track["positions"]) >= opts.min_track_length:
                    done.append(_finalize_track(track))
            else:
                kept.append(track)
        active = kept

    for track in active:
        if len(track["positions"]) >= opts.min_track_length:
            done.append(_finalize_track(track))
    return done


def _finalize_track(track: dict) -> dict:
    out = {
        "id": int(track["id"]),
        "positions": np.asarray(track["positions"], dtype=np.float32),
        "frames": np.asarray(track["frames"], dtype=np.float32),
        "length": int(len(track["positions"])),
    }
    if track.get("intensities"):
        out["intensities"] = np.asarray(track["intensities"], dtype=np.float32)
    return out


def compute_zscore_knee(zscores: np.ndarray) -> float:
    """Z-score value at the knee of the sorted z-score curve (2nd-derivative max).

    Marks the transition from signal to noise so detections below it can be
    dropped. Returns NaN if there are too few detections.
    """
    zscores = np.sort(np.asarray(zscores, dtype=np.float64))[::-1]
    n = len(zscores)
    if n < 100:
        return float("nan")
    smooth = gaussian_filter1d(zscores, sigma=max(1, n // 100))
    d2 = np.gradient(np.gradient(smooth))
    start = n // 20  # skip the top 5% outliers
    knee_idx = int(np.argmax(d2[start:])) + start
    return float(zscores[knee_idx])


def _knee_filter_batch(batch: list, opts: TrackingOptions) -> list:
    """Drop per-acquisition detections below the z-score knee (adaptive mode)."""
    if not (opts.knee_filter and opts.svd_method == "adaptive"):
        return batch
    zs_all = [z for (_, _, z) in batch if len(z)]
    if not zs_all:
        return batch
    knee = compute_zscore_knee(np.concatenate(zs_all))
    if not np.isfinite(knee):
        return batch
    out = []
    for pixels, intensities, zscores in batch:
        if len(zscores):
            mask = zscores >= knee
            out.append((pixels[mask], intensities[mask], zscores[mask]))
        else:
            out.append((pixels, intensities, zscores))
    return out


def _resolve_frame_rate(opts: TrackingOptions, h5) -> TrackingOptions:
    """Take the frame rate from the beamformed file unless one was given.

    An explicit --frame-rate always wins; this only fills the gap, and only
    from a value the beamform stage carried over from the source acquisition.
    """
    if opts.frame_rate_hz is not None:
        return opts
    if "frame_rate_hz" not in h5.attrs:
        return opts

    frame_rate_hz = float(h5.attrs["frame_rate_hz"])
    print(f"Frame rate {frame_rate_hz:.4f} Hz (from {opts.beamformed_path.name})")
    return replace(opts, frame_rate_hz=frame_rate_hz)


def _resolve_tissue_threshold(opts: TrackingOptions, h5) -> TrackingOptions:
    """Convert a velocity threshold to this file's Doppler frequency threshold.

    A Hz threshold is tied to the carrier it was chosen at; a mm/s one is not,
    so ``--tissue-velocity`` is converted per file using the tx frequency and
    speed of sound recorded at beamform time.
    """
    if opts.tissue_velocity_mm_s is None:
        return opts

    missing = [k for k in ("tx_freq_hz", "speed_of_sound_m_s") if k not in h5.attrs]
    if missing:
        raise ValueError(
            f"--tissue-velocity needs {', '.join(missing)} in {opts.beamformed_path.name}, "
            "but this file was beamformed before those were recorded. Re-beamform, "
            "or pass --tissue-freq in Hz instead."
        )

    tx_freq_hz = float(h5.attrs["tx_freq_hz"])
    speed_of_sound_m_s = float(h5.attrs["speed_of_sound_m_s"])
    freq_hz = doppler_velocity_to_freq(
        opts.tissue_velocity_mm_s, tx_freq_hz, speed_of_sound_m_s
    )
    print(
        f"Tissue boundary {opts.tissue_velocity_mm_s:g} mm/s -> {freq_hz:.1f} Hz "
        f"(tx {tx_freq_hz / 1e6:g} MHz, c {speed_of_sound_m_s:g} m/s)"
    )
    return replace(opts, tissue_freq_hz=freq_hz)


def _filter_acquisition(compound: np.ndarray, opts: TrackingOptions) -> np.ndarray:
    if opts.filter_method == "none":
        mag = np.abs(compound).astype(np.float32, copy=False)
        if opts.temporal_sigma > 0:
            mag = gaussian_filter1d(mag, sigma=opts.temporal_sigma, axis=0)
        return mag
    return filtered_magnitude(
        compound,
        low_cutoff=opts.svd_low_cutoff,
        high_cutoff=opts.svd_high_cutoff,
        method=opts.svd_method,
        temporal_sigma=opts.temporal_sigma,
        n_components=opts.svd_n_components,
        frame_rate_hz=opts.frame_rate_hz,
        tissue_freq_hz=opts.tissue_freq_hz,
    )


def _run_selected(opts: TrackingOptions, selected: list[int], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    detections_by_frame: list[np.ndarray] = []
    intensities_by_frame: list[np.ndarray] = []
    all_indices = []
    all_positions = []
    all_intensities = []
    all_zscores = []
    all_frames = []
    all_acqs = []
    grid_x = grid_y = grid_z = None
    frames_per_acq = 0

    with open_h5(opts.beamformed_path) as h5:
        for acq_order, acq_id in enumerate(selected):
            compound = load_compound(h5, acq_id)
            if frames_per_acq == 0:
                frames_per_acq = int(compound.shape[0])
            grid_x, grid_y, grid_z = grid_arrays(h5, acq_id)
            filtered = _filter_acquisition(compound, opts)
            batch = detect_batch(
                filtered,
                sigma_threshold=opts.sigma_threshold,
                min_distance=opts.min_distance,
                smoothing_sigma=opts.smoothing_sigma,
            )
            batch = _knee_filter_batch(batch, opts)
            print(f"[track] acq={acq_id} frames={filtered.shape[0]} shape={filtered.shape[1:]}")
            for frame_in_acq, (pixels, intensities, zscores) in enumerate(batch):
                global_frame = acq_order * int(compound.shape[0]) + frame_in_acq
                if len(pixels) == 0:
                    detections_by_frame.append(np.empty((0, 3), dtype=np.float32))
                    intensities_by_frame.append(np.empty(0, dtype=np.float32))
                    continue
                subpix = subpixel_localize_3d(
                    filtered[frame_in_acq],
                    pixels,
                    method=opts.subpixel,
                    window_size=opts.window_size,
                )
                positions = indices_to_mm(subpix, grid_x, grid_y, grid_z)
                detections_by_frame.append(positions)
                intensities_by_frame.append(intensities)
                all_indices.append(subpix)
                all_positions.append(positions)
                all_intensities.append(intensities)
                all_zscores.append(zscores)
                all_frames.append(np.full(len(positions), global_frame, dtype=np.int32))
                all_acqs.append(np.full(len(positions), int(acq_id), dtype=np.int32))

    if grid_x is None or grid_y is None or grid_z is None:
        raise RuntimeError("No acquisitions processed")

    spacing = _grid_spacing(grid_x, grid_y, grid_z)
    tracks = _track_detections(detections_by_frame, intensities_by_frame, opts, spacing)
    data = {
        "tracks": tracks,
        "density": None,
        "detections": {
            "indices": np.concatenate(all_indices) if all_indices else np.empty((0, 3), dtype=np.float32),
            "positions_mm": np.concatenate(all_positions) if all_positions else np.empty((0, 3), dtype=np.float32),
            "intensities": np.concatenate(all_intensities) if all_intensities else np.empty(0, dtype=np.float32),
            "zscores": np.concatenate(all_zscores) if all_zscores else np.empty(0, dtype=np.float32),
            "frame_indices": np.concatenate(all_frames) if all_frames else np.empty(0, dtype=np.int32),
            "acq_indices": np.concatenate(all_acqs) if all_acqs else np.empty(0, dtype=np.int32),
        },
        "detections_by_frame": detections_by_frame,
        "intensities_by_frame": intensities_by_frame,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "grid_z": grid_z,
        "spacing": spacing,
        "n_acquisitions": len(selected),
        "frames_per_acq": int(frames_per_acq),
        "n_frames": int(len(detections_by_frame)),
        "params": {
            "standalone_tracker": True,
            "selected_acq_ids": [int(v) for v in selected],
            "frame_origin": "selected_sequence",
            "sigma_threshold": float(opts.sigma_threshold),
            "svd_low_cutoff": float(opts.svd_low_cutoff),
            "svd_high_cutoff": opts.svd_high_cutoff,
            "svd_n_components": opts.svd_n_components,
            "svd_method": opts.svd_method,
            "knee_filter": bool(opts.knee_filter),
            "tissue_freq_hz": float(opts.tissue_freq_hz),
            "filter_method": opts.filter_method,
            "temporal_sigma": float(opts.temporal_sigma),
            "min_distance": int(opts.min_distance),
            "smoothing_sigma": float(opts.smoothing_sigma),
            "subpixel_method": opts.subpixel,
            "window_size": int(opts.window_size),
            "tracking_method": opts.tracking,
            "max_distance_mm": _tracking_gate(opts, spacing),
            "frame_rate_hz": opts.frame_rate_hz,
            "max_gap": int(opts.max_gap),
            "min_track_length": int(opts.min_track_length),
            "max_cost": float(opts.max_cost),
        },
    }
    dump_pickle(data, output_path)
    print(f"Wrote {len(tracks)} tracks -> {output_path}")
    return output_path


def run_tracking(opts: TrackingOptions) -> Path:
    with open_h5(opts.beamformed_path) as h5:
        selected = select_acquisitions(acq_keys(h5), opts.acq_start, opts.num_acqs, opts.acq_step)
        opts = _resolve_frame_rate(opts, h5)
        opts = _resolve_tissue_threshold(opts, h5)

    if not opts.output_per_acq:
        return _run_selected(opts, selected, opts.tracks_path)

    for acq_id in selected:
        path = opts.tracks_path.with_name(f"{opts.tracks_path.stem}_acq{int(acq_id):04d}.pkl")
        if path.exists():
            print(f"[track] keeping existing {path}")
            continue
        _run_selected(opts, [int(acq_id)], path)
    return combine_per_acq_pickles(str(opts.tracks_path.with_name(f"{opts.tracks_path.stem}_acq*.pkl")), opts.tracks_path)


def combine_per_acq_pickles(pattern: str, output_path: Path) -> Path:
    pattern_path = Path(pattern)
    paths = sorted(pattern_path.parent.glob(pattern_path.name), key=_acq_id_from_path)
    if not paths:
        raise SystemExit(f"No per-acquisition pickles matched {pattern}")

    combined_tracks = []
    selected_ids = []
    base = None
    frames_per_acq = None
    for order, path in enumerate(paths):
        acq = _acq_id_from_path(path)
        selected_ids.append(acq)
        data = load_pickle(path)
        if base is None:
            base = dict(data)
        fp = int(data.get("frames_per_acq") or data.get("n_frames") or 0)
        if fp <= 0:
            raise ValueError(f"Cannot determine frames_per_acq for {path}")
        frames_per_acq = fp
        for track in data.get("tracks", []):
            positions = np.asarray(track["positions"], dtype=np.float32)
            frames = np.asarray(track["frames"], dtype=np.float32) + order * fp
            out = dict(track)
            out["positions"] = positions
            out["frames"] = frames
            out["length"] = int(len(frames))
            out["acq_index"] = int(acq)
            combined_tracks.append(out)

    assert base is not None and frames_per_acq is not None
    base["tracks"] = combined_tracks
    base["n_acquisitions"] = len(paths)
    base["frames_per_acq"] = int(frames_per_acq)
    base["n_frames"] = int(frames_per_acq * len(paths))
    base.setdefault("params", {})
    base["params"].update(
        {
            "boundary_safe_per_acq": True,
            "combined_from_per_acq": pattern,
            "selected_acq_ids": selected_ids,
            "frame_origin": "selected_sequence",
        }
    )
    base.pop("detections_by_frame", None)
    base.pop("intensities_by_frame", None)
    dump_pickle(base, output_path)
    print(f"Combined {len(paths)} per-acquisition pickles -> {output_path}")
    return output_path


def _moving_average_smooth(data: np.ndarray, window: int = 5) -> np.ndarray:
    """Moving-average smoothing along axis 0.

    Boundary points use a shrinking, symmetric average instead of zero-padding,
    so the endpoints are not pulled toward zero. A fractional ``window`` (<1) is
    interpreted as a fraction of the track length (made odd).
    """
    if data.ndim < 2:
        data = np.expand_dims(data, 1)
    if data.shape[0] < window:
        return data
    if window < 1:
        window = ceil(data.shape[0] * window)
        window = int(window - 1 + (window % 2))
    window = int(window)
    mask = np.ones((window, 1)) / window
    out = scipy.signal.convolve(data, mask, "valid")
    r = np.arange(1, window - 1, 2)[:, None]
    head = np.cumsum(data[: window - 1, :], axis=0)[::2, :] / r
    tail = np.cumsum(data[:-window:-1, :], axis=0)[::2, :] / r
    tail = tail[::-1, :]
    return np.concatenate((head, out, tail), axis=0)


def _curvilinear_abscissa(points: np.ndarray) -> np.ndarray:
    """Cumulative arc length along a polyline of (N, D) points."""
    seg = np.linalg.norm(np.diff(points, axis=0), ord=2, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def _clean_and_interpolate_track(
    pos: np.ndarray,
    scale: np.ndarray,
    index_frames: np.ndarray,
    interp_factor: float,
    smooth_window: int = 5,
    intensities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Arc-length track cleaning: smooth, then resample uniformly in arc
    length (not frame index) and recover a fractional-frame timeline.

    Returns (positions, velocities_mm_per_s, frames, intensities) all aligned
    to the resampled point count (``intensities`` is None when not supplied).
    Brightness is interpolated against the same arc-length parameterization as
    position. Falls back to the input track when it is too short or degenerate
    (zero total arc length).
    """
    pos = _moving_average_smooth(np.asarray(pos, dtype=np.float64), window=smooth_window)
    inten = None if intensities is None else np.asarray(intensities, dtype=np.float64)
    ca = _curvilinear_abscissa(pos)
    step = float(interp_factor) * float(np.min(scale[:3]))
    if ca[-1] <= 0.0 or step <= 0.0:
        return pos, np.zeros_like(pos), np.asarray(index_frames, dtype=np.float64), inten
    ca_interp = np.arange(ca[0], ca[-1], step)
    if len(ca_interp) < 2:
        return pos, np.zeros_like(pos), np.asarray(index_frames, dtype=np.float64), inten

    pos_i = np.zeros([len(ca_interp), pos.shape[1]], dtype=pos.dtype)
    for axis in range(pos_i.shape[1]):
        pos_i[:, axis] = np.interp(ca_interp, ca, pos[:, axis])
    pos_i = _moving_average_smooth(pos_i, scale[2] * 2)
    # Brightness rides the same arc-length grid as the pre-resample positions.
    int_i = None if inten is None else np.interp(ca_interp, ca, inten)

    ca_i = _curvilinear_abscissa(pos_i)
    tl = np.asarray(index_frames, dtype=np.float64) * scale[-1]  # timeline in seconds
    tl_i = np.interp(ca_i / ca_i[-1], ca / ca[-1], tl)
    dt_line = np.diff(tl_i)
    dt_line[dt_line == 0] = 1e-12
    vel = np.diff(pos_i, axis=0) / dt_line[:, None]
    vel = np.vstack([vel[0, :], vel])
    frames_i = tl_i / scale[-1]
    # The second smoothing pass can change pos_i's length; keep brightness aligned.
    if int_i is not None and len(int_i) != len(pos_i):
        int_i = np.interp(
            np.linspace(0.0, 1.0, len(pos_i)),
            np.linspace(0.0, 1.0, len(int_i)),
            int_i,
        )
    return pos_i, vel, frames_i, int_i


def _smooth_tracks(
    tracks: list[dict],
    sigma: float,
    method: str = "gaussian",
    scale: np.ndarray | None = None,
    interp_factor: float = 0.1,
    window: int = 5,
) -> list[dict]:
    """Post-process track positions.

    method="gaussian" (default): per-axis 1D Gaussian along the track.
    method="3dulm": arc-length resampling/interpolation (needs ``scale``,
    a length-4 [dx, dy, dz, dt] array). 3dulm changes the point count; per-point
    ``intensities`` (brightness) are interpolated onto the resampled points.
    """
    smoothed = []
    for track in tracks:
        positions = np.asarray(track["positions"], dtype=np.float32)
        frames = np.asarray(track["frames"], dtype=np.float32)
        track_int = track.get("intensities")
        out = dict(track)
        if method == "3dulm":
            if scale is None:
                raise ValueError("3dulm smoothing requires a [dx, dy, dz, dt] scale")
            if len(positions) >= 3:
                pos_i, _vel, frames_i, int_i = _clean_and_interpolate_track(
                    positions, np.asarray(scale, dtype=np.float64),
                    frames, interp_factor, window, intensities=track_int,
                )
                out["positions"] = pos_i.astype(np.float32)
                out["frames"] = frames_i.astype(np.float32)
                out["length"] = int(len(pos_i))
                if int_i is not None:
                    out["intensities"] = int_i.astype(np.float32)
            else:
                out["positions"] = positions
                out["frames"] = frames
                out["length"] = int(len(positions))
        else:
            if len(positions) >= 3:
                positions = positions.copy()
                for axis in range(3):
                    positions[:, axis] = gaussian_filter1d(positions[:, axis], sigma=sigma, mode="nearest")
            out["positions"] = positions
            out["frames"] = frames
            out["length"] = int(len(positions))
        smoothed.append(out)
    return smoothed


def _scale_from_pickle(data: dict) -> np.ndarray:
    """Build a [dx, dy, dz, dt] scale from a tracks pickle's spacing + frame rate."""
    spacing = data.get("spacing") or {}
    dx, dy, dz = spacing.get("dx"), spacing.get("dy"), spacing.get("dz")
    fr = (data.get("params") or {}).get("frame_rate_hz")
    if None in (dx, dy, dz) or not fr:
        raise SystemExit(
            "3dulm smoothing needs spacing dx/dy/dz and params.frame_rate_hz in the pickle"
        )
    return np.array([dx, dy, dz, 1.0 / float(fr)], dtype=np.float64)


def smooth_tracks_pickle(
    tracks_path: Path,
    output_path: Path | None,
    sigma: float,
    method: str = "gaussian",
    interp_factor: float = 0.1,
    window: int = 5,
) -> Path:
    data = load_pickle(tracks_path)
    scale = _scale_from_pickle(data) if method == "3dulm" else None
    smoothed = _smooth_tracks(
        data.get("tracks", []), sigma=sigma, method=method,
        scale=scale, interp_factor=interp_factor, window=window,
    )
    data["tracks_smoothed"] = smoothed
    data.setdefault("params", {})
    data["params"].update(
        {
            "post_smoothing_method": method,
            "post_smoothing_sigma": float(sigma),
            "post_smoothing_interp_factor": float(interp_factor),
            "post_smoothing_window": int(window),
        }
    )
    out = output_path or tracks_path.with_name(f"{tracks_path.stem}_smoothed.pkl")
    dump_pickle(data, out)
    print(f"Smoothed {len(smoothed)} tracks ({method}) -> {out}")
    return out


def _speed_mm_s(positions: np.ndarray, frames: np.ndarray, frame_rate_hz: float) -> np.ndarray:
    n = len(positions)
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    dp = np.diff(positions, axis=0)
    df = np.diff(frames).astype(np.float64)
    df[df == 0] = 1.0
    velocity = dp / df[:, None]
    full = np.empty((n, 3), dtype=np.float64)
    full[0] = velocity[0]
    full[-1] = velocity[-1]
    full[1:-1] = 0.5 * (velocity[:-1] + velocity[1:])
    if n > 6:
        for axis in range(3):
            full[:, axis] = gaussian_filter1d(full[:, axis], sigma=3.0, mode="nearest")
    return (np.linalg.norm(full, axis=1) * frame_rate_hz).astype(np.float32)


def export_tracks_bin(
    pickle_path: Path,
    output_path: Path,
    min_length: int,
    use_smoothed: bool = True,
) -> Path:
    data = load_pickle(pickle_path)
    key = "tracks_smoothed" if use_smoothed and "tracks_smoothed" in data else "tracks"
    tracks = [t for t in data.get(key, []) if int(t.get("length", len(t["positions"]))) >= min_length]
    if not tracks:
        raise SystemExit(f"No tracks with length >= {min_length} in {pickle_path}")

    frame_rate_hz = float(data.get("params", {}).get("frame_rate_hz") or 1.0)
    n_acqs = int(data.get("n_acquisitions") or data.get("n_acqs") or 0)
    frames_per_acq = int(data.get("frames_per_acq") or 0)
    if frames_per_acq == 0 and n_acqs > 0:
        max_frame = max(float(np.asarray(t["frames"]).max()) for t in tracks)
        frames_per_acq = int(max_frame / n_acqs) + 1

    all_mins = []
    all_maxs = []
    speeds = []
    max_speed = 0.0
    for track in tracks:
        positions = np.asarray(track["positions"], dtype=np.float32)
        frames = np.asarray(track["frames"], dtype=np.float32)
        all_mins.append(positions.min(axis=0))
        all_maxs.append(positions.max(axis=0))
        s = _speed_mm_s(positions, frames, frame_rate_hz)
        speeds.append(s)
        if len(s):
            max_speed = max(max_speed, float(s.max()))

    bounds_min = np.min(all_mins, axis=0).astype(np.float32)
    bounds_max = np.max(all_maxs, axis=0).astype(np.float32)
    total_points = sum(int(t.get("length", len(t["positions"]))) for t in tracks)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as fp:
        header = struct.pack(
            "<IIIIf3f3fIIf",
            0x554C4D54,
            4,
            len(tracks),
            total_points,
            max_speed,
            *bounds_min.tolist(),
            *bounds_max.tolist(),
            frames_per_acq,
            n_acqs,
            frame_rate_hz,
        )
        fp.write(header + b"\x00" * (64 - len(header)))

        offset = 0
        for track in tracks:
            length = int(track.get("length", len(track["positions"])))
            if "acq_index" in track:
                acq_index = int(track["acq_index"])
            elif frames_per_acq:
                acq_index = int(float(np.asarray(track["frames"])[0]) // frames_per_acq)
            else:
                acq_index = 0
            fp.write(struct.pack("<IIHH", offset, length, acq_index, 0))
            offset += length

        for track, speed in zip(tracks, speeds):
            positions = np.asarray(track["positions"], dtype=np.float32)
            frames = np.asarray(track["frames"], dtype=np.float32)
            points = np.column_stack([positions, frames[:, None], speed[:, None]])
            fp.write(points.astype(np.float32).tobytes())

    meta = {
        "source_pickle": str(pickle_path),
        "track_key": key,
        "min_length": int(min_length),
        "n_tracks": len(tracks),
        "total_points": int(total_points),
        "frame_rate_hz": frame_rate_hz,
        "frames_per_acq": frames_per_acq,
        "n_acquisitions": n_acqs,
        "bounds_min": bounds_min.tolist(),
        "bounds_max": bounds_max.tolist(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Exported {len(tracks)} tracks -> {output_path}")
    return output_path


def run_tracking_outputs(opts: TrackingOptions) -> Path:
    tracks = run_tracking(opts)
    smoothed = smooth_tracks_pickle(
        tracks, None, sigma=opts.smooth_sigma, method=opts.smooth_method,
        interp_factor=opts.smooth_interp_factor, window=opts.smooth_window,
    )
    if opts.export_dir:
        stem = opts.export_stem or tracks.stem
        for min_length in opts.export_min_lengths:
            export_tracks_bin(
                smoothed,
                opts.export_dir / f"{stem}_min{min_length}.bin",
                min_length=min_length,
            )
    return smoothed


def _tuple_arg(value) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if len(value) != 3:
        raise ValueError("Expected three values")
    return (float(value[0]), float(value[1]), float(value[2]))


def make_options(args) -> TrackingOptions:
    return TrackingOptions(
        beamformed_path=Path(args.beamformed).expanduser().resolve(),
        tracks_path=Path(args.tracks).expanduser().resolve(),
        num_acqs=args.num_acqs,
        acq_start=args.acq_start,
        acq_step=args.acq_step,
        output_per_acq=args.output_per_acq,
        sigma_threshold=args.sigma_threshold,
        svd_low_cutoff=args.svd_low_cutoff,
        svd_high_cutoff=args.svd_high_cutoff,
        svd_n_components=args.svd_n_components,
        svd_method=args.svd_method,
        knee_filter=getattr(args, "knee_filter", True),
        tissue_freq_hz=getattr(args, "tissue_freq_hz", 100.0),
        tissue_velocity_mm_s=getattr(args, "tissue_velocity_mm_s", None),
        temporal_sigma=args.temporal_sigma,
        filter_method=args.filter_method,
        min_distance=args.min_distance,
        smoothing_sigma=args.smoothing_sigma,
        subpixel=args.subpixel,
        window_size=args.window_size,
        patch_radius=args.patch_radius,
        tracking=args.tracking,
        frame_rate_hz=args.frame_rate_hz,
        max_dist=_tuple_arg(args.max_dist),
        max_dist_mms=_tuple_arg(args.max_dist_mms),
        max_gap=args.max_gap,
        min_track_length=args.min_track_length,
        reversal_penalty=args.reversal_penalty,
        max_cost=args.max_cost,
        smooth_sigma=args.smooth_sigma,
        smooth_method=getattr(args, "smooth_method", "gaussian"),
        smooth_window=getattr(args, "smooth_window", 5),
        smooth_interp_factor=getattr(args, "smooth_interp_factor", 0.1),
        export_dir=Path(args.export_dir).expanduser().resolve() if args.export_dir else None,
        export_stem=args.export_stem,
        export_min_lengths=tuple(args.min_lengths),
    )
