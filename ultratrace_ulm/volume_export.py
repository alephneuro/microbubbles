from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from .h5_io import acq_keys, axis_bounds, grid_arrays, load_compound, open_h5, select_acquisitions
from .runtime import load_pickle
from .svd import filtered_magnitude


@dataclass(frozen=True)
class VolumeExportOptions:
    beamformed_path: Path
    output_dir: Path
    tracks_path: Path | None = None
    acq_start: int = 0
    num_acqs: int = 1
    acq_step: int = 1
    svd_low_cutoff: float = 0.1
    svd_high_cutoff: float | None = None
    svd_method: str = "fast"
    temporal_sigma: float = 0.0
    dynamic_range_db: float = 15.0
    percentile: float = 99.7
    voxel_percentile: float = 99.9
    max_points_per_frame: int = 8000
    background_percentile: float = 99.8
    background_dynamic_range_db: float = 45.0
    background_max_points_per_frame: int = 18000
    background_intro_seconds: float = 1.5
    subtraction_fade_seconds: float = 1.0
    fps: float = 30.0
    track_min_length: int = 5
    prefer_smoothed_tracks: bool = True
    tail_frames: int = 18
    max_frames: int | None = None


def _copy_web_assets(output_dir: Path) -> None:
    asset_root = resources.files("ultratrace_ulm.web.volume_viewer")
    for name in ["index.html", "app.js", "styles.css"]:
        with resources.as_file(asset_root / name) as src:
            shutil.copyfile(src, output_dir / name)


def _encode_sparse_points(volume: np.ndarray, opts: VolumeExportOptions) -> tuple[bytes, list[int], dict]:
    scale = float(np.percentile(volume, opts.percentile))
    scale = max(scale, np.finfo(np.float32).eps)
    threshold = float(np.percentile(volume, opts.voxel_percentile))
    counts: list[int] = []
    chunks = []

    for frame in volume:
        flat = frame.reshape(-1)
        candidates = np.flatnonzero(flat >= threshold)
        if len(candidates) > opts.max_points_per_frame:
            values = flat[candidates]
            keep = np.argpartition(values, -opts.max_points_per_frame)[-opts.max_points_per_frame :]
            candidates = candidates[keep]
        values = flat[candidates]
        if len(values):
            order = np.argsort(values)
            candidates = candidates[order]
            values = values[order]

        elev, z, x = np.unravel_index(candidates, frame.shape)
        db = 20.0 * np.log10(np.maximum(values, np.finfo(np.float32).eps) / scale)
        db = np.clip(db, -opts.dynamic_range_db, 0.0)
        intensity = ((db + opts.dynamic_range_db) / opts.dynamic_range_db * 255.0).astype(np.uint8)
        records = np.zeros(
            len(candidates),
            dtype=[
                ("x", "<u2"),
                ("y", "<u2"),
                ("z", "<u2"),
                ("i", "u1"),
                ("pad", "u1"),
            ],
        )
        records["x"] = x.astype(np.uint16)
        records["y"] = elev.astype(np.uint16)
        records["z"] = z.astype(np.uint16)
        records["i"] = intensity
        chunks.append(records.tobytes())
        counts.append(int(len(candidates)))

    display = {
        "scale_percentile": opts.percentile,
        "scale_value": scale,
        "voxel_percentile": opts.voxel_percentile,
        "voxel_threshold": threshold,
        "dynamic_range_db": opts.dynamic_range_db,
        "max_points_per_frame": opts.max_points_per_frame,
    }
    return b"".join(chunks), counts, display


def _background_indices(shape: tuple[int, int, int], max_points: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(np.prod(shape))
    count = min(total, max(1, int(max_points)))
    rng = np.random.default_rng(0)
    flat_indices = rng.choice(total, size=count, replace=False)
    flat_indices.sort()
    y_idx, z_idx, x_idx = np.unravel_index(flat_indices, shape)
    return (
        flat_indices.astype(np.int64, copy=False),
        y_idx.astype(np.uint16, copy=False),
        z_idx.astype(np.uint16, copy=False),
        x_idx.astype(np.uint16, copy=False),
    )


def _encode_background_points(volume: np.ndarray, opts: VolumeExportOptions) -> tuple[bytes, bytes, list[int], dict]:
    scale = float(np.percentile(volume, opts.background_percentile))
    scale = max(scale, np.finfo(np.float32).eps)
    flat_indices, y_idx, z_idx, x_idx = _background_indices(
        volume.shape[1:],
        opts.background_max_points_per_frame,
    )
    counts: list[int] = []
    intensity_chunks = []
    position_records = np.zeros(
        len(flat_indices),
        dtype=[
            ("x", "<u2"),
            ("y", "<u2"),
            ("z", "<u2"),
        ],
    )
    position_records["x"] = x_idx
    position_records["y"] = y_idx
    position_records["z"] = z_idx

    for frame in volume:
        values = frame.reshape(-1)[flat_indices]
        db = 20.0 * np.log10(np.maximum(values, np.finfo(np.float32).eps) / scale)
        db = np.clip(db, -opts.background_dynamic_range_db, 0.0)
        intensity = ((db + opts.background_dynamic_range_db) / opts.background_dynamic_range_db * 255.0).astype(
            np.uint8
        )
        intensity_chunks.append(intensity.tobytes())
        counts.append(int(len(values)))

    display = {
        "scale_percentile": opts.background_percentile,
        "scale_value": scale,
        "dynamic_range_db": opts.background_dynamic_range_db,
        "max_points_per_frame": opts.background_max_points_per_frame,
        "sample_count": int(len(flat_indices)),
        "sample_seed": 0,
        "mode": "random_full_volume_sample",
    }
    return position_records.tobytes(), b"".join(intensity_chunks), counts, display


def _encode_background_planes(volume: np.ndarray, opts: VolumeExportOptions) -> tuple[dict[str, bytes], dict]:
    scale = float(np.percentile(volume, opts.background_percentile))
    scale = max(scale, np.finfo(np.float32).eps)

    planes = {
        "xz": volume.max(axis=1),
        "xy": volume.max(axis=2),
        "yz": volume.max(axis=3).transpose(0, 2, 1),
    }
    encoded: dict[str, bytes] = {}
    plane_meta: dict[str, dict] = {}
    for name, frames in planes.items():
        db = 20.0 * np.log10(np.maximum(frames, np.finfo(np.float32).eps) / scale)
        db = np.clip(db, -opts.background_dynamic_range_db, 0.0)
        pixels = ((db + opts.background_dynamic_range_db) / opts.background_dynamic_range_db * 255.0).astype(np.uint8)
        encoded[name] = pixels.tobytes()
        plane_meta[name] = {
            "file": f"background_{name}.raw",
            "width": int(pixels.shape[2]),
            "height": int(pixels.shape[1]),
        }

    display = {
        "scale_percentile": opts.background_percentile,
        "scale_value": scale,
        "dynamic_range_db": opts.background_dynamic_range_db,
        "dtype": "uint8",
        "mode": "orthogonal_raw_mip_planes",
        "planes": plane_meta,
    }
    return encoded, display


def _export_tracks(
    tracks_path: Path | None,
    output_dir: Path,
    acq_ids: list[int],
    frames_per_acq: int,
    opts: VolumeExportOptions,
) -> None:
    if tracks_path is None:
        (output_dir / "tracks.json").write_text('{"tracks":[]}\n')
        return

    data = load_pickle(tracks_path)
    key = "tracks_smoothed" if opts.prefer_smoothed_tracks and "tracks_smoothed" in data else "tracks"
    selected_tracks = data.get("params", {}).get("selected_acq_ids")
    if selected_tracks == [int(v) for v in acq_ids]:
        frame_min = 0
        frame_max = len(acq_ids) * frames_per_acq
    else:
        frame_min = acq_ids[0] * frames_per_acq
        frame_max = (acq_ids[-1] + 1) * frames_per_acq
    tracks = []
    for track_id, track in enumerate(data.get(key, [])):
        length = int(track.get("length", len(track.get("positions", []))))
        if length < opts.track_min_length:
            continue
        frames = np.asarray(track["frames"], dtype=np.float32)
        positions = np.asarray(track["positions"], dtype=np.float32)
        mask = (frames >= frame_min) & (frames < frame_max)
        if int(mask.sum()) < opts.track_min_length:
            continue
        local = frames[mask] - frame_min
        pos = positions[mask]
        tracks.append(
            {
                "id": int(track.get("id", track_id)),
                "frames": np.round(local, 3).tolist(),
                "x": np.round(pos[:, 0], 4).tolist(),
                "y": np.round(pos[:, 1], 4).tolist(),
                "z": np.round(pos[:, 2], 4).tolist(),
            }
        )

    payload = {
        "source": tracks_path.name,
        "track_key": key,
        "track_min_length": opts.track_min_length,
        "frame_offset": int(frame_min),
        "frames_per_acq": int(frames_per_acq),
        "tracks": tracks,
    }
    (output_dir / "tracks.json").write_text(json.dumps(payload) + "\n")


def export_svd_volume(opts: VolumeExportOptions) -> Path:
    opts.output_dir.mkdir(parents=True, exist_ok=True)

    with open_h5(opts.beamformed_path) as h5:
        selected = select_acquisitions(acq_keys(h5), opts.acq_start, opts.num_acqs, opts.acq_step)
        grid_x, grid_y, grid_z = grid_arrays(h5, selected[0])
        volumes = []
        backgrounds = []
        frames_per_acq = None
        for acq_id in selected:
            compound = load_compound(h5, acq_id)
            if frames_per_acq is None:
                frames_per_acq = int(compound.shape[0])
            backgrounds.append(np.abs(compound).astype(np.float32, copy=False))
            filtered = filtered_magnitude(
                compound,
                low_cutoff=opts.svd_low_cutoff,
                high_cutoff=opts.svd_high_cutoff,
                method=opts.svd_method,
                temporal_sigma=opts.temporal_sigma,
            )
            volumes.append(filtered)

    volume = np.concatenate(volumes, axis=0)
    background = np.concatenate(backgrounds, axis=0)
    if opts.max_frames is not None:
        volume = volume[: opts.max_frames]
        background = background[: opts.max_frames]
    point_bytes, counts, display = _encode_sparse_points(volume, opts)
    (opts.output_dir / "points.bin").write_bytes(point_bytes)
    background_positions, background_intensities, background_counts, background_display = _encode_background_points(
        background,
        opts,
    )
    background_planes, background_plane_display = _encode_background_planes(background, opts)
    (opts.output_dir / "background_positions.bin").write_bytes(background_positions)
    (opts.output_dir / "background_intensities.bin").write_bytes(background_intensities)
    for plane_name, plane_bytes in background_planes.items():
        (opts.output_dir / f"background_{plane_name}.raw").write_bytes(plane_bytes)

    shape = volume.shape[1:]
    meta = {
        "source": opts.beamformed_path.name,
        "points": "points.bin",
        "background_positions": "background_positions.bin",
        "background_intensities": "background_intensities.bin",
        "background_position_record_bytes": 6,
        "background_planes": background_plane_display,
        "tracks": "tracks.json",
        "record_bytes": 8,
        "frames": int(volume.shape[0]),
        "fps": float(opts.fps),
        "tail_frames": int(opts.tail_frames),
        "counts": counts,
        "background_counts": background_counts,
        "shape": {"elev": int(shape[0]), "z": int(shape[1]), "x": int(shape[2])},
        "acq_ids": [int(v) for v in selected],
        "frames_per_acq": int(frames_per_acq or 0),
        "bounds_mm": axis_bounds(grid_x, grid_y, grid_z),
        "display": display,
        "background_display": background_display,
        "presentation": {
            "background_intro_seconds": float(opts.background_intro_seconds),
            "subtraction_fade_seconds": float(opts.subtraction_fade_seconds),
        },
        "svd": {
            "low_cutoff": opts.svd_low_cutoff,
            "high_cutoff": opts.svd_high_cutoff,
            "temporal_sigma": opts.temporal_sigma,
        },
    }
    (opts.output_dir / "volume.json").write_text(json.dumps(meta, indent=2) + "\n")
    _export_tracks(opts.tracks_path, opts.output_dir, selected, int(frames_per_acq or 0), opts)
    _copy_web_assets(opts.output_dir)
    print(f"Wrote 3D SVD volume viewer bundle to {opts.output_dir}")
    return opts.output_dir / "index.html"


def make_options(args) -> VolumeExportOptions:
    return VolumeExportOptions(
        beamformed_path=Path(args.beamformed).expanduser().resolve(),
        tracks_path=Path(args.tracks).expanduser().resolve() if args.tracks else None,
        output_dir=Path(args.output_dir).expanduser().resolve(),
        acq_start=args.acq_start,
        num_acqs=args.num_acqs,
        acq_step=args.acq_step,
        svd_low_cutoff=args.svd_low_cutoff,
        svd_high_cutoff=args.svd_high_cutoff,
        svd_method=args.svd_method,
        temporal_sigma=args.temporal_sigma,
        dynamic_range_db=args.dynamic_range_db,
        percentile=args.percentile,
        voxel_percentile=args.voxel_percentile,
        max_points_per_frame=args.max_points_per_frame,
        background_percentile=args.background_percentile,
        background_dynamic_range_db=args.background_dynamic_range_db,
        background_max_points_per_frame=args.background_max_points_per_frame,
        background_intro_seconds=args.background_intro_seconds,
        subtraction_fade_seconds=args.subtraction_fade_seconds,
        fps=args.fps,
        track_min_length=args.track_min_length,
        tail_frames=args.tail_frames,
        max_frames=args.max_frames,
    )
