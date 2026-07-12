"""Standalone MACH beamforming from neutral ultratrace IQ to a beamformed H5.

Reads the neutral ultratrace schema (demodulated IQ + transmit delays + a
``/config`` scalar group) and writes a beamformed H5 whose per-acquisition layout
matches what the ``track`` command reads:

    acquisitions/<id>/meta/compound_image   complex64 (frames, elev, z, x)
    acquisitions/<id>/meta/grid/{x,y,z}     float64   (z, elev, x)

No external imaging-SDK import and no raw-frame decoding -- the neutral file
already contains demodulated IQ. The ``mach`` GPU kernel is an optional
dependency (see beamform_core).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .beamform_core import (
    NeutralConfig,
    beamform_iq,
    compute_global_tgc,
)


@dataclass(frozen=True)
class BeamformOptions:
    input_path: Path
    output_path: Path
    acq_start: int = 0
    num_acqs: int | None = 1
    acq_step: int = 1
    all_acqs: bool = False
    elev_planes: int = 25
    z_coarseness: float = 0.5
    x_coarseness: float = 0.5
    large_fov: bool = True
    xlarge_fov: bool = False
    n_chunks: int = 4
    spatial_tgc: bool = False
    tgc_acqs: int = 12
    tgc_sigma_lambda: float = 9.0
    tgc_svd_cut: float = 0.05
    resume: bool = False
    dry_run: bool = False


def selected_acquisitions(total: int, opts: BeamformOptions) -> list[int]:
    if opts.acq_step < 1:
        raise ValueError("--acq-step must be >= 1")
    if opts.acq_start < 0 or opts.acq_start >= total:
        raise ValueError(f"--acq-start must be in [0, {total - 1}]")
    if opts.all_acqs:
        end = total
    else:
        count = 1 if opts.num_acqs is None else opts.num_acqs
        if count < 1:
            raise ValueError("--num-acqs must be >= 1")
        end = min(total, opts.acq_start + count * opts.acq_step)
    ids = list(range(opts.acq_start, end, opts.acq_step))
    if not ids:
        raise ValueError("No acquisitions selected")
    return ids


# --------------------------------------------------------------------------- #
# Neutral file IO
# --------------------------------------------------------------------------- #
def _neutral_acq_ids(h5: h5py.File) -> list[int]:
    return sorted(int(k) for k in h5["acquisitions"].keys() if str(k).isdigit())


def _load_neutral_config(h5: h5py.File, opts: BeamformOptions) -> NeutralConfig:
    attrs = dict(h5["config"].attrs)
    # CLI options override the few grid knobs that may be tuned at beamform time.
    attrs = dict(attrs)
    attrs["z_coarseness"] = opts.z_coarseness
    attrs["x_coarseness"] = opts.x_coarseness
    attrs["large_fov"] = opts.large_fov
    attrs["xlarge_fov"] = opts.xlarge_fov
    attrs["num_elev_planes"] = opts.elev_planes
    return NeutralConfig.from_h5_attrs(attrs)


def _load_acq(h5: h5py.File, acq_id: int):
    g = h5[f"acquisitions/{acq_id}"]
    iq = np.asarray(g["iq_frames"], dtype=np.complex64)
    txd = np.asarray(g["tx_delays"], dtype=np.float64)
    txd_elev = np.asarray(g["tx_delays_elev"], dtype=np.float64)
    return iq, txd, txd_elev


# --------------------------------------------------------------------------- #
# Output writing
# --------------------------------------------------------------------------- #
def _write_acq(out: h5py.File, out_id: int, compound: np.ndarray, grid) -> None:
    meta = out.require_group(f"acquisitions/{out_id}/meta")
    meta.create_dataset("compound_image", data=compound.astype(np.complex64))
    gg = meta.require_group("grid")
    gg.create_dataset("x", data=grid.x.astype(np.float64))
    gg.create_dataset("y", data=grid.y.astype(np.float64))
    gg.create_dataset("z", data=grid.z.astype(np.float64))


def beamform_mach(opts: BeamformOptions) -> Path:
    if opts.dry_run:
        print("MACH beamforming dry run only; no output written.")
        return opts.output_path

    with h5py.File(opts.input_path, "r") as h5:
        config = _load_neutral_config(h5, opts)
        total = len(_neutral_acq_ids(h5))
        ids = selected_acquisitions(total, opts)

        # Pass 1 (optional): derive the spatial-TGC profile from a small subset
        # of acqs, holding only that subset in memory (not all of them).
        inv_sqrt = None
        if opts.spatial_tgc:
            tgc_ids = ids
            if opts.tgc_acqs and opts.tgc_acqs < len(ids):
                sel = np.unique(
                    np.linspace(0, len(ids) - 1, opts.tgc_acqs).round().astype(int)
                )
                tgc_ids = [ids[i] for i in sel]
            print(f"[tgc] beamforming {len(tgc_ids)} acq(s) for global TGC", flush=True)
            tgc_compounds = []
            ref_grid = None
            for acq_id in tgc_ids:
                iq, txd, txd_elev = _load_acq(h5, acq_id)
                compound, grid = beamform_iq(iq, txd, txd_elev, config)
                if ref_grid is None:
                    ref_grid = grid
                tgc_compounds.append(compound)
            print(f"[tgc] computing global TGC from {len(tgc_ids)} acq(s)", flush=True)
            inv_sqrt = compute_global_tgc(
                tgc_compounds,
                ref_grid,
                tx_freq_hz=config.tx_freq_hz,
                svd_cut=opts.tgc_svd_cut,
                sigma_lambda=opts.tgc_sigma_lambda,
            )
            del tgc_compounds

        # Pass 2: beamform every acq and stream it straight to disk, keeping at
        # most one compound volume in memory (the previous code kept all of them,
        # which OOMs for full multi-hundred-acq runs).
        opts.output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(opts.output_path, "w") as out:
            out.attrs["description"] = (
                f"MACH-beamformed neutral ultratrace from {opts.input_path.name} "
                f"(z/x coarseness {opts.z_coarseness}, elev {opts.elev_planes}, "
                f"large_fov {opts.large_fov}, spatial_tgc {opts.spatial_tgc})"
            )
            for out_id, acq_id in enumerate(ids):
                iq, txd, txd_elev = _load_acq(h5, acq_id)
                print(f"[beamform] acq {acq_id}: iq {iq.shape} -> mach", flush=True)
                compound, grid = beamform_iq(iq, txd, txd_elev, config)
                if inv_sqrt is not None:
                    compound = (compound * inv_sqrt).astype(np.complex64)
                _write_acq(out, out_id, compound, grid)
                print(f"[beamform] acq {acq_id}: compound {compound.shape} written", flush=True)
                del compound, iq
    print(f"DONE: {opts.output_path}")
    return opts.output_path


def make_options(args) -> BeamformOptions:
    return BeamformOptions(
        input_path=Path(args.input).expanduser().resolve(),
        output_path=Path(args.output).expanduser().resolve(),
        acq_start=args.acq_start,
        num_acqs=args.num_acqs,
        acq_step=args.acq_step,
        all_acqs=args.all_acqs,
        elev_planes=args.elev_planes,
        z_coarseness=args.z_coarseness,
        x_coarseness=args.x_coarseness,
        large_fov=not args.no_large_fov,
        xlarge_fov=args.xlarge_fov,
        n_chunks=args.n_chunks,
        spatial_tgc=args.spatial_tgc,
        tgc_acqs=args.tgc_acqs,
        tgc_sigma_lambda=args.tgc_sigma_lambda,
        tgc_svd_cut=args.tgc_svd_cut,
        resume=args.resume,
        dry_run=args.dry_run,
    )
