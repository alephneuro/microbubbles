from __future__ import annotations

import argparse
from pathlib import Path

from .beamform_mach import beamform_mach, make_options as make_beamform_options
from .download import SAMPLE_FILENAME, SAMPLE_URL, download_sample
from .tracking import (
    export_tracks_bin,
    make_options as make_tracking_options,
    run_tracking_outputs,
    smooth_tracks_pickle,
)
from .track_viewer_export import write_track_viewer
from .launcher_export import parse_viewer_spec, write_launcher
from .volume_export import export_svd_volume, make_options as make_volume_options


def add_beamform_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Input raw ultratrace.")
    parser.add_argument("--output", required=True, help="Output beamformed ultratrace.")
    parser.add_argument("--acq-start", type=int, default=0)
    parser.add_argument("--num-acqs", type=int, default=1)
    parser.add_argument("--acq-step", type=int, default=1)
    parser.add_argument(
        "--all-acqs",
        action="store_true",
        help="Process every acquisition. Required for full re-beamforming.",
    )
    parser.add_argument("--elev-planes", type=int, default=25)
    parser.add_argument("--z-coarseness", type=float, default=0.5)
    parser.add_argument("--x-coarseness", type=float, default=0.5)
    parser.add_argument("--no-large-fov", action="store_true")
    parser.add_argument("--xlarge-fov", action="store_true")
    parser.add_argument("--n-chunks", type=int, default=4)
    parser.add_argument("--spatial-tgc", action="store_true")
    parser.add_argument("--tgc-acqs", type=int, default=12)
    parser.add_argument("--tgc-sigma-lambda", type=float, default=9.0)
    parser.add_argument("--tgc-svd-cut", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")


def add_tracking_args(parser: argparse.ArgumentParser, require_beamformed: bool = True) -> None:
    parser.add_argument(
        "--beamformed",
        required=require_beamformed,
        help="Beamformed ultratrace.",
    )
    parser.add_argument("--tracks", required=True, help="Output tracking pickle.")
    parser.add_argument("--num-acqs", type=int, default=None)
    parser.add_argument("--acq-start", type=int, default=None)
    parser.add_argument("--acq-step", type=int, default=1)
    parser.add_argument("--output-per-acq", action="store_true")
    parser.add_argument("--sigma-threshold", type=float, default=2.0)
    parser.add_argument("--svd-low-cutoff", "--svd-cutoff", type=float, default=0.1)
    parser.add_argument("--svd-high-cutoff", type=float, default=None)
    parser.add_argument("--svd-n-components", type=int, default=None)
    parser.add_argument(
        "--svd-method",
        choices=["fast", "full", "adaptive", "none"],
        default="fast",
        help="Temporal SVD implementation. 'adaptive' picks the clutter cutoff "
        "per acquisition from the temporal spectral centroid (needs --frame-rate).",
    )
    parser.add_argument(
        "--knee-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="In adaptive mode, drop per-acquisition detections below the z-score knee.",
    )
    parser.add_argument(
        "--tissue-freq",
        dest="tissue_freq_hz",
        type=float,
        default=100.0,
        help="Tissue/blood frequency boundary (Hz) for adaptive SVD.",
    )
    parser.add_argument(
        "--filter",
        dest="filter_method",
        choices=["svd", "none"],
        default="svd",
        help="Use SVD clutter filtering, or treat compound_image as already filtered.",
    )
    parser.add_argument("--temporal-sigma", type=float, default=0.0)
    parser.add_argument("--min-distance", type=int, default=2)
    parser.add_argument("--smoothing-sigma", type=float, default=1.0)
    parser.add_argument("--subpixel", choices=["parabolic", "centroid", "gaussian_fit"], default="centroid")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--patch-radius", type=int, default=3)
    parser.add_argument("--tracking", choices=["kalman", "greedy"], default="kalman")
    parser.add_argument("--frame-rate", dest="frame_rate_hz", type=float, default=None)
    parser.add_argument("--max-dist", type=float, nargs=3, default=None)
    parser.add_argument("--max-dist-mms", type=float, nargs=3, default=None)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--min-track-length", type=int, default=15)
    parser.add_argument("--reversal-penalty", type=float, default=10.0)
    parser.add_argument("--max-cost", type=float, default=10.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--smooth-method", choices=["gaussian", "3dulm"], default="gaussian")
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--smooth-interp-factor", type=float, default=0.1)
    parser.add_argument("--export-dir", default=None)
    parser.add_argument("--export-stem", default=None)
    parser.add_argument("--min-lengths", type=int, nargs="+", default=[5, 20, 50])


def cmd_beamform(args: argparse.Namespace) -> None:
    opts = make_beamform_options(args)
    beamform_mach(opts)


def cmd_track(args: argparse.Namespace) -> None:
    opts = make_tracking_options(args)
    run_tracking_outputs(opts)


def cmd_track_export(args: argparse.Namespace) -> None:
    opts = make_tracking_options(args)
    run_tracking_outputs(opts)


def cmd_download(args: argparse.Namespace) -> None:
    download_sample(args.url, args.output, force=args.force, downloader=args.downloader)


def _stage(label: str) -> None:
    print(f"\n\033[1m==> {label}\033[0m")


def cmd_run(args: argparse.Namespace) -> None:
    """End-to-end demo: download -> beamform -> track -> track viewer."""
    work = Path(args.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)

    # 1. Resolve input: explicit beamformed file, explicit raw input, or sample.
    beamformed = Path(args.beamformed).expanduser().resolve() if args.beamformed else None
    raw_input = Path(args.input).expanduser().resolve() if args.input else None

    if beamformed is None:
        if raw_input is None:
            _stage("Download sample ultratrace")
            raw_input = work / SAMPLE_FILENAME
            if raw_input.exists() and not args.force:
                print(f"Using existing {raw_input} (--force to re-download)")
            else:
                download_sample(args.url, raw_input, force=args.force, downloader=args.downloader)

        # 2. Beamform raw -> beamformed.h5
        beamformed = work / "beamformed.h5"
        if beamformed.exists() and not args.force:
            _stage(f"Beamform (skipped, {beamformed.name} exists)")
        else:
            _stage("Beamform")
            beamform_ns = argparse.Namespace(
                input=str(raw_input),
                output=str(beamformed),
                acq_start=args.acq_start if args.acq_start is not None else 0,
                num_acqs=args.num_acqs if args.num_acqs is not None else 1,
                acq_step=args.acq_step,
                all_acqs=args.all_acqs,
                elev_planes=25,
                z_coarseness=0.5,
                x_coarseness=0.5,
                no_large_fov=False,
                xlarge_fov=False,
                n_chunks=4,
                spatial_tgc=args.spatial_tgc,
                tgc_acqs=12,
                tgc_sigma_lambda=9.0,
                tgc_svd_cut=0.05,
                resume=False,
                dry_run=False,
            )
            beamform_mach(make_beamform_options(beamform_ns))

    # 3. Track + smooth + export bins
    tracks = work / "tracks.pkl"
    _stage("Track")
    track_ns = argparse.Namespace(
        beamformed=str(beamformed),
        tracks=str(tracks),
        num_acqs=args.num_acqs,
        acq_start=args.acq_start,
        acq_step=args.acq_step,
        output_per_acq=False,
        sigma_threshold=2.0,
        svd_low_cutoff=0.1,
        svd_high_cutoff=None,
        svd_n_components=None,
        svd_method=args.svd_method,
        knee_filter=True,
        tissue_freq_hz=100.0,
        temporal_sigma=0.0,
        filter_method="svd",
        min_distance=2,
        smoothing_sigma=1.0,
        subpixel="centroid",
        window_size=5,
        patch_radius=3,
        tracking="kalman",
        frame_rate_hz=args.frame_rate,
        max_dist=None,
        max_dist_mms=None,
        max_gap=3,
        min_track_length=5,
        reversal_penalty=10.0,
        max_cost=10.0,
        smooth_sigma=2.0,
        export_dir=str(work),
        export_stem="tracks",
        min_lengths=list(args.export_min_lengths),
    )
    smoothed = run_tracking_outputs(make_tracking_options(track_ns))

    # 4. Build the animated 3D track viewer (the focus bundle).
    _stage("Build track viewer")
    viewer_dir = work / "viewer"
    write_track_viewer(
        smoothed,
        viewer_dir,
        min_length=args.min_length,
        beamformed=beamformed,
    )

    print(f"\n\033[1mDone.\033[0m Serve the viewer with:")
    print(f"  cd {viewer_dir} && python3 -m http.server 8080")
    print("  open http://localhost:8080")


def cmd_export(args: argparse.Namespace) -> None:
    smoothed = smooth_tracks_pickle(
        Path(args.tracks).expanduser().resolve(),
        Path(args.smoothed).expanduser().resolve() if args.smoothed else None,
        sigma=args.smooth_sigma,
        method=args.smooth_method,
        interp_factor=args.smooth_interp_factor,
        window=args.smooth_window,
    )
    output_dir = Path(args.export_dir).expanduser().resolve()
    stem = args.export_stem or smoothed.stem
    for min_length in args.min_lengths:
        export_tracks_bin(smoothed, output_dir / f"{stem}_min{min_length}.bin", min_length)


def cmd_volume(args: argparse.Namespace) -> None:
    export_svd_volume(make_volume_options(args))


def cmd_track_viewer(args: argparse.Namespace) -> None:
    write_track_viewer(
        Path(args.tracks).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        min_length=args.min_length,
        sigma=args.sigma,
        use_smoothed=not args.raw,
        beamformed=Path(args.beamformed).expanduser().resolve() if args.beamformed else None,
        svd_cutoff=args.svd_cutoff,
    )


def cmd_launcher(args: argparse.Namespace) -> None:
    viewers = [parse_viewer_spec(spec) for spec in args.viewer]
    write_launcher(
        Path(args.output_dir).expanduser().resolve(),
        title=args.title,
        subtitle=args.subtitle,
        footer=args.footer,
        viewers=viewers,
    )


def cmd_doctor(args: argparse.Namespace) -> None:
    import h5py
    import numpy
    import scipy

    print("Standalone runtime imports OK")
    print(f"h5py: {h5py.__version__}")
    print(f"numpy: {numpy.__version__}")
    print(f"scipy: {scipy.__version__}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ultratrace-ulm")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "download",
        help="Download the public sample ultratrace (resumable, ~96 GB).",
    )
    p.add_argument("-o", "--output", default=SAMPLE_FILENAME, help="Destination path.")
    p.add_argument("--url", default=SAMPLE_URL, help="Source URL.")
    p.add_argument("--force", action="store_true", help="Re-download even if present.")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser(
        "run",
        help="End-to-end demo: download -> beamform -> track -> 3D track viewer.",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input", help="Raw neutral ultratrace to beamform.")
    src.add_argument("--beamformed", help="Existing beamformed H5 (skip beamform).")
    p.add_argument("--url", default=SAMPLE_URL, help="Sample URL when downloading.")
    p.add_argument("--work-dir", default="outputs", help="Output directory for all artifacts.")
    p.add_argument("--frame-rate", dest="frame_rate", type=float, default=222.0)
    p.add_argument("--svd-method", choices=["fast", "full", "adaptive", "none"], default="adaptive")
    p.add_argument("--spatial-tgc", action="store_true", help="Spatial TGC during beamforming.")
    p.add_argument("--min-length", type=int, default=35, help="Min track length for the viewer.")
    p.add_argument("--export-min-lengths", type=int, nargs="+", default=[5, 20, 50])
    p.add_argument("--acq-start", type=int, default=None)
    p.add_argument("--num-acqs", type=int, default=None)
    p.add_argument("--acq-step", type=int, default=1)
    p.add_argument("--all-acqs", action="store_true", help="Beamform every acquisition.")
    p.add_argument("--force", action="store_true", help="Re-run stages whose outputs exist.")
    parser.add_argument("--downloader", choices=("aria2", "urllib"), default="aria2", help=( "Download backend"))

    p.set_defaults(func=cmd_run)

    p = sub.add_parser("beamform", help="MACH-only beamform selected acquisitions.")
    add_beamform_args(p)
    p.set_defaults(func=cmd_beamform)

    p = sub.add_parser("track", help="Run streaming ULM tracking on a beamformed file.")
    add_tracking_args(p)
    p.set_defaults(func=cmd_track)

    p = sub.add_parser(
        "track-export",
        help="Track, smooth, and export bins from an existing beamformed file.",
        conflict_handler="resolve",
    )
    add_tracking_args(p, require_beamformed=True)
    p.set_defaults(func=cmd_track_export)

    p = sub.add_parser("export", help="Smooth and export an existing track pickle.")
    p.add_argument("--tracks", required=True)
    p.add_argument("--smoothed", default=None)
    p.add_argument("--smooth-sigma", type=float, default=2.0)
    p.add_argument("--export-dir", required=True)
    p.add_argument("--export-stem", default=None)
    p.add_argument("--min-lengths", type=int, nargs="+", default=[5, 20, 50])
    p.set_defaults(func=cmd_export)

    p = sub.add_parser(
        "volume",
        help="Export a rotatable 3D SVD voxel viewer with track overlay.",
    )
    p.add_argument("--beamformed", required=True, help="Beamformed ultratrace.")
    p.add_argument("--tracks", default=None, help="Optional ULM track pickle.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--acq-start", type=int, default=0)
    p.add_argument("--num-acqs", type=int, default=1)
    p.add_argument("--acq-step", type=int, default=1)
    p.add_argument("--svd-low-cutoff", type=float, default=0.1)
    p.add_argument("--svd-high-cutoff", type=float, default=None)
    p.add_argument("--svd-method", choices=["fast", "full"], default="fast")
    p.add_argument("--temporal-sigma", type=float, default=0.0)
    p.add_argument("--dynamic-range-db", type=float, default=15.0)
    p.add_argument("--percentile", type=float, default=99.7)
    p.add_argument("--voxel-percentile", type=float, default=99.9)
    p.add_argument("--max-points-per-frame", type=int, default=8000)
    p.add_argument("--background-percentile", type=float, default=99.8)
    p.add_argument("--background-dynamic-range-db", type=float, default=45.0)
    p.add_argument("--background-max-points-per-frame", type=int, default=18000)
    p.add_argument("--background-intro-seconds", type=float, default=1.5)
    p.add_argument("--subtraction-fade-seconds", type=float, default=1.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--track-min-length", type=int, default=5)
    p.add_argument("--tail-frames", type=int, default=18)
    p.add_argument("--max-frames", type=int, default=None)
    p.set_defaults(func=cmd_volume)

    p = sub.add_parser(
        "track-viewer",
        help="Export the animated 3D track viewer (Three.js point-flow) bundle.",
    )
    p.add_argument("--tracks", required=True, help="ULM track pickle.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--min-length", type=int, default=35)
    p.add_argument("--sigma", type=float, default=10.0, help="Gaussian sigma for velocity smoothing.")
    p.add_argument("--raw", action="store_true", help="Use raw tracks instead of smoothed.")
    p.add_argument(
        "--beamformed",
        default=None,
        help="Optional beamformed H5 for per-point B-mode intensity lookup.",
    )
    p.add_argument(
        "--svd-cutoff",
        type=float,
        default=0.0,
        help="SVD low cutoff for intensity lookup (0 = raw magnitude).",
    )
    p.set_defaults(func=cmd_track_viewer)

    p = sub.add_parser(
        "launcher",
        help="Write a landing page (index.html + viewers.json) linking viewer bundles.",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--title", default="Ultratrace ULM Viewers")
    p.add_argument("--subtitle", default="")
    p.add_argument("--footer", default="")
    p.add_argument(
        "--viewer",
        action="append",
        default=[],
        metavar="HREF|TITLE|DESC|EMOJI",
        help="Repeatable viewer card: relative href plus optional title, description, emoji.",
    )
    p.set_defaults(func=cmd_launcher)

    p = sub.add_parser("doctor", help="Check runtime imports.")
    p.set_defaults(func=cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
