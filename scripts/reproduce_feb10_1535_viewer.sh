#!/usr/bin/env bash
#
# Reproduce the corrected 15:35 (2026-02-10) ULM viewer bundle end to end:
# tracking -> volume / track-flow / movie viewers -> a single launcher page.
#
# IMPORTANT recipe note
# ---------------------
# Use ADAPTIVE SVD with TEMPORAL-SIGMA 0. The earlier "golden" recipe in the
# README (temporal-sigma 7 + the "fast" SVD variant) tracked stationary
# tissue/clutter: it produced ~7x too many tracks per acquisition with ~4x too
# little net displacement (long but near-stationary). Adaptive SVD (per-acq
# spectral-centroid clutter cutoff, needs --frame-rate) with no temporal blur
# reproduces the production reference (~260 tracks/acq, genuinely flowing).
#
# Usage:
#   scripts/reproduce_feb10_1535_viewer.sh /path/to/ultratrace_1535_25el_n4_tgc.h5 /path/to/outdir
#
# Env overrides:
#   PYTHON      python interpreter (default: python3)
#   NUM_ACQS    acquisitions to process (default: 216 = all; lower for a quick preview)
#   FRAME_RATE  acquisition frame rate in Hz (default: 222.4306816130359)
#
set -euo pipefail

PYTHON="${PYTHON:-python3}"
BEAMFORMED="${1:?usage: reproduce_feb10_1535_viewer.sh /path/to/beamformed.h5 /path/to/outdir}"
OUTDIR="${2:?usage: reproduce_feb10_1535_viewer.sh /path/to/beamformed.h5 /path/to/outdir}"
NUM_ACQS="${NUM_ACQS:-216}"          # acquisitions to TRACK (the ULM image accumulates over all of them)
VIEW_ACQS="${VIEW_ACQS:-12}"         # acquisitions of b-mode context to render in the volume/movie viewers
FRAME_RATE="${FRAME_RATE:-222.4306816130359}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTDIR}"

TRACKS="${OUTDIR}/tracks.pkl"
SMOOTHED="${OUTDIR}/tracks_smoothed.pkl"

echo "[1/5] tracking (adaptive SVD, temporal-sigma 0) on ${NUM_ACQS} acquisitions"
"${PYTHON}" -m ultratrace_ulm.cli run \
  --beamformed "${BEAMFORMED}" \
  --tracks "${TRACKS}" \
  --num-acqs "${NUM_ACQS}" \
  --output-per-acq \
  --svd-method adaptive --frame-rate "${FRAME_RATE}" --knee-filter --tissue-freq 100.0 \
  --temporal-sigma 0 --sigma-threshold 2.0 --svd-low-cutoff 0.1 --min-distance 2 \
  --smoothing-sigma 1.0 --tracking kalman --max-gap 3 --min-track-length 5 --max-cost 10 \
  --export-dir "${OUTDIR}/tracks" --export-stem tracks

echo "[2/5] 3D SVD volume viewer (secondary, served at /volume/)"
"${PYTHON}" -m ultratrace_ulm.cli volume \
  --beamformed "${BEAMFORMED}" --tracks "${SMOOTHED}" \
  --output-dir "${OUTDIR}/viewer/volume" \
  --num-acqs "${VIEW_ACQS}" --voxel-percentile 99.9 --max-points-per-frame 8000 \
  --dynamic-range-db 15 --temporal-sigma 0

echo "[3/5] SVD b-mode movie viewer (secondary, served at /movie/)"
"${PYTHON}" -m ultratrace_ulm.cli movie \
  --beamformed "${BEAMFORMED}" --tracks "${SMOOTHED}" \
  --output-dir "${OUTDIR}/viewer/movie" \
  --num-acqs "${VIEW_ACQS}" --projection xz-slab-mip --elev-slabs 6 \
  --dynamic-range-db 15 --temporal-sigma 0

echo "[4/5] 3D track-flow viewer == DEFAULT viewer at the bundle root"
# The track-flow viewer is written to the bundle ROOT so opening the served
# directory shows the tracks immediately (no menu in the way).
# --svd-cutoff 0.1 colors points by SVD-filtered B-mode intensity, which weights
# bubble signal over tissue brightness (cleaner than raw --svd-cutoff 0).
"${PYTHON}" -m ultratrace_ulm.cli track-viewer \
  --tracks "${SMOOTHED}" --output-dir "${OUTDIR}/viewer" \
  --min-length 5 --sigma 10 --beamformed "${BEAMFORMED}" --svd-cutoff 0.1

echo "[5/5] optional menu of all views at /menu/"
"${PYTHON}" -m ultratrace_ulm.cli launcher \
  --output-dir "${OUTDIR}/viewer/menu" \
  --title "Ultratrace ULM - 15:35 (2026-02-10)" \
  --subtitle "Source: $(basename "${BEAMFORMED}") - ${NUM_ACQS} acqs - adaptive SVD, temporal-sigma 0" \
  --viewer "../|3D Track-Flow Viewer (default)|Animated point-flow of the microbubble tracks.|TRK" \
  --viewer "../volume/|3D SVD Volume Viewer|Rotatable super-resolution volume with track overlay.|VOL" \
  --viewer "../movie/|SVD B-mode Movie Viewer|Six-row elevation-slab SVD b-mode movie with track overlay.|MOV"

echo
echo "Done. The DEFAULT view is the track-flow viewer."
echo "Serve the bundle with:"
echo "  cd ${OUTDIR}/viewer && ${PYTHON} -m http.server 8137 --bind 0.0.0.0"
echo "Then: /  = tracks, /volume/ , /movie/ , /menu/ = all views."
