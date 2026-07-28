from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _component_count(cutoff: float | int | None, n_frames: int) -> int:
    if cutoff is None:
        return 0
    value = float(cutoff)
    if 0.0 <= value <= 1.0:
        return int(round(value * n_frames))
    return int(round(value))


def spectral_centroid_cutoff(
    matrix: np.ndarray,
    frame_rate_hz: float,
    tissue_freq_hz: float = 100.0,
) -> int:
    """Data-driven tissue/blood SVD boundary via temporal spectral centroid.

    For each temporal singular vector, compute the power-weighted mean
    frequency; the cutoff is the first vector whose centroid exceeds
    ``tissue_freq_hz`` (Ghosh et al., PNAS 2025). Falls back to 10% of frames.
    ``matrix`` is the (frames, voxels) temporal matrix.

    The eigendecomposition runs in double precision (complex128): the
    complex64 Gram matrix has near-degenerate eigenvalues whose eigenvectors
    are not reproducible across runs, which would make the selected cutoff --
    and therefore every downstream detection -- non-deterministic. complex128
    resolves the eigenvalue gaps and yields bit-stable results.
    """
    n_frames = int(matrix.shape[0])
    x = np.asarray(matrix, dtype=np.complex128)
    x = x - x.mean(axis=0, keepdims=True)
    cov = x @ x.conj().T
    evals, u = np.linalg.eigh(cov)
    u = u[:, np.argsort(evals)[::-1]]
    freqs = np.fft.rfftfreq(n_frames, d=1.0 / frame_rate_hz)
    centroid = np.zeros(n_frames)
    for i in range(n_frames):
        spectrum = np.abs(np.fft.rfft(u[:, i].real)) ** 2
        spectrum[0] = 0.0  # exclude DC
        total = spectrum.sum()
        if total > 0:
            centroid[i] = float(np.sum(freqs * spectrum) / total)
    above = np.where(centroid > tissue_freq_hz)[0]
    return int(above[0]) if len(above) else max(1, round(n_frames * 0.1))


def filter_svd_3d(
    data: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    n_components: int | None = None,
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
) -> np.ndarray:
    """Apply temporal SVD clutter filtering to (frames,elev,z,x) data.

    method="adaptive" picks the low cutoff per-acquisition from the temporal
    spectral centroid (requires frame_rate_hz); "fast"/"full" use a fixed
    low_cutoff (or n_components). "fast" is the covariance projection; "full"
    is the numerically stable SVD. Both decompositions run in double precision
    (complex128) so the result is deterministic across runs.
    """
    if data.ndim == 3:
        data = data[:, None, :, :]
        squeeze = True
    elif data.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Expected 3D or 4D compound data, got shape {data.shape}")

    n_frames = int(data.shape[0])
    spatial_shape = data.shape[1:]
    matrix = np.asarray(data, dtype=np.complex64).reshape(n_frames, -1)

    if method == "adaptive":
        if frame_rate_hz is None:
            raise ValueError("method='adaptive' requires frame_rate_hz")
        low = spectral_centroid_cutoff(matrix, frame_rate_hz, tissue_freq_hz)
        normalized_method = "fast"
    else:
        low = int(n_components) if n_components is not None else _component_count(low_cutoff, n_frames)
        normalized_method = "fast" if method in {"gpu", "gpu_full", "randomized"} else method

    high = 1.0 if high_cutoff is None else float(high_cutoff)
    high_remove = max(0, min(n_frames, int(round((1.0 - high) * n_frames))))
    if low + high_remove >= n_frames:
        raise ValueError(
            f"SVD cutoff removes all components: low={low}, high={high_cutoff}"
        )

    if normalized_method == "none":
        filtered = matrix
    elif normalized_method == "fast":
        # Double precision: the complex64 Gram matrix has near-degenerate
        # eigenvalues whose eigenvectors vary run-to-run, which shifts the
        # filtered magnitude and makes detections non-deterministic.
        m = np.asarray(matrix, dtype=np.complex128)
        cov = m @ m.conj().T
        evals, u = np.linalg.eigh(cov)
        u = u[:, np.argsort(evals)[::-1]]
        stop = n_frames - high_remove if high_remove > 0 else n_frames
        uc = u[:, low:stop]
        filtered = uc @ (uc.conj().T @ m)
    elif normalized_method == "full":
        m = np.asarray(matrix, dtype=np.complex128)
        u, s, vh = np.linalg.svd(m, full_matrices=False)
        s[:low] = 0
        if high_remove > 0:
            s[-high_remove:] = 0
        filtered = (u * s[None, :]) @ vh
    else:
        raise ValueError(f"Unsupported SVD method: {method}")

    out = filtered.reshape((n_frames, *spatial_shape))
    if squeeze:
        out = out[:, 0]
    return out.astype(np.complex64, copy=False)


def filtered_magnitude(
    compound: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    temporal_sigma: float = 0.0,
    n_components: int | None = None,
    frame_rate_hz: float | None = None,
    tissue_freq_hz: float = 100.0,
) -> np.ndarray:
    filtered = filter_svd_3d(
        compound,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        method=method,
        n_components=n_components,
        frame_rate_hz=frame_rate_hz,
        tissue_freq_hz=tissue_freq_hz,
    )
    magnitude = np.abs(filtered).astype(np.float32, copy=False)
    if temporal_sigma > 0:
        magnitude = gaussian_filter1d(magnitude, sigma=temporal_sigma, axis=0)
    return magnitude
