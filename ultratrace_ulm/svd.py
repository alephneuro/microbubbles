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


def filter_svd_3d(
    data: np.ndarray,
    low_cutoff: float = 0.1,
    high_cutoff: float | None = None,
    method: str = "fast",
    n_components: int | None = None,
) -> np.ndarray:
    """Apply temporal SVD clutter filtering to (frames,elev,z,x) data."""
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
    low = int(n_components) if n_components is not None else _component_count(low_cutoff, n_frames)
    high = 1.0 if high_cutoff is None else float(high_cutoff)
    high_remove = max(0, min(n_frames, int(round((1.0 - high) * n_frames))))
    if low + high_remove >= n_frames:
        raise ValueError(
            f"SVD cutoff removes all components: low={low_cutoff}, high={high_cutoff}"
        )

    normalized_method = "fast" if method in {"adaptive", "gpu", "gpu_full", "randomized"} else method
    if normalized_method == "none":
        filtered = matrix
    elif normalized_method == "fast":
        cov = matrix @ matrix.conj().T
        evals, u = np.linalg.eigh(cov)
        u = u[:, np.argsort(evals)[::-1]]
        stop = n_frames - high_remove if high_remove > 0 else n_frames
        uc = u[:, low:stop]
        filtered = uc @ (uc.conj().T @ matrix)
    elif normalized_method == "full":
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
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
) -> np.ndarray:
    filtered = filter_svd_3d(
        compound,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        method=method,
        n_components=n_components,
    )
    magnitude = np.abs(filtered).astype(np.float32, copy=False)
    if temporal_sigma > 0:
        magnitude = gaussian_filter1d(magnitude, sigma=temporal_sigma, axis=0)
    return magnitude
