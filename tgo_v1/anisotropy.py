from __future__ import annotations

import numpy as np


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def mean_pairwise_cosine_similarity(x: np.ndarray, n_pairs: int = 2048, rng: np.random.Generator | None = None) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] < 2:
        return 0.0
    rng = rng or np.random.default_rng(0)
    x = _normalize_rows(x)
    n = x.shape[0]
    ii = rng.integers(0, n, size=n_pairs)
    jj = rng.integers(0, n, size=n_pairs)
    sims = np.sum(x[ii] * x[jj], axis=1)
    return float(np.mean(sims))


def spectral_anisotropy_score(eigs: np.ndarray) -> float:
    eigs = np.asarray(eigs, dtype=np.float64)
    if eigs.size == 0 or eigs.sum() <= 0:
        return 0.0
    p = eigs / eigs.sum()
    return float(p.max())
