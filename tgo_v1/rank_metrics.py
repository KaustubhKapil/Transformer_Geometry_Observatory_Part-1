from __future__ import annotations

import numpy as np

from .spectral import effective_rank, stable_rank, participation_ratio, spectral_entropy, spectral_flatness


def compute_rank_metrics(eigs: np.ndarray) -> dict:
    return {
        "effective_rank": effective_rank(eigs),
        "stable_rank": stable_rank(eigs),
        "participation_ratio": participation_ratio(eigs),
        "spectral_entropy": spectral_entropy(eigs),
        "spectral_flatness": spectral_flatness(eigs),
    }
