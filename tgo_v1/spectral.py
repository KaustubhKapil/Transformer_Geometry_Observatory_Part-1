from __future__ import annotations

import numpy as np


def singular_values_from_eigs(eigs: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(eigs, 0.0))


def effective_rank(eigs: np.ndarray) -> float:
    eigs = np.maximum(np.asarray(eigs, dtype=np.float64), 0.0)
    s = eigs.sum()
    if s <= 0:
        return 0.0
    p = eigs / s
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def stable_rank(eigs: np.ndarray) -> float:
    eigs = np.maximum(np.asarray(eigs, dtype=np.float64), 0.0)
    if eigs.size == 0 or eigs.max() <= 0:
        return 0.0
    return float(eigs.sum() / eigs.max())


def participation_ratio(eigs: np.ndarray) -> float:
    eigs = np.maximum(np.asarray(eigs, dtype=np.float64), 0.0)
    num = eigs.sum() ** 2
    den = np.sum(eigs ** 2)
    return float(num / den) if den > 0 else 0.0


def spectral_entropy(eigs: np.ndarray) -> float:
    eigs = np.maximum(np.asarray(eigs, dtype=np.float64), 0.0)
    s = eigs.sum()
    if s <= 0:
        return 0.0
    p = eigs / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def spectral_flatness(eigs: np.ndarray) -> float:
    eigs = np.maximum(np.asarray(eigs, dtype=np.float64), 1e-12)
    return float(np.exp(np.mean(np.log(eigs))) / np.mean(eigs))


def spectral_decay(eigs: np.ndarray) -> np.ndarray:
    eigs = np.maximum(np.asarray(eigs, dtype=np.float64), 0.0)
    if eigs.size == 0 or eigs[0] <= 0:
        return np.zeros_like(eigs)
    return eigs / eigs[0]
