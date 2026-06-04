from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .covariance import StreamingCovariance
from .anisotropy import mean_pairwise_cosine_similarity, spectral_anisotropy_score


@dataclass
class LayerAnalysisResult:
    covariance: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    singular_values: np.ndarray
    metrics: Dict[str, float]
    pc_vectors: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    spectral_decay: np.ndarray
    anisotropy_sample: np.ndarray


class LayerAccumulator:
    def __init__(self, dim: int, anisotropy_sample_size: int = 5000):
        self.dim = dim
        self.cov = StreamingCovariance(dim)
        self.anisotropy_sample_size = anisotropy_sample_size
        self.sample = np.empty((0, dim), dtype=np.float32)
        self.total_seen = 0
        self.rng = np.random.default_rng(0)

    def update(self, x: np.ndarray) -> None:
        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        x = np.asarray(x, dtype=np.float32)
        self.cov.update(x)
        self._reservoir_update(x)

    def _reservoir_update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        if self.anisotropy_sample_size <= 0:
            return
        remaining = self.anisotropy_sample_size - self.sample.shape[0]
        if remaining > 0:
            take = min(remaining, x.shape[0])
            if take > 0:
                idx = self.rng.choice(x.shape[0], size=take, replace=False)
                self.sample = np.concatenate([self.sample, x[idx]], axis=0)
        else:
            # Random replacement from a small subset of x for efficiency
            take = min(max(1, x.shape[0] // 8), x.shape[0])
            idx = self.rng.choice(x.shape[0], size=take, replace=False)
            for row in x[idx]:
                j = self.rng.integers(0, self.total_seen + 1)
                if j < self.anisotropy_sample_size:
                    self.sample[j] = row
        self.total_seen += x.shape[0]

    def finalize(self, pca_components: int = 20) -> LayerAnalysisResult:
        cov = self.cov.covariance().astype(np.float64)
        vals, vecs = np.linalg.eigh(cov)
        idx = np.argsort(vals)[::-1]
        vals = np.maximum(vals[idx], 0.0)
        vecs = vecs[:, idx]
        svals = np.sqrt(vals)
        n_comp = min(pca_components, vecs.shape[1])
        pc_vecs = vecs[:, :n_comp].T.astype(np.float32)
        expl_var = vals[:n_comp].astype(np.float32)
        denom = vals.sum() if vals.sum() > 0 else 1.0
        expl_ratio = (vals[:n_comp] / denom).astype(np.float32)
        metrics = {
            "effective_rank": _effective_rank(vals),
            "stable_rank": _stable_rank(vals),
            "participation_ratio": _participation_ratio(vals),
            "spectral_entropy": _spectral_entropy(vals),
            "spectral_flatness": _spectral_flatness(vals),
            "spectral_anisotropy": _spectral_anisotropy(vals),
        }
        decay = (svals / max(float(svals[0]), 1e-12)).astype(np.float32) if svals.size else np.zeros_like(svals, dtype=np.float32)
        return LayerAnalysisResult(
            covariance=cov.astype(np.float32),
            eigenvalues=vals.astype(np.float32),
            eigenvectors=vecs.astype(np.float32),
            singular_values=svals.astype(np.float32),
            metrics=metrics,
            pc_vectors=pc_vecs,
            explained_variance=expl_var,
            explained_variance_ratio=expl_ratio,
            spectral_decay=decay,
            anisotropy_sample=self.sample.astype(np.float32),
        )


def _effective_rank(eigs: np.ndarray) -> float:
    s = eigs.sum()
    if s <= 0:
        return 0.0
    p = eigs / s
    p = p[p > 0]
    return float(np.exp(-np.sum(p * np.log(p))))


def _stable_rank(eigs: np.ndarray) -> float:
    return float(eigs.sum() / eigs.max()) if eigs.size and eigs.max() > 0 else 0.0


def _participation_ratio(eigs: np.ndarray) -> float:
    num = eigs.sum() ** 2
    den = np.sum(eigs ** 2)
    return float(num / den) if den > 0 else 0.0


def _spectral_entropy(eigs: np.ndarray) -> float:
    s = eigs.sum()
    if s <= 0:
        return 0.0
    p = eigs / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _spectral_flatness(eigs: np.ndarray) -> float:
    eigs = np.maximum(eigs, 1e-12)
    return float(np.exp(np.mean(np.log(eigs))) / np.mean(eigs))


def _spectral_anisotropy(eigs: np.ndarray) -> float:
    s = eigs.sum()
    return float(eigs.max() / s) if s > 0 else 0.0
