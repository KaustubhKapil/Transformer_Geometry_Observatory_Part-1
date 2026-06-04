from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class LayerStats:
    n: int
    sum_x: np.ndarray
    sum_xx: np.ndarray


class StreamingCovariance:
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0
        self.sum_x = np.zeros(dim, dtype=np.float64)
        self.sum_xx = np.zeros((dim, dim), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        if x.ndim != 2:
            x = x.reshape(-1, x.shape[-1])
        x = x.astype(np.float64, copy=False)
        self.n += x.shape[0]
        self.sum_x += x.sum(axis=0)
        self.sum_xx += x.T @ x

    def mean(self) -> np.ndarray:
        return self.sum_x / max(self.n, 1)

    def covariance(self) -> np.ndarray:
        if self.n <= 1:
            return np.zeros((self.dim, self.dim), dtype=np.float64)
        mean = self.mean()
        cov = (self.sum_xx - self.n * np.outer(mean, mean)) / (self.n - 1)
        cov = (cov + cov.T) * 0.5
        return cov

    def to_state(self) -> dict:
        return {"n": self.n, "sum_x": self.sum_x, "sum_xx": self.sum_xx}

    @classmethod
    def from_state(cls, state: dict, dim: int) -> "StreamingCovariance":
        obj = cls(dim)
        obj.n = int(state["n"])
        obj.sum_x = state["sum_x"]
        obj.sum_xx = state["sum_xx"]
        return obj


def eigen_decomposition(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = np.linalg.eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = np.maximum(vals[idx], 0.0)
    vecs = vecs[:, idx]
    return vals, vecs


def covariance_from_activations(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x.reshape(-1, x.shape[-1])
    x = x - x.mean(axis=0, keepdims=True)
    return (x.T @ x) / max(x.shape[0] - 1, 1)
