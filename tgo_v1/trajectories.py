from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.decomposition import PCA


def fit_shared_pca_basis(samples: np.ndarray, n_components: int = 2, random_state: int = 0):
    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="randomized")
    pca.fit(samples)
    return pca


def project_trajectory(pca: PCA, x: np.ndarray) -> np.ndarray:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    z = pca.transform(x2)
    return z.reshape(*shape[:-1], z.shape[-1])


def cls_stack_from_layer_dict(layer_dict: Dict[str, np.ndarray]) -> np.ndarray:
    layers = list(layer_dict.keys())
    cls = []
    for k in layers:
        arr = layer_dict[k]
        if arr.ndim == 3:
            cls.append(arr[:, 0, :])
        elif arr.ndim == 2:
            cls.append(arr)
        else:
            raise ValueError(f"Unexpected CLS array shape for {k}: {arr.shape}")
    return np.stack(cls, axis=1)


def token_stack_from_layer_dict(layer_dict: Dict[str, np.ndarray], token_indices: List[int]) -> np.ndarray:
    layers = list(layer_dict.keys())
    toks = []
    for k in layers:
        arr = layer_dict[k]
        if arr.ndim != 3:
            raise ValueError(f"Unexpected token array shape for {k}: {arr.shape}")
        idx = np.clip(np.asarray(token_indices, dtype=np.int64), 0, arr.shape[1] - 1)
        toks.append(arr[:, idx, :])
    return np.stack(toks, axis=1)
