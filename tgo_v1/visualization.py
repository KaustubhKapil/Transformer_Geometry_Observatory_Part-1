from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dir


def _save_fig(fig, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(mat: np.ndarray, path: str | Path, title: str = "", cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, path)


def plot_curve(values: Iterable[float], path: str | Path, title: str = "", xlabel: str = "Epoch", ylabel: str = "Value"):
    vals = list(values)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(vals) + 1), vals)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def plot_multi_curve(series: Dict[str, Iterable[float]], path: str | Path, title: str = "", xlabel: str = "Epoch", ylabel: str = "Value"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, vals in series.items():
        vals = list(vals)
        ax.plot(range(1, len(vals) + 1), vals, label=name)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_fig(fig, path)


def plot_lines(x: np.ndarray, y: np.ndarray, path: str | Path, title: str = "", xlabel: str = "", ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)


def plot_trajectory(coords: np.ndarray, path: str | Path, title: str = ""):
    fig, ax = plt.subplots(figsize=(6, 6))
    if coords.ndim == 3:
        for i in range(coords.shape[0]):
            ax.plot(coords[i, :, 0], coords[i, :, 1], alpha=0.25)
        mean = coords.mean(axis=0)
        ax.plot(mean[:, 0], mean[:, 1], linewidth=2.5)
    elif coords.ndim == 2:
        ax.plot(coords[:, 0], coords[:, 1], linewidth=2.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    _save_fig(fig, path)
