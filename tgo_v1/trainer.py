from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .activation_manager import LayerAccumulator
from .covariance import eigen_decomposition
from .spectral import singular_values_from_eigs
from .utils import ensure_dir, is_main_process, save_json, set_seed, setup_logging, timestamp, unwrap_model
from .visualization import plot_curve, plot_heatmap, plot_multi_curve, plot_trajectory
from .trajectories import fit_shared_pca_basis, project_trajectory


LAYER_NAMES = [
    "Layer_00_PatchEmbed",
    "Layer_00_PosEmbed",
    *[f"Layer_{i:02d}_Block{i:02d}" for i in range(1, 13)],
    "Layer_13_CLS_Final",
]


class Trainer:
    def __init__(self, cfg, model, train_loader, val_loader, analysis_loader, trajectory_loader, device):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.analysis_loader = analysis_loader
        self.trajectory_loader = trajectory_loader
        self.device = device
        self.output_dir = ensure_dir(cfg.output_dir)
        self.checkpoint_dir = ensure_dir(self.output_dir / "checkpoints")
        self.results_dir = ensure_dir(self.output_dir / "results")
        self.global_dir = ensure_dir(self.output_dir / "global_analysis")
        self.logger = setup_logging(self.output_dir)
        self.best_acc = -1.0
        self.scaler = GradScaler(enabled=bool(cfg.train.amp))
        self.optim = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.analysis_cache = {}
        self.trajectory_raw_dir = ensure_dir(self.output_dir / "trajectory_raw")
        self.layer_names = LAYER_NAMES

    def _build_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.cfg.train.opt.lower() == "sgd":
            return torch.optim.SGD(params, lr=self.cfg.train.lr, momentum=self.cfg.train.momentum, weight_decay=self.cfg.train.weight_decay)
        return torch.optim.AdamW(params, lr=self.cfg.train.lr, weight_decay=self.cfg.train.weight_decay)

    def _build_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=self.cfg.train.epochs, eta_min=self.cfg.train.min_lr)

    def save_checkpoint(self, epoch: int, train_loss: float, val_acc: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": unwrap_model(self.model).state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "train_loss": train_loss,
            "validation_accuracy": val_acc,
            "random_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
            },
            "config": self.cfg.to_dict(),
        }
        last_path = self.checkpoint_dir / "last.pth"
        torch.save(state, last_path)
        if is_best and self.cfg.checkpoint.save_best:
            torch.save(state, self.checkpoint_dir / "best.pth")
        if self.cfg.checkpoint.every_epoch:
            torch.save(state, self.checkpoint_dir / f"epoch_{epoch:03d}.pth")

    def load_checkpoint(self, path: str | Path):
        ckpt = torch.load(path, map_location="cpu")
        unwrap_model(self.model).load_state_dict(ckpt["model_state_dict"])
        self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.scaler.load_state_dict(ckpt.get("scaler_state_dict", {}))
        return ckpt

    def train(self):
        self.model.to(self.device)
        for epoch in range(1, self.cfg.train.epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_acc = self.validate(epoch)
            self.analyze_epoch(epoch)
            self.capture_trajectories(epoch)
            self.scheduler.step()
            is_best = val_acc > self.best_acc
            self.best_acc = max(self.best_acc, val_acc)
            self.save_checkpoint(epoch, train_loss, val_acc, is_best=is_best)
            self.logger.info(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f} | best={self.best_acc:.4f}")
        self.final_global_analysis()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Train {epoch:03d}", disable=not is_main_process())
        for images, targets, _ in pbar:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optim.zero_grad(set_to_none=True)
            with autocast(enabled=bool(self.cfg.train.amp)):
                logits = self.model(images)
                loss = F.cross_entropy(logits, targets, label_smoothing=self.cfg.train.label_smoothing)
            self.scaler.scale(loss).backward()
            if self.cfg.train.clip_grad and self.cfg.train.clip_grad > 0:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad)
            self.scaler.step(self.optim)
            self.scaler.update()
            total_loss += loss.item() * images.size(0)
            total += images.size(0)
            pbar.set_postfix(loss=loss.item())
        return total_loss / max(total, 1)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for images, targets, _ in tqdm(self.val_loader, desc=f"Val {epoch:03d}", disable=not is_main_process()):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            logits = self.model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
        return correct / max(total, 1)

    def _build_accumulators(self):
        return {name: LayerAccumulator(dim=384, anisotropy_sample_size=self.cfg.analysis.anisotropy_sample_size) for name in self.layer_names}

    @torch.no_grad()
    def analyze_epoch(self, epoch: int):
        self.model.eval()
        hooks_mgr = self.model.hooks_mgr
        accs = self._build_accumulators()
        for images, _, _ in tqdm(self.analysis_loader, desc=f"Analysis {epoch:03d}", disable=not is_main_process()):
            images = images.to(self.device, non_blocking=True)
            _ = self.model(images)
            cache = dict(hooks_mgr.cache)
            # Update metrics layer-by-layer using current batch only
            for name in self.layer_names:
                t = cache.get(name)
                if t is None:
                    continue
                arr = t.detach().float().cpu().numpy()
                if name == "Layer_13_CLS_Final" and arr.ndim == 2:
                    arr = arr[:, None, :]
                if name == "Layer_00_PatchEmbed":
                    arr = arr  # B x 196 x 384
                accs[name].update(arr)
            hooks_mgr.clear()

        epoch_dir = ensure_dir(self.results_dir / f"epoch_{epoch:03d}")
        layer_results = {}
        for name, acc in accs.items():
            layer_res = acc.finalize(pca_components=self.cfg.analysis.pca_components)
            layer_results[name] = layer_res
            layer_dir = ensure_dir(epoch_dir / name.lower())
            np.save(layer_dir / "covariance.npy", layer_res.covariance)
            np.save(layer_dir / "eigenvalues.npy", layer_res.eigenvalues)
            np.save(layer_dir / "eigenvectors.npy", layer_res.eigenvectors)
            np.save(layer_dir / "singular_values.npy", layer_res.singular_values)
            np.save(layer_dir / "pc_vectors.npy", layer_res.pc_vectors)
            np.save(layer_dir / "explained_variance.npy", layer_res.explained_variance)
            np.save(layer_dir / "explained_variance_ratio.npy", layer_res.explained_variance_ratio)
            np.save(layer_dir / "spectral_decay.npy", layer_res.spectral_decay)
            np.save(layer_dir / "anisotropy_sample.npy", layer_res.anisotropy_sample)
            save_json(layer_res.metrics, layer_dir / "metrics.json")
            if epoch in self.cfg.analysis.covariance_snapshot_epochs:
                from .visualization import plot_heatmap, plot_curve
                plot_heatmap(layer_res.covariance, layer_dir / "covariance_heatmap.png", title=f"{name} Covariance")
                plot_curve(layer_res.eigenvalues[:50], layer_dir / "eigenspectrum.png", title=f"{name} Eigenspectrum", ylabel="Eigenvalue")
                plot_curve(layer_res.singular_values[:50], layer_dir / "svd_spectrum.png", title=f"{name} Singular Values", ylabel="Singular Value")
                plot_curve(layer_res.explained_variance_ratio, layer_dir / "explained_variance_curve.png", title=f"{name} Explained Variance Ratio", ylabel="Ratio")
                plot_curve(layer_res.spectral_decay[:50], layer_dir / "spectral_decay.png", title=f"{name} Spectral Decay", ylabel="σ_i / σ_1")
        save_json({"epoch": epoch, "layers": {k: v.metrics for k, v in layer_results.items()}}, epoch_dir / "epoch_metrics.json")

        # store epoch summaries
        summary_path = self.results_dir / "epoch_summaries.jsonl"
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "layers": {k: v.metrics for k, v in layer_results.items()}}) + "\n")

    @torch.no_grad()
    def capture_trajectories(self, epoch: int):
        self.model.eval()
        hooks_mgr = self.model.hooks_mgr
        cls_layers = []
        token_layers = []
        for images, _, _ in tqdm(self.trajectory_loader, desc=f"Trajectory {epoch:03d}", disable=not is_main_process()):
            images = images.to(self.device, non_blocking=True)
            _ = self.model(images)
            cache = dict(hooks_mgr.cache)
            ordered = [cache.get(name) for name in self.layer_names if cache.get(name) is not None]
            # CLS trajectory: stack CLS from all layers for current batch
            cls_batch = []
            token_batch = []
            for name in self.layer_names:
                t = cache.get(name)
                if t is None:
                    continue
                arr = t.detach().float().cpu().numpy()
                if arr.ndim == 3:
                    cls_batch.append(arr[:, 0, :])
                    token_batch.append(arr[:, [10, 50, 100], :])
                elif arr.ndim == 2:
                    cls_batch.append(arr)
                    token_batch.append(arr[:, None, :].repeat(3, axis=1))
            if cls_batch:
                cls_layers.append(np.stack(cls_batch, axis=1))
            if token_batch:
                token_layers.append(np.stack(token_batch, axis=1))
            hooks_mgr.clear()

        traj_dir = ensure_dir(self.trajectory_raw_dir / f"epoch_{epoch:03d}")
        if cls_layers:
            cls_raw = np.concatenate(cls_layers, axis=0)
            np.save(traj_dir / "cls_raw.npy", cls_raw)
        if token_layers:
            tok_raw = np.concatenate(token_layers, axis=0)
            np.save(traj_dir / "token_raw.npy", tok_raw)

    def final_global_analysis(self):
        # build plots from summary JSONL
        summary_file = self.results_dir / "epoch_summaries.jsonl"
        if not summary_file.exists():
            return
        epochs = []
        metrics_by_layer = {}
        with open(summary_file, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                epochs.append(rec["epoch"])
                for lname, m in rec["layers"].items():
                    metrics_by_layer.setdefault(lname, {k: [] for k in m.keys()})
                    for k, v in m.items():
                        metrics_by_layer[lname][k].append(v)
        global_dir = ensure_dir(self.global_dir)
        # layer-wise plots for representative layer-wise metrics
        for metric in ["effective_rank", "stable_rank", "participation_ratio", "spectral_entropy", "spectral_flatness", "spectral_anisotropy"]:
            series = {lname: vals[metric] for lname, vals in metrics_by_layer.items() if metric in vals}
            if series:
                plot_multi_curve(series, global_dir / f"{metric}_layers.png", title=f"{metric} across layers", ylabel=metric)

        # aggregate across layers (mean)
        for metric in ["effective_rank", "stable_rank", "participation_ratio", "spectral_entropy", "anisotropy"]:
            vals = []
            for eidx in range(len(epochs)):
                per_layer = []
                for lname, d in metrics_by_layer.items():
                    if metric in d:
                        per_layer.append(d[metric][eidx])
                vals.append(float(np.mean(per_layer)) if per_layer else 0.0)
            plot_curve(vals, global_dir / f"{metric}_vs_epoch.png", title=f"{metric} vs Epoch", ylabel=metric)

        # trajectory plots from raw arrays
        self._finalize_trajectories()

    def _finalize_trajectories(self):
        raw_dirs = sorted(self.trajectory_raw_dir.glob("epoch_*"))
        if not raw_dirs:
            return

        # Fit shared PCA on the final epoch to keep a fixed coordinate system across all epochs.
        final_dir = raw_dirs[-1]
        cls_raw = np.load(final_dir / "cls_raw.npy") if (final_dir / "cls_raw.npy").exists() else None
        token_raw = np.load(final_dir / "token_raw.npy") if (final_dir / "token_raw.npy").exists() else None

        if cls_raw is not None:
            # cls_raw: [B, L, D]
            pca = fit_shared_pca_basis(cls_raw.reshape(-1, cls_raw.shape[-1]), n_components=2, random_state=0)
            np.save(self.global_dir / "cls_pca_components.npy", pca.components_)
            np.save(self.global_dir / "cls_pca_mean.npy", pca.mean_)

            all_cls_coords = []
            for rd in raw_dirs:
                arr = np.load(rd / "cls_raw.npy")
                coords = project_trajectory(pca, arr)
                all_cls_coords.append(coords)
                np.save(self.global_dir / f"{rd.name}_cls_coords.npy", coords)
                plot_trajectory(coords.mean(axis=0), self.global_dir / f"{rd.name}_cls_trajectory.png", title=f"CLS trajectory {rd.name}")

            np.save(self.global_dir / "cls_coords_all_epochs.npy", np.array(all_cls_coords, dtype=object), allow_pickle=True)

        if token_raw is not None:
            # token_raw: [B, L, 3, D] where the last axis indexes tokens [10, 50, 100]
            pca = fit_shared_pca_basis(token_raw.reshape(-1, token_raw.shape[-1]), n_components=2, random_state=1)
            np.save(self.global_dir / "token_pca_components.npy", pca.components_)
            np.save(self.global_dir / "token_pca_mean.npy", pca.mean_)

            token_names = ["token_10", "token_50", "token_100"]
            all_token_coords = {name: [] for name in token_names}
            for rd in raw_dirs:
                arr = np.load(rd / "token_raw.npy")
                coords = project_trajectory(pca, arr)  # [B, L, 3, 2]
                np.save(self.global_dir / f"{rd.name}_token_coords.npy", coords)
                for t, name in enumerate(token_names):
                    coords_t = coords[:, :, t, :]  # [B, L, 2]
                    all_token_coords[name].append(coords_t)
                    plot_trajectory(coords_t, self.global_dir / f"{rd.name}_{name}_trajectory.png", title=f"{name} trajectory {rd.name}")

            np.save(self.global_dir / "token_coords_all_epochs.npy", np.array(all_token_coords, dtype=object), allow_pickle=True)
