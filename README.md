# TGO-Part-1

This repository implements a streaming Transformer Geometry Observatory for ViT-Small/16 on ImageNet-100. It is one of the 8 parts of my transformer observatory. This part explores only the Linear geometry of the **Vision Tranformer** baseline. Here I have analyzed:
- SVD
- Eigenspectra and their decay
- Effective Ranks
- Covariance Accumulation

The above help us understand and vizualize the Vision Transformer as a linear transformation engine, answering questions about emerging dominant directions, tranforming representations and a broader *Dimensional Analysis*

## Files

- `tgo_v1/main.py` — entry point
- `tgo_v1/trainer.py` — training + observatory pipeline
- `tgo_v1/dataset.py` — dataset construction and fixed subset selection
- `tgo_v1/hooks.py` — ViT forward-hook activation capture
- `tgo_v1/covariance.py` — streaming covariance accumulation
- `tgo_v1/spectral.py` — eigenspectrum / rank metrics
- `tgo_v1/anisotropy.py` — anisotropy metrics
- `tgo_v1/trajectories.py` — CLS/token trajectory PCA projection
- `tgo_v1/visualization.py` — matplotlib plotting helpers

## Run

```bash
python -m tgo_v1.main --config configs/vit_small_imagenet100_p5000.yaml
```

Optional resume:

```bash
python -m tgo_v1.main --config configs/vit_small_imagenet100_p5000.yaml --resume results_tgo_v1/checkpoints/last.pth
```

## Expected data layout

```text
/path/to/imagenet100/train/<class_name>/*.JPEG
/path/to/imagenet100/val/<class_name>/*.JPEG
```

## Notes

- This is the first of 8 different analyses protocols. 
- Analysis metrics are computed on a fixed validation subset.
- Covariance accumulation is streaming and does not require storing all activations.
- Trajectories are stored per epoch and projected using a shared PCA basis.


## Results

### Progressive Expansion of Representation Geometry

We analyzed the evolution of Vision Transformer (ViT-Small/16) representations throughout training using a suite of spectral and geometric observables, including Effective Rank, Participation Ratio, Stable Rank, Spectral Entropy, Spectral Flatness, and Spectral Anisotropy.

Across all transformer layers, we observed a consistent increase in Effective Rank, Participation Ratio, Stable Rank, Spectral Entropy, and Spectral Flatness, accompanied by a decrease in Spectral Anisotropy.

Collectively, these measurements indicate that the representation manifold becomes progressively less concentrated and increasingly distributed during training. Rather than collapsing into a small number of dominant directions, variance is redistributed across a larger fraction of the available feature space.

This phenomenon was consistently observed across epochs and layers, suggesting that training induces a gradual expansion of the occupied representation subspace.

---

### Layer-wise Evolution of Dimensional Utilization

Layer-wise analysis revealed a progressive increase in effective dimensionality from Patch Embedding layers toward deeper transformer blocks.

Early representations occupied a relatively low-dimensional anisotropic manifold. As depth increased, representations exhibited increased dimensional utilization, reduced spectral concentration, and higher spectral entropy.

The strongest effect was observed in the final CLS representation.

These observations suggest that information is progressively redistributed across a larger set of representational directions as depth increases.

---

### Emergence of Distributed CLS Representations

A particularly notable result was observed in the final CLS token.

Compared to intermediate transformer layers, the CLS representation exhibited:

* Highest Effective Rank
* Highest Participation Ratio
* Highest Stable Rank
* Highest Spectral Entropy
* Lowest Spectral Anisotropy

Covariance analysis and eigenspectrum visualizations further revealed that variance becomes broadly distributed across many principal directions rather than concentrating into a small number of dominant components.

This behavior suggests that the CLS token functions as a progressively richer global information integrator throughout training rather than a simple low-dimensional bottleneck.

---

### Spectral Redistribution During Training

Eigenspectrum and singular value analyses performed at Epochs 10, 50, and 100 demonstrated a progressive flattening of the spectrum.

Early training was characterized by strong variance concentration in a small number of dominant directions. As training progressed, variance became distributed across a larger number of eigenmodes.

This observation is consistent with the simultaneous increase in Effective Rank, Participation Ratio, Stable Rank, Spectral Entropy, and Spectral Flatness, together with decreasing Spectral Anisotropy.

The resulting representation geometry becomes increasingly distributed and less dominated by individual principal directions.

---

### Interpretation

The results provide strong evidence that ViT representations undergo progressive geometric expansion throughout training.

The observed increase in dimensional utilization may arise from one or more of the following mechanisms:

1. Progressive feature decorrelation.
2. Token diversification.
3. Semantic factor discovery and expansion.

The current study directly supports the first mechanism through consistent spectral evidence. The latter two remain open hypotheses that require additional observables for validation.

Future installments of the Transformer Geometry Observatory (TGO) framework will investigate these hypotheses through token-level, similarity-based, and attention-based analyses.

---

### Main Finding

**The aggregated token representation manifold becomes progressively higher-dimensional, less anisotropic, and more uniformly distributed throughout training.**

This behavior is consistently observed across multiple independent spectral and geometric metrics and is most pronounced in the final CLS representation.

The results suggest that ViT training is characterized by progressive geometric expansion rather than representational collapse.
