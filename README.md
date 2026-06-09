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
