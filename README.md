# CompGS-GSPLaT: Compact3D with GSPLaT Rasterizer

This repository is a fork of [Compact3D (CompGS)](https://arxiv.org/abs/2311.18159), accepted to ECCV 2024, where the original `diff-gaussian-rasterization` backend has been **fully replaced by [Nerfstudio's GSPLaT rasterizer](https://github.com/nerfstudio-project/gsplat)**.

We preserve the full training and evaluation pipeline of Compact3D while leveraging GSPLaT’s **modern, CUDA-efficient rasterization backend**, which provides:

-  **~20% training speedup**
- **Lower memory usage**
- **Slight improvements in rendering quality**
- Compatibility with OpenGL-style camera & gradient flow

## Overview

**Compact3D** is a method to compress large 3D Gaussian Splatting models via vector quantization of Gaussian parameters (color & covariance).  
This fork integrates the GSPLaT rasterizer, yielding faster and more memory-efficient training, especially on large scenes.

## Installation (GSPLaT Backend)

We use the GSPLaT rasterizer from [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat).

Please follow the **official installation instructions from GSPLaT** to set up the rasterization backend and its dependencies:  
👉 https://github.com/nerfstudio-project/gsplat#installation

Once installed, make sure the `gsplat` Python package is available in your environment.

## Getting Started

Please follow the **official Compact3D documentation** for data preparation, training, rendering, and evaluation:  
👉 https://github.com/UCDvision/compact3d#readme

## **Modifications Summary**

The following components have been **modified or extended** to support `gsplat` rendering:

\- `scene/gaussian_model.py`  

 → Rewrote `add_densification_stats()` to support `absgrad` and screen-space scaling based on image size.

\- `gaussian_renderer/__init__.py`  

 → Replaced all rasterizer calls with `gsplat.rasterization()`.

\- `train_kmeans.py`  

 → Updated training loop to support GSPLaT gradients and `max_radii2D` image-space scaling.  

 → Added memory usage printout (`torch.cuda.max_memory_allocated()`).

These changes are **minimal, focused**, and preserve the original Compact3D pipeline and structure.

## Citation

If you use this repository, please cite:

```bibtex
@article{navaneet2023compact3d,
  title={CompGS: Smaller and Faster Gaussian Splatting with Vector Quantization},
  author={Navaneet, KL and Meibodi, Kossar Pourahmadi and Koohpayegani, Soroush Abbasi and Pirsiavash, Hamed},
  journal={ECCV},
  year={2024}
}
```
