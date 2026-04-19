# DEVELOPMENT.md — JointPCA Filtered Variant Internals

This document describes the PC selection protocol used in the **filtered variant** (`filtered: true`). It is not needed to reproduce the main results — those use the full-spectrum variant (`filtered: false`).

---

## Overview

The filtered variant restricts the Mahalanobis score to a spectral interval $I = [T_1, T_2]$:

$$d^2(x) = \sum_{\alpha:\, \lambda_\alpha \in I} \frac{q_\alpha(x)^2}{\lambda_\alpha}$$

$T_1$ and $T_2$ are determined automatically from the eigenvalue spectrum and the participation ratio of each PC. No manual tuning is required for ResNet-50 and ViT on ImageNet-1K with the default `min_spike_gap = 5.0`.

---

## Participation ratio

For PC $u_\alpha$ (a unit vector of dimension $D$), split into $m$ layer blocks according to `layer_dims`:

$$\kappa_\alpha^{(\ell)} = \| u_\alpha[\text{block}_\ell] \|_2$$

with $\sum_\ell (\kappa_\alpha^{(\ell)})^2 = 1$. The **participation ratio** is:

$$\mathcal{N}_\alpha = \frac{1}{\sum_{\ell=1}^{m} (\kappa_\alpha^{(\ell)})^4}$$

Range: $[1, m]$.
- $\mathcal{N}_\alpha = m$: mode uniformly distributed across all layers (maximally joint).
- $\mathcal{N}_\alpha = 1$: mode localised on a single layer.

Modes with high $\mathcal{N}_\alpha$ are the most genuinely multi-layer and are most informative for OOD detection.

---

## T1 — noise-spike cutoff

Build a histogram of $\log_{10}(\lambda_\alpha)$ with bin width 0.15. Find peaks with prominence ≥ 10.

**ResNet-50:** The spectrum has a pronounced cluster of very-low-variance PCs on the far left (small eigenvalue), well separated from the signal region. Two histogram peaks appear with a gap ≥ `min_spike_gap` (default 5.0) log₁₀ units. The leftmost peak is the noise spike; $T_1$ is placed at the valley between the two peaks, discarding those noisy components.

**ViT:** No such spike exists (peaks are too close together). $T_1$ is set to the minimum $\log_{10}(\lambda)$ automatically — nothing is excluded on the left.

---

## T2 — participation ratio boundary

Let $\alpha^* = \arg\max_\alpha \, \mathcal{N}_\alpha$ be the index of the PC with the highest participation ratio. Then:

$$T_2 = \log_{10}(\lambda_{\alpha^*})$$

This is the eigenvalue of the single PC most broadly shared across all layers. The selected interval $[T_1, T_2]$ therefore captures the spectral region where modes are simultaneously signal-bearing (above the noise floor) and multi-layer (below the PC that is most broadly distributed).

---

## Verifying T1/T2 placement

Every run with `filtered: true` saves a two-panel plot to `results/jointpca_cache/plots/spectrum_{fp}.png`:

- **Top panel:** $\log_{10}(\lambda)$ histogram with T1 (red solid), T2 (blue dashed), and the selected zone (blue shaded).
- **Bottom panel:** $\mathcal{N}_\alpha per PC plotted against $\log_{10}(\lambda_\alpha)$, with the argmax marked.

Check that:
1. The red T1 line sits at the valley between noise spike and signal (ResNet), or at the leftmost eigenvalue (ViT).
2. The blue T2 line sits at a visible peak of the participation ratio curve.
3. The blue shaded region covers a reasonable signal band.

---

## Adjusting `min_spike_gap`

`min_spike_gap` (default 5.0) controls how far apart two histogram peaks must be (in $\log_{10}$ units) to declare a noise spike. It lives in `__init__` of `JointPCAPostprocessor`:

```python
self.min_spike_gap = 5.0
```

- **Larger value** (e.g. 7.0): stricter — only a very pronounced, isolated spike is detected. If T1 is falling too far right and cutting into signal PCs, increase this.
- **Smaller value** (e.g. 3.0): more permissive — smaller separations count as a spike. If the spike is clearly visible in the plot but T1 defaults to leftmost, decrease this.

The default of 5.0 is correct for ResNet-50 on ImageNet-1K and ViT on ImageNet-1K.

---

## Implementation location

All PC selection logic lives in `jointpca_utils.py`:

- `compute_participation_ratios` — computes $\mathcal{N}_\alpha$ for all PCs
- `select_pcs` — applies T1/T2 protocol, returns boolean mask
- `save_spectrum_plot` — generates the two-panel diagnostic plot

The postprocessor class (`jointpca_postprocessor.py`) calls these functions during `setup()` when `filtered: true` and stores the resulting boolean mask in `self.selected_mask`. The `mahalanobis_scores` function then applies the mask at inference time.
