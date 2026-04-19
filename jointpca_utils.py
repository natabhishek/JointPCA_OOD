"""
jointpca_utils.py
-----------------
Pure utility functions for JointPCA. No class state, no OpenOOD imports.
All functions are independently testable.

Contents:
  pool_activation          — CNN global-avg-pool or ViT CLS+patch-GAP
  compute_layer_dims       — split joint component vector into per-layer blocks
  compute_participation_ratios — N_alpha: layer-spread of each PC
  select_pcs               — boolean mask for filtered variant [T1, T2]
  mahalanobis_scores       — spectrally restricted Mahalanobis distance
  save_spectrum_plot       — eigenvalue spectrum with T1/T2 marked (optional)
"""

import os
import numpy as np
import torch
from typing import Optional


# ====================================================================== #
# Pooling                                                                 #
# ====================================================================== #

def pool_activation(act: torch.Tensor) -> torch.Tensor:
    """
    Reduce a layer activation to a 2-D (batch, features) tensor.

    CNN (B, C, H, W)  → global average pool → (B, C)
    ViT (B, T, D)     → CLS token + patch-GAP → (B, 2D)
    Other (B, D)      → identity
    """
    if act.dim() == 4:
        return act.mean(dim=[2, 3])
    if act.dim() == 3:
        cls_tok   = act[:, 0, :]
        patch_gap = act[:, 1:, :].mean(dim=1)
        return torch.cat([cls_tok, patch_gap], dim=1)
    return act


# ====================================================================== #
# Participation ratio                                                     #
# ====================================================================== #

def compute_layer_dims(components: np.ndarray,
                       layer_dims: list) -> list:
    """
    Verify that layer_dims sums to the component width.
    Returns layer_dims unchanged (used as a sanity check at call sites).
    """
    total = sum(layer_dims)
    if total != components.shape[1]:
        raise ValueError(
            f'layer_dims sum ({total}) does not match '
            f'component width ({components.shape[1]})'
        )
    return layer_dims


def compute_participation_ratios(components: np.ndarray,
                                  layer_dims: list) -> np.ndarray:
    """
    Compute the participation ratio N_alpha for each PC alpha.

    For PC u_alpha (shape D), split into layer blocks according to
    layer_dims. The layer contribution is the L2 norm of each block:

        kappa_alpha^(l) = || u_alpha[block_l] ||_2

    with sum_l kappa^2 = 1 (since u_alpha is a unit vector).

    The participation ratio is:

        N_alpha = 1 / sum_l ( kappa_alpha^(l) )^4

    Range: [1, m] where m = number of layers.
    N_alpha = m  → mode spread uniformly across all layers (most joint)
    N_alpha = 1  → mode localised on a single layer

    Parameters
    ----------
    components : (K, D) float32
        PCA components, each row is a unit vector.
    layer_dims : list[int]
        Width of each layer's feature block. Must sum to D.

    Returns
    -------
    pr : (K,) float64
        Participation ratio for each PC.
    """
    compute_layer_dims(components, layer_dims)   # sanity check

    K      = components.shape[0]
    pr     = np.zeros(K, dtype=np.float64)
    splits = np.cumsum([0] + layer_dims)

    for alpha in range(K):
        u       = components[alpha].astype(np.float64)
        kappas  = np.array([
            np.linalg.norm(u[splits[i]:splits[i + 1]])
            for i in range(len(layer_dims))
        ])
        # kappas^2 should sum to 1; enforce numerically
        kappas2 = kappas ** 2
        kappas2 /= kappas2.sum() + 1e-30
        pr[alpha] = 1.0 / (np.sum(kappas2 ** 2) + 1e-30)

    return pr


# ====================================================================== #
# PC selection (filtered variant only)                                    #
# ====================================================================== #

def select_pcs(explained_variance: np.ndarray,
               participation_ratio: np.ndarray,
               min_spike_gap: float = 5.0,
               plot_path: Optional[str] = None,
               config_fp: Optional[str] = None) -> np.ndarray:
    """
    Compute a boolean mask selecting PCs in the spectral interval [T1, T2].

    PCs are ordered by decreasing eigenvalue (sklearn default), so index 0
    is the largest eigenvalue and index K-1 is the smallest.
    log10(ev) therefore runs from large (left, high-variance) to small
    (right, low-variance) as index increases — but the histogram is built
    on the values themselves so the x-axis runs low-to-high.

    T1  (noise-spike cutoff, left boundary in eigenvalue space)
        Build a histogram of log10(ev) with bin width 0.15.
        Find peaks with prominence >= 10.
        If two peaks exist separated by >= min_spike_gap log10 units,
        the leftmost (smallest eigenvalue) peak is a noise spike;
        T1 = valley between the two peaks.
        Otherwise (ViT or no spike) T1 = min(log10(ev)).

    T2  (participation-ratio boundary, right boundary in eigenvalue space)
        T2 = log10(ev[alpha*])  where alpha* = argmax(N_alpha).
        This is the eigenvalue of the PC most broadly shared across layers.
        Selected interval: T1 <= log10(ev) <= T2.

    Parameters
    ----------
    explained_variance  : (K,) — eigenvalues from PCA, decreasing order
    participation_ratio : (K,) — N_alpha per PC
    min_spike_gap       : float — minimum log10-unit gap to declare a spike
    plot_path           : str or None — if given, save spectrum plot here
    config_fp           : str or None — label for the plot title

    Returns
    -------
    mask : (K,) bool — True for selected PCs
    """
    from scipy.signal import find_peaks

    ev       = explained_variance.astype(np.float64)
    log10_ev = np.log10(np.maximum(ev, 1e-30))
    xmin     = float(log10_ev.min())

    # ── T1: noise-spike detection ─────────────────────────────────────── #
    bin_width   = 0.15
    bin_edges   = np.arange(xmin - bin_width,
                            log10_ev.max() + bin_width * 2,
                            bin_width)
    counts, _   = np.histogram(log10_ev, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    hist_peaks, _ = find_peaks(counts, prominence=10)
    hist_peaks     = hist_peaks[np.argsort(bin_centers[hist_peaks])]

    has_spike = False
    T1        = xmin

    if len(hist_peaks) >= 2:
        gap = bin_centers[hist_peaks[1]] - bin_centers[hist_peaks[0]]
        if gap >= min_spike_gap:
            spike_peak    = hist_peaks[0]
            search_end    = hist_peaks[1]
            valley_region = counts[spike_peak: search_end + 1]
            valley_idx    = spike_peak + int(np.argmin(valley_region))
            T1            = float(bin_edges[valley_idx + 1])
            has_spike     = True
            print(f'[JointPCA][Filter] Noise spike detected. '
                  f'T1 (valley) = {T1:.4f}')
        else:
            print(f'[JointPCA][Filter] Peaks too close (gap={gap:.3f} < '
                  f'{min_spike_gap}) — no spike. T1 = leftmost = {T1:.4f}')
    else:
        print(f'[JointPCA][Filter] No spike detected. '
              f'T1 = leftmost = {T1:.4f}')

    # ── T2: eigenvalue of the PC with maximum participation ratio ─────── #
    alpha_star = int(np.argmax(participation_ratio))
    T2         = float(log10_ev[alpha_star])
    print(f'[JointPCA][Filter] T2 = log10(ev[{alpha_star}]) = {T2:.4f}  '
          f'(N_alpha* = {participation_ratio[alpha_star]:.2f})')

    # ── Selected PCs ─────────────────────────────────────────────────── #
    mask  = (log10_ev >= T1) & (log10_ev <= T2)
    n_sel = int(mask.sum())
    print(f'[JointPCA][Filter] T1={T1:.4f}  T2={T2:.4f}  '
          f'Selected PCs: {n_sel} / {len(mask)}')

    if n_sel == 0:
        raise RuntimeError(
            '[JointPCA] No PCs selected by filter protocol. '
            'Check eigenvalue spectrum and min_spike_gap. '
            'See DEVELOPMENT.md for guidance.'
        )

    if plot_path is not None:
        save_spectrum_plot(
            log10_ev=log10_ev, bin_centers=bin_centers, counts=counts,
            hist_peaks=hist_peaks, participation_ratio=participation_ratio,
            T1=T1, T2=T2, alpha_star=alpha_star,
            has_spike=has_spike, n_sel=n_sel,
            plot_path=plot_path, config_fp=config_fp or '',
            min_spike_gap=min_spike_gap,
        )

    return mask


# ====================================================================== #
# Mahalanobis scoring                                                     #
# ====================================================================== #

def mahalanobis_scores(features: np.ndarray,
                        mean: np.ndarray,
                        components: np.ndarray,
                        explained_variance: np.ndarray,
                        selected_mask: Optional[np.ndarray] = None
                        ) -> np.ndarray:
    """
    Spectrally restricted squared Mahalanobis distance.

        score(z) = sum_{alpha in I}  q_alpha(z)^2 / lambda_alpha

    where q_alpha(z) = u_alpha^T (z - mu).

    When selected_mask is None all PCs are used (full-spectrum variant).
    When selected_mask is a boolean array only the True PCs are used
    (filtered variant).

    Higher score → more OOD.

    Parameters
    ----------
    features           : (N, D)
    mean               : (D,)
    components         : (K, D) — PCA eigenvectors, unit rows
    explained_variance : (K,)   — eigenvalues
    selected_mask      : (K,) bool or None

    Returns
    -------
    scores : (N,) float64
    """
    if selected_mask is not None:
        comps = components[selected_mask]
        ev    = explained_variance[selected_mask].astype(np.float64)
    else:
        comps = components
        ev    = explained_variance.astype(np.float64)

    centered = np.asarray(features, dtype=np.float32) - mean
    proj     = centered @ comps.T                          # (N, n_sel)
    var      = ev + 1e-10
    return np.sum((proj.astype(np.float64) ** 2) / var, axis=1)


# ====================================================================== #
# Spectrum plot (optional, only generated when filtered=True)            #
# ====================================================================== #

def save_spectrum_plot(log10_ev, bin_centers, counts, hist_peaks,
                        participation_ratio, T1, T2, alpha_star,
                        has_spike, n_sel, plot_path, config_fp,
                        min_spike_gap):
    """
    Save the eigenvalue spectrum with T1, T2, and participation ratio overlay.
    Generated automatically when filtered=True. See DEVELOPMENT.md.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[JointPCA] matplotlib not available — skipping spectrum plot.')
        return

    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    xmin = float(log10_ev.min())
    xmax = float(log10_ev.max())
    bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.15

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # ── Top: eigenvalue histogram ─────────────────────────────────────── #
    ax1.bar(bin_centers, counts, width=bin_width * 0.9,
            alpha=0.5, color='steelblue', edgecolor='gray', linewidth=0.4,
            label='Histogram of log₁₀(λ)')
    ax1.axvline(T1, color='red',  lw=2.5, ls='-',
                label=f'T1 = {T1:.3f}  '
                      f'({"spike valley" if has_spike else "leftmost"})')
    ax1.axvline(T2, color='blue', lw=2.0, ls='--',
                label=f'T2 = {T2:.3f}  (max N_α at PC {alpha_star})')
    ax1.axvspan(T1, T2, alpha=0.12, color='blue',
                label=f'Selected zone ({n_sel} PCs)')
    if has_spike:
        ax1.axvspan(xmin - 0.5, T1, alpha=0.07, color='red',
                    label='Noise spike (excluded)')
    for i, pk in enumerate(hist_peaks[:4]):
        ax1.axvline(bin_centers[pk], color='orange', lw=1.0, ls=':',
                    label=f'Peak {i} = {bin_centers[pk]:.2f}' if i < 2 else None)
    ax1.set_ylabel('Count (per bin)', fontsize=10)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_title(
        f'Eigenvalue spectrum — {config_fp}\n'
        f'T1={T1:.3f}  T2={T2:.3f}  Selected: {n_sel} PCs  '
        f'Spike: {"detected" if has_spike else "not found"}  '
        f'min_spike_gap={min_spike_gap}',
        fontsize=9, fontweight='bold'
    )

    # ── Bottom: participation ratio per PC (vs log10 eigenvalue) ─────── #
    # PCs are sorted by decreasing eigenvalue, so we plot against log10_ev
    ax2.scatter(log10_ev, participation_ratio, s=2, alpha=0.4,
                color='steelblue', label='N_α per PC')
    ax2.axvline(T1, color='red',  lw=2.5, ls='-')
    ax2.axvline(T2, color='blue', lw=2.0, ls='--')
    ax2.axvspan(T1, T2, alpha=0.12, color='blue')
    ax2.scatter([log10_ev[alpha_star]], [participation_ratio[alpha_star]],
                s=60, color='blue', zorder=5,
                label=f'argmax N_α = PC {alpha_star}  '
                      f'(N_α = {participation_ratio[alpha_star]:.2f})')
    ax2.set_xlabel('log₁₀(eigenvalue)', fontsize=10)
    ax2.set_ylabel('Participation ratio N_α', fontsize=10)
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'[JointPCA] Spectrum plot saved: {plot_path}')
    print('[JointPCA] Inspect this plot to verify T1/T2 placement. '
          'See DEVELOPMENT.md if adjustments are needed.')
