"""
jointpca_utils.py
-----------------
Pure utility functions for Joint-PCA. No class state, no OpenOOD imports.

Contents:
  pool_activation    — CNN global-avg-pool or ViT CLS+patch-GAP
  mahalanobis_scores — ALL-PC Mahalanobis distance in PCA space
"""

import numpy as np
import torch


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
# Mahalanobis scoring                                                     #
# ====================================================================== #

def mahalanobis_scores(features: np.ndarray,
                        mean: np.ndarray,
                        components: np.ndarray,
                        explained_variance: np.ndarray) -> np.ndarray:
    """
    ALL-PC squared Mahalanobis distance in PCA space.

        score(z) = sum_alpha  q_alpha(z)^2 / lambda_alpha

    where q_alpha(z) = u_alpha^T (z - mu).

    Higher score -> more OOD.

    Parameters
    ----------
    features           : (N, D)
    mean               : (D,)
    components         : (K, D) -- PCA eigenvectors, unit rows
    explained_variance : (K,)   -- eigenvalues

    Returns
    -------
    scores : (N,) float64
    """
    centered = np.asarray(features, dtype=np.float32) - mean
    proj     = centered @ components.T                  # (N, K)
    var      = explained_variance.astype(np.float64) + 1e-10
    return np.sum((proj.astype(np.float64) ** 2) / var, axis=1)
