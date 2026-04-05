"""
JointPCA OOD Detection Postprocessor for OpenOOD
-------------------------------------------------
Extracts multi-layer activations via forward hooks, concatenates them into
a joint feature vector, fits PCA (sklearn randomized), and scores test
samples using the clipped-Mahalanobis protocol:

  1. Compute log10(eigenvalue) spectrum of the fitted PCA.
  2. Detect T1 (noise-spike cutoff) and T2 (mean of signal):
       - ResNet50: histogram peak detection finds the noise spike on the
         far left of the spectrum; T1 = valley right of that spike.
         If no spike is found, T1 = leftmost eigenvalue.
       - ViT: no noise spike present (peaks are too close together);
         T1 = leftmost eigenvalue automatically.
       T2 = mean(log10(ev)) for all PCs with log10(ev) >= T1.
  3. Selected PCs: those with T1 <= log10(ev) <= T2.
  4. Mahalanobis distance computed only on those selected PCs.

Layer strategy:
  - ResNet50 : all Conv2d layers + residual block outputs + penultimate
  - ViT      : CLS token + patch-GAP from every encoder block

Caching:
  All intermediate artefacts (features, PCA, projections, scores) are
  stored under ./jointpca_cache/ as memory-mapped NumPy arrays so that
  repeated runs on the same dataset skip expensive recomputation.

Usage:
  Set `postprocessor.name: jointpca` in your pipeline config and point
  to configs/postprocessors/jointpca.yml.  See README.md for full
  instructions.
"""

import os
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from tqdm import tqdm
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')   # non-interactive backend, safe on servers
import matplotlib.pyplot as plt

from .base_postprocessor import BasePostprocessor


class JointPCAPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super().__init__(config)

        if hasattr(config.postprocessor, 'postprocessor_args'):
            self.args = config.postprocessor.postprocessor_args
        else:
            self.args = config.postprocessor

        self.args_dict = getattr(config.postprocessor, 'postprocessor_sweep', {})

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f'[JointPCA] GPU: {torch.cuda.get_device_name(0)}')
        else:
            print('[JointPCA] GPU not available, using CPU')

        self.mean               = None
        self.components         = None
        self.explained_variance = None
        self.n_components_used  = None
        self.selected_mask      = None   # boolean mask: PCs in [T1, T2]
        self.min_spike_gap      = 5.0    # log10(ev) units; see clip protocol
        self._config_fp         = None
        self.cache_dir          = './jointpca_cache'
        self.layer_names        = []
        self.hooks              = []
        self.activations        = {}

    # ------------------------------------------------------------------ #
    # Cache helpers                                                        #
    # ------------------------------------------------------------------ #

    def _ensure_cache_dirs(self):
        for sub in ('features', 'scores', 'pca', 'metadata', 'projections'):
            os.makedirs(os.path.join(self.cache_dir, sub), exist_ok=True)

    @staticmethod
    def _make_config_fp(model_name, max_samples):
        s = f'{max_samples // 1000}k' if max_samples >= 1000 else str(max_samples)
        return f'{model_name}_{s}'

    def _p(self, subdir, name):
        return os.path.join(self.cache_dir, subdir, name)

    def _path_feat_train(self):
        return self._p('features', f'features_train_{self._config_fp}.npy')

    def _path_feat_ood(self, dataset):
        return self._p('features', f'features_{dataset}_{self._config_fp}.npy')

    def _path_meta(self):
        return self._p('metadata', f'metadata_{self._config_fp}.npz')

    def _path_pca(self):
        return self._p('pca', f'pca_{self._config_fp}.npz')

    def _path_proj(self):
        return self._p('projections',
                       f'projections_train_{self._config_fp}.npy')

    def _path_scores(self, dataset):
        return self._p('scores', f'scores_{dataset}_{self._config_fp}_clipmd.npz')

    @staticmethod
    def _mmap_write(path, shape, dtype=np.float32):
        return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)

    @staticmethod
    def _mmap_read(path):
        if not os.path.exists(path):
            return None
        return np.load(path, mmap_mode='r')

    # ------------------------------------------------------------------ #
    # Hook management                                                      #
    # ------------------------------------------------------------------ #

    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _is_vit(self, model):
        """Return True when the model looks like a Vision Transformer."""
        for _, m in model.named_modules():
            if isinstance(m, nn.MultiheadAttention):
                return True
        return False

    def _get_vit_block_names(self, model):
        block_names = []
        seen_ids    = set()
        for name, module in model.named_modules():
            mid = id(module)
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            children = list(module.children())
            if not children:
                continue
            has_attn = any(isinstance(c, nn.MultiheadAttention) for c in children)
            has_norm = any(isinstance(c, nn.LayerNorm)          for c in children)
            if has_attn and has_norm:
                block_names.append(name)
        return block_names

    def _get_resnet_block_names(self, model):
        """Match ResNet50 residual block outputs: layer1.0, layer2.1, etc."""
        r_resnet = re.compile(r'^layer[1-4]\.\d+$')
        return [name for name, _ in model.named_modules() if r_resnet.match(name)]

    def _find_penultimate(self, model):
        p_name, p_mod = None, None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                break
            if (len(list(module.children())) == 0 and
                    not isinstance(module, nn.Conv2d)):
                p_name, p_mod = name, module
        return p_name, p_mod

    def _register_hooks(self, model):
        def make_hook(name):
            def hook(module, input, output):
                self.activations[name] = (
                    output.detach() if self.device.type == 'cuda'
                    else output.detach().cpu()
                )
            return hook

        self._clear_hooks()
        self.layer_names = []
        self.activations = {}
        count = 0

        if self._is_vit(model):
            vit_blocks = self._get_vit_block_names(model)
            if not vit_blocks:
                raise RuntimeError('[JointPCA] No ViT encoder blocks found.')
            for name, m in model.named_modules():
                if name in vit_blocks:
                    self.hooks.append(m.register_forward_hook(make_hook(name)))
                    self.layer_names.append(name)
                    count += 1
            print(f'[JointPCA] ViT blocks hooked: {count}')
        else:
            # ResNet50: all Conv2d + residual block outputs + penultimate
            block_names = self._get_resnet_block_names(model)
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    self.hooks.append(m.register_forward_hook(make_hook(name)))
                    self.layer_names.append(name)
                    count += 1
            for name, m in model.named_modules():
                if name in block_names:
                    self.hooks.append(m.register_forward_hook(make_hook(name)))
                    self.layer_names.append(name)
                    count += 1
            p_name, p_mod = self._find_penultimate(model)
            if p_name:
                self.hooks.append(p_mod.register_forward_hook(make_hook(p_name)))
                self.layer_names.append(p_name)
                count += 1
            print(f'[JointPCA] ResNet50 hooks registered: {count} layers')

    def _attach_hooks(self, model):
        """Re-attach hooks using the layer names discovered during setup."""
        def make_hook(name):
            def hook(module, input, output):
                self.activations[name] = (
                    output.detach() if self.device.type == 'cuda'
                    else output.detach().cpu()
                )
            return hook

        self._clear_hooks()
        self.activations = {}
        name_set = set(self.layer_names)
        for name, m in model.named_modules():
            if name in name_set:
                self.hooks.append(m.register_forward_hook(make_hook(name)))
        print(f'[JointPCA] Re-attached {len(self.hooks)} hooks '
              f'(expected {len(self.layer_names)})')

    # ------------------------------------------------------------------ #
    # Pooling                                                              #
    # ------------------------------------------------------------------ #

    def _pool(self, act):
        if act.dim() == 4:
            return act.mean(dim=[2, 3])          # global average pool (CNN)
        if act.dim() == 3:
            cls_tok   = act[:, 0, :]             # CLS token
            patch_gap = act[:, 1:, :].mean(dim=1)
            return torch.cat([cls_tok, patch_gap], dim=1)  # CLS + patch-GAP
        return act

    # ------------------------------------------------------------------ #
    # Feature extraction                                                   #
    # ------------------------------------------------------------------ #

    def _extract_batch(self, net, data, run_forward=True):
        if run_forward:
            with torch.no_grad():
                net(data)
        parts = [
            self._pool(self.activations[n]).cpu().numpy()
            for n in self.layer_names
            if n in self.activations
        ]
        self.activations = {}
        return np.concatenate(parts, axis=1) if parts else None

    def _extract_loader_to_mmap(self, net, loader, path, max_n, feat_dim, desc):
        mm  = self._mmap_write(path, (max_n, feat_dim))
        idx = 0
        net.eval()
        for batch in tqdm(loader, desc=desc):
            data  = batch['data'] if isinstance(batch, dict) else batch[0]
            feats = self._extract_batch(net, data.to(self.device))
            if feats is None:
                continue
            bs  = feats.shape[0]
            end = min(idx + bs, max_n)
            mm[idx:end] = feats[:end - idx].astype(np.float32)
            idx = end
            if idx % 5120 == 0:
                mm.flush()
            if idx >= max_n:
                break
        mm.flush()
        del mm
        return idx

    # ------------------------------------------------------------------ #
    # Dataset name (parsed from loader metadata)                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dataset_name(loader):
        def parse_name(raw):
            if raw is None:
                return None
            fname = os.path.basename(str(raw)).replace('.txt', '')
            for prefix in ('test_', 'val_', 'train_'):
                if fname.startswith(prefix):
                    stem = fname[len(prefix):]
                    return 'imagenet_test' if (prefix == 'test_' and stem == 'imagenet') else stem
            return fname

        ds   = getattr(loader, 'dataset', None)
        seen = set()
        while ds is not None and id(ds) not in seen:
            seen.add(id(ds))
            for attr in ('imglist_pth', 'imglist_path', 'imglist_file',
                         'ann_file', 'meta_file'):
                if hasattr(ds, attr):
                    parsed = parse_name(getattr(ds, attr))
                    if parsed:
                        return parsed
            if hasattr(ds, 'imglist'):
                try:
                    line  = str(ds.imglist[0]).strip()
                    root  = line.split(' ', 1)[0].split('/', 1)[0]
                    if root in ('imagenet_1k', 'imagenet'):
                        return 'imagenet_test'
                    if root not in ('', '.'):
                        return root
                except Exception:
                    pass
            ds = getattr(ds, 'dataset', None)

        ds = getattr(loader, 'dataset', None)
        if ds is not None and hasattr(ds, 'name'):
            return str(ds.name)
        return 'unknown'

    # ------------------------------------------------------------------ #
    # PCA                                                                  #
    # ------------------------------------------------------------------ #

    def _fit_pca(self, train_features, actual_n_train, feat_dim, n_comp):
        from sklearn.decomposition import PCA
        import gc

        print(f'[JointPCA] Fitting PCA: N={actual_n_train}, D={feat_dim}, K={n_comp}')
        X   = np.asarray(train_features[:actual_n_train], dtype=np.float32)
        # 'randomized' requires n_components < min(n_samples, n_features), which is
        # guaranteed by the n_comp = min(N, D) - 1 cap applied in setup().
        pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=0, copy=False)
        pca.fit(X)
        del X
        gc.collect()

        mean      = pca.mean_.astype(np.float32)
        comps     = pca.components_.astype(np.float32)
        ev        = pca.explained_variance_.astype(np.float32)
        ev_ratio  = pca.explained_variance_ratio_.astype(np.float32)
        print(f'[JointPCA] PCA done. Top-100 PCs explain '
              f'{ev_ratio[:100].sum() * 100:.1f}% variance.')
        return mean, comps, ev, ev_ratio

    # ------------------------------------------------------------------ #
    # setup()                                                              #
    # ------------------------------------------------------------------ #

    def setup(self, net, id_loader_dict, ood_loader_dict=None):
        os.makedirs(self.cache_dir, exist_ok=True)
        self._ensure_cache_dirs()
        net.eval()
        net = net.to(self.device)

        model_name  = net.__class__.__name__.lower()
        max_samples = int(self.args.get('max_train_samples', 50000))

        self._register_hooks(net)
        print(f'[JointPCA] Layer names ({len(self.layer_names)} total):')
        for n in self.layer_names:
            print(f'  {n}')

        # ── 1. Probe feature dimension ──────────────────────────────── #
        train_loader = id_loader_dict['train']
        for batch in train_loader:
            data     = batch['data'] if isinstance(batch, dict) else batch[0]
            probe    = self._extract_batch(net, data.to(self.device))
            feat_dim = probe.shape[1]
            break

        self._config_fp = self._make_config_fp(model_name, max_samples)
        print(f'[JointPCA] Config FP  : {self._config_fp}')
        print(f'[JointPCA] Feature dim: {feat_dim}')

        # ── 2. ID training features (cached) ───────────────────────── #
        skip = False
        if os.path.exists(self._path_feat_train()) and os.path.exists(self._path_meta()):
            try:
                meta = np.load(self._path_meta(), allow_pickle=True)
                if (int(meta['n_features']) == feat_dim and
                        int(meta['n_samples']) > 0 and
                        list(meta['layer_names']) == self.layer_names):
                    print(f'[JointPCA] ID train features cached: {self._path_feat_train()}')
                    skip = True
                else:
                    print('[JointPCA] Cache config changed -- re-extracting...')
            except Exception as e:
                print(f'[JointPCA] Cache error: {e} -- re-extracting...')

        if not skip:
            n = self._extract_loader_to_mmap(
                net, train_loader, self._path_feat_train(),
                max_samples, feat_dim, 'ID train features'
            )
            np.savez(self._path_meta(), n_samples=n, n_features=feat_dim,
                     layer_names=np.array(self.layer_names, dtype=object))
            print(f'[JointPCA] Saved ({n}, {feat_dim})')

        meta           = np.load(self._path_meta(), allow_pickle=True)
        actual_n_train = int(meta['n_samples'])
        train_features = self._mmap_read(self._path_feat_train())[:actual_n_train]

        # Free model memory before PCA
        self._clear_hooks()
        self.activations = {}
        del net
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── 3. n_components ────────────────────────────────────────── #
        # sklearn randomized solver requires n_components < min(N, D).
        # We cap at min(N, D) - 1; for very small datasets use 'full' instead.
        n_comp = min(actual_n_train, feat_dim) - 1
        self.n_components_used = n_comp

        # ── 4. PCA (cached) ─────────────────────────────────────────── #
        pca_path = self._path_pca()
        skip_pca = False
        if os.path.exists(pca_path):
            try:
                d = np.load(pca_path, allow_pickle=True)
                if int(d['n_components']) == n_comp:
                    print(f'[JointPCA] PCA cached: {pca_path}')
                    self.mean               = d['mean']
                    self.components         = d['components']
                    self.explained_variance = d['explained_variance']
                    skip_pca = True
                else:
                    print('[JointPCA] n_components changed -- refitting PCA...')
            except Exception as e:
                print(f'[JointPCA] PCA cache error: {e} -- refitting...')

        if not skip_pca:
            (self.mean, self.components,
             self.explained_variance, ev_ratio) = self._fit_pca(
                train_features, actual_n_train, feat_dim, n_comp
            )
            np.savez(pca_path, n_components=n_comp,
                     mean=self.mean, components=self.components,
                     explained_variance=self.explained_variance,
                     explained_variance_ratio=ev_ratio)
            print(f'[JointPCA] PCA saved: {pca_path}')

        # ── 5. Train projections (cached) ───────────────────────────── #
        proj_path = self._path_proj()
        p = self._mmap_read(proj_path)
        if p is not None and p.shape == (actual_n_train, n_comp):
            print(f'[JointPCA] Projections cached: {proj_path}')
        else:
            print(f'[JointPCA] Computing train projections...')
            mm_proj = self._mmap_write(proj_path, (actual_n_train, n_comp))
            for i in tqdm(range(0, actual_n_train, 5000), desc='Projections'):
                chunk    = train_features[i:i + 5000]
                centered = chunk - self.mean
                proj     = (centered @ self.components.T).astype(np.float32)
                mm_proj[i:i + proj.shape[0]] = proj
                mm_proj.flush()
            mm_proj.flush()
            del mm_proj

        # ── 5. Select PCs via clip-to-mean protocol ────────────────── #
        self._select_pcs()

        print(f'[JointPCA] Setup complete.')

    # ------------------------------------------------------------------ #
    # PC selection: clip-to-mean protocol                                  #
    # ------------------------------------------------------------------ #

    def _select_pcs(self):
        """
        Compute the boolean mask selecting PCs in [T1, T2] of log10(ev),
        and save a spectrum plot to ./jointpca_cache/plots/ for visual
        verification that T1/T2 landed correctly.

        T1 (noise-spike cutoff):
          Build a histogram of log10(ev) with bin width 0.15.
          Find histogram peaks with prominence >= 10.
          If two peaks exist and the gap between them is >= min_spike_gap,
          the leftmost is a noise spike; T1 = valley between the two peaks.
          Otherwise (ViT, or no real spike) T1 = min(log10(ev)).

        T2 (mean of signal):
          T2 = mean(log10(ev)) for all PCs with log10(ev) >= T1.

        Selected PCs: T1 <= log10(ev) <= T2.
        """
        ev       = self.explained_variance.astype(np.float64)
        log10_ev = np.log10(np.maximum(ev, 1e-30))
        xmin, xmax = log10_ev.min(), log10_ev.max()

        bin_width   = 0.15
        bin_edges   = np.arange(xmin - bin_width, xmax + bin_width * 2, bin_width)
        counts, _   = np.histogram(log10_ev, bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        hist_peaks, _ = find_peaks(counts, prominence=10)
        hist_peaks     = hist_peaks[np.argsort(bin_centers[hist_peaks])]

        has_spike = False
        T1 = float(xmin)

        if len(hist_peaks) >= 2:
            gap = bin_centers[hist_peaks[1]] - bin_centers[hist_peaks[0]]
            print(f'[JointPCA][Clip] Histogram peaks at log10(ev): '
                  f'{[round(float(bin_centers[p]), 2) for p in hist_peaks[:4]]}')
            print(f'[JointPCA][Clip] Gap between peak 0 and peak 1: '
                  f'{gap:.3f} (min_spike_gap={self.min_spike_gap})')
            if gap >= self.min_spike_gap:
                spike_peak    = hist_peaks[0]
                search_end    = hist_peaks[1]
                valley_region = counts[spike_peak: search_end + 1]
                valley_idx    = spike_peak + int(np.argmin(valley_region))
                T1            = float(bin_edges[valley_idx + 1])
                has_spike     = True
                print(f'[JointPCA][Clip] Spike confirmed -- '
                      f'T1 (valley) = {T1:.4f}')
            else:
                print(f'[JointPCA][Clip] Peaks too close -- no spike, '
                      f'T1 = leftmost = {T1:.4f}')
        else:
            print(f'[JointPCA][Clip] No spike detected -- '
                  f'T1 = leftmost = {T1:.4f}')

        log_ev_right = log10_ev[log10_ev >= T1]
        T2 = float(np.mean(log_ev_right))

        mask    = (log10_ev >= T1) & (log10_ev <= T2)
        n_sel   = int(mask.sum())
        n_spike = int((log10_ev < T1).sum())
        n_right = int((log10_ev >= T1).sum())

        print(f'[JointPCA][Clip] T1={T1:.4f}  T2={T2:.4f}')
        print(f'[JointPCA][Clip] Spike PCs (<T1): {n_spike}  '
              f'Signal PCs (>=T1): {n_right}  '
              f'Selected [T1,T2]: {n_sel}')

        if n_sel == 0:
            raise RuntimeError(
                '[JointPCA] No PCs selected by clip protocol. '
                'Check eigenvalue spectrum.'
            )

        self.selected_mask = mask
        self._save_spectrum_plot(
            log10_ev, bin_centers, counts, hist_peaks,
            T1, T2, has_spike, n_spike, n_sel
        )

    def _save_spectrum_plot(self, log10_ev, bin_centers, counts,
                             hist_peaks, T1, T2, has_spike, n_spike, n_sel):
        """
        Save the log10(eigenvalue) spectrum to jointpca_cache/plots/ so
        the user can visually verify that T1 and T2 are correctly placed.
        Inspect this plot if results seem off -- adjust min_spike_gap if needed.
        """
        plot_dir = os.path.join(self.cache_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f'spectrum_{self._config_fp}.png')

        bin_width = bin_centers[1] - bin_centers[0] if len(bin_centers) > 1 else 0.15
        xmin, xmax = float(log10_ev.min()), float(log10_ev.max())

        fig, ax = plt.subplots(figsize=(11, 5))

        ax.bar(bin_centers, counts, width=bin_width * 0.9,
               alpha=0.5, color='steelblue', edgecolor='gray', linewidth=0.4,
               label='Histogram of log₁₀(λ)')

        ax.axvline(T1, color='red', lw=2.5, ls='-',
                   label=f"T1 ({'valley/spike end' if has_spike else 'leftmost — no spike'}) = {T1:.3f}")
        ax.axvline(T2, color='blue', lw=2.0, ls='--',
                   label=f'T2 (mean of signal) = {T2:.3f}')

        ax.axvspan(T1, T2, alpha=0.12, color='blue',
                   label=f'Selected zone ({n_sel} PCs)')
        if has_spike:
            ax.axvspan(xmin - 0.5, T1, alpha=0.07, color='red',
                       label=f'Noise spike ({n_spike} PCs, excluded)')

        # Mark detected histogram peaks
        for i, pk in enumerate(hist_peaks[:4]):
            ax.axvline(bin_centers[pk], color='orange', lw=1.0, ls=':',
                       label=f'Peak {i} = {bin_centers[pk]:.2f}' if i < 2 else None)

        ax.set_xlabel('log₁₀(eigenvalue)', fontsize=11)
        ax.set_ylabel('Count (per bin)', fontsize=11)
        ax.set_title(
            f'Eigenvalue spectrum — {self._config_fp}
'
            f'T1={T1:.3f}  T2={T2:.3f}  '
            f'Selected (blue): {n_sel} PCs  '
            f'Spike (red): {"detected" if has_spike else "not found"}  '
            f'min_spike_gap={self.min_spike_gap}',
            fontsize=9, fontweight='bold'
        )
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'[JointPCA][Clip] Spectrum plot saved: {plot_path}')
        print(f'[JointPCA][Clip] Inspect this plot to verify T1/T2 placement. '
              f'Adjust min_spike_gap in __init__ if needed (current={self.min_spike_gap}).')

    # ------------------------------------------------------------------ #
    # Mahalanobis scoring (on selected PCs only)                           #
    # ------------------------------------------------------------------ #

    def _mahalanobis_scores(self, features):
        """
        Squared Mahalanobis distance using only the selected PCs (clip protocol).

        score = sum_{i in selected} (y_i^2 / lambda_i)

        where y_i = u_i^T (z - mu) and selected = PCs with T1 <= log10(ev) <= T2.
        Higher score = more OOD.
        """
        sel_comps = self.components[self.selected_mask]          # (n_sel, D)
        sel_ev    = self.explained_variance[self.selected_mask].astype(np.float64)
        proj      = (np.asarray(features, dtype=np.float32) - self.mean) @ sel_comps.T
        var       = sel_ev + 1e-10
        return np.sum((proj.astype(np.float64) ** 2) / var, axis=1)

    # ------------------------------------------------------------------ #
    # inference()                                                          #
    # ------------------------------------------------------------------ #

    def inference(self, net, data_loader):
        dataset    = self._dataset_name(data_loader)
        feat_path  = self._path_feat_ood(dataset)
        score_path = self._path_scores(dataset)
        feat_dim   = self.components.shape[1]

        print(f'\n[JointPCA] Inference: {dataset}')

        net.eval()
        net = net.to(self.device)
        self._attach_hooks(net)

        # Score cache check
        if os.path.exists(score_path):
            try:
                sc = np.load(score_path, allow_pickle=True)
                n  = int(sc['n_samples'])
                print(f'[JointPCA] Loaded score cache ({n} samples)')
                return sc['pred'], -sc['scores'], sc['labels']
            except Exception as e:
                print(f'[JointPCA] Score cache error: {e} -- recomputing...')

        # Feature extraction or reuse
        feat_mm   = self._mmap_read(feat_path)
        total_est = getattr(data_loader, 'dataset', None)
        total_est = len(total_est) if total_est is not None else 200000

        all_pred = None
        all_lbl  = None
        N        = None

        if (feat_mm is not None and
                feat_mm.ndim == 2 and
                feat_mm.shape[1] == feat_dim):
            N = feat_mm.shape[0]
            print(f'[JointPCA] Reusing cached features: ({N}, {feat_dim})')
            preds_list, lbls_list = [], []
            for batch in tqdm(data_loader, desc=f'Labels: {dataset}'):
                if isinstance(batch, dict):
                    data, lbl = batch['data'], batch['label']
                else:
                    data, lbl = batch[0], batch[1]
                with torch.no_grad():
                    out = net(data.to(self.device))
                _, pred = torch.softmax(out, dim=1).max(dim=1)
                self.activations = {}
                preds_list.append(pred.cpu().numpy())
                lbls_list.append(
                    lbl.numpy() if isinstance(lbl, torch.Tensor) else np.array(lbl)
                )
            all_pred = np.concatenate(preds_list)[:N]
            all_lbl  = np.concatenate(lbls_list)[:N]
        else:
            mm            = self._mmap_write(feat_path, (total_est, feat_dim))
            all_pred_list = []
            all_lbl_list  = []
            idx           = 0

            for batch in tqdm(data_loader, desc=f'Features: {dataset}'):
                if isinstance(batch, dict):
                    data, lbl = batch['data'], batch['label']
                else:
                    data, lbl = batch[0], batch[1]
                data = data.to(self.device)
                with torch.no_grad():
                    out = net(data)
                _, pred = torch.softmax(out, dim=1).max(dim=1)
                parts = [
                    self._pool(self.activations[n]).cpu().numpy()
                    for n in self.layer_names
                    if n in self.activations
                ]
                self.activations = {}
                if not parts:
                    print('[JointPCA] WARNING: no activations captured')
                    continue
                feats = np.concatenate(parts, axis=1)
                bs    = feats.shape[0]
                mm[idx:idx + bs] = feats.astype(np.float32)
                mm.flush()
                all_pred_list.append(pred.cpu().numpy())
                all_lbl_list.append(
                    lbl.numpy() if isinstance(lbl, torch.Tensor) else np.array(lbl)
                )
                idx += bs
            mm.flush()
            del mm

            if idx == 0:
                raise RuntimeError('[JointPCA] 0 samples extracted. Hooks never fired.')

            N        = idx
            all_pred = np.concatenate(all_pred_list)[:N]
            all_lbl  = np.concatenate(all_lbl_list)[:N]
            print(f'[JointPCA] Features: ({N}, {feat_dim}) -> {feat_path}')

        # Score
        feat_mm = np.load(feat_path, mmap_mode='r')[:N]
        scores  = self._mahalanobis_scores(feat_mm)

        np.savez(score_path,
                 scores    = scores.astype(np.float32),
                 labels    = all_lbl,
                 pred      = all_pred,
                 n_samples = N,
                 dataset   = dataset,
                 config_fp = self._config_fp)
        print(f'[JointPCA] Scores saved: {score_path}')

        return all_pred, -scores, all_lbl

    # ------------------------------------------------------------------ #
    # postprocess() -- single-sample / small-batch API                    #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def postprocess(self, net, data: Any):
        net.eval()
        if isinstance(data, dict):
            x = data['data']
        elif isinstance(data, (list, tuple)):
            x = data[0]
        else:
            x = data
        if x.dim() == 3:
            x = x.unsqueeze(0)

        x                = x.to(self.device)
        self.activations = {}
        out              = net(x)
        _, pred          = torch.softmax(out, dim=1).max(dim=1)
        feats            = self._extract_batch(net, x, run_forward=False)
        raw_scores       = self._mahalanobis_scores(feats)
        return pred, torch.from_numpy(-raw_scores).float()

    def set_hyperparam(self, hyperparam: list):
        pass

    def get_hyperparam(self):
        return None

    def __del__(self):
        self._clear_hooks()
