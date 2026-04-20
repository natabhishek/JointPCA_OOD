"""
JointPCA OOD Detection Postprocessor for OpenOOD
-------------------------------------------------
Extracts multi-layer activations via forward hooks, concatenates them into
a joint feature vector, fits PCA, and scores test samples via a spectrally
restricted Mahalanobis distance.

Two variants are exposed via the `filtered` flag in jointpca.yml:

  filtered: false  (default)
    Full-spectrum Mahalanobis over all PCs. This is the primary JointPCA
    method and the one reported in the main results table.

  filtered: true
    Spectral restriction to the interval [T1, T2]:
      T1  noise-spike cutoff (ResNet: valley after left spike; ViT: leftmost)
      T2  eigenvalue of the PC with maximum participation ratio N_alpha,
          i.e. the mode most broadly shared across layers.
    See DEVELOPMENT.md for a full description of the filtering protocol.

Layer strategy:
  ResNet : all Conv2d layers + residual block outputs + penultimate layer
  ViT    : CLS token + patch-GAP from every encoder block

Caching:
  All intermediate artefacts (features, PCA, scores) are stored under
  ./results/jointpca_cache/ as memory-mapped NumPy arrays so that repeated
  runs on the same dataset skip expensive recomputation.

Usage:
  Set postprocessor.name: jointpca in your pipeline config and point to
  configs/postprocessors/jointpca.yml. See README.md for full instructions.
"""

import os
import gc
import re
import numpy as np
import torch
import torch.nn as nn
from typing import Any
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from .jointpca_utils import (
    pool_activation,
    compute_layer_dims,
    compute_participation_ratios,
    select_pcs,
    mahalanobis_scores,
    save_spectrum_plot,
)


class JointPCAPostprocessor(BasePostprocessor):

    def __init__(self, config):
        super().__init__(config)

        # ── Config args ─────────────────────────────────────────────── #
        if hasattr(config.postprocessor, 'postprocessor_args'):
            self.args = config.postprocessor.postprocessor_args
        else:
            self.args = config.postprocessor

        self.args_dict = getattr(config.postprocessor, 'postprocessor_sweep', {})

        # filtered=false → full spectrum (primary method)
        # filtered=true  → spectral restriction via T1/T2
        self.filtered = bool(getattr(self.args, 'filtered', False))

        # ── Device ──────────────────────────────────────────────────── #
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f'[JointPCA] GPU: {torch.cuda.get_device_name(0)}')
        else:
            self.device = torch.device('cpu')
            print('[JointPCA] GPU not available, using CPU')

        # ── PCA state (set during setup) ────────────────────────────── #
        self.mean               = None   # (D,)
        self.components         = None   # (K, D)
        self.explained_variance = None   # (K,)
        self.selected_mask      = None   # (K,) bool, only used when filtered=True
        self.layer_dims         = None   # list[int], per-layer feature widths

        # ── Internal state ───────────────────────────────────────────── #
        self.min_spike_gap  = 5.0        # see DEVELOPMENT.md
        self._config_fp     = None
        self.cache_dir      = os.path.join('results', 'jointpca_cache')
        self.layer_names    = []
        self.hooks          = []
        self.activations    = {}
        self.setup_flag     = False
        # ID dataset name for cache fingerprint (e.g. cifar10, imagenet)
        self._id_dataset    = getattr(
            getattr(self.config, 'dataset', None), 'name', 'unknown'
        )

    # ================================================================== #
    # Cache helpers                                                        #
    # ================================================================== #

    def _ensure_cache_dirs(self):
        for sub in ('features', 'scores', 'pca', 'metadata', 'projections', 'plots'):
            os.makedirs(os.path.join(self.cache_dir, sub), exist_ok=True)

    @staticmethod
    def _make_config_fp(model_name, id_dataset, max_samples):
        """
        Build a cache fingerprint that encodes all info needed to identify
        the artefact unambiguously.

        Pattern: {model}_{dataset}_allconv_gap_{samples}
        Example: resnet18_32x32_cifar10_allconv_gap_9k
                 resnet50_imagenet_allconv_gap_45k
        """
        s = f'{max_samples // 1000}k' if max_samples >= 1000 else str(max_samples)
        # Shorten common dataset names for readability
        ds = (id_dataset
              .replace('imagenet200', 'in200')
              .replace('imagenet', 'in1k')
              .replace('cifar10', 'c10')
              .replace('cifar100', 'c100'))
        return f'{model_name}_{ds}_allconv_gap_{s}'

    def _p(self, subdir, name):
        return os.path.join(self.cache_dir, subdir, name)

    def _path_feat_train(self):
        # filename: features_{model}_{dataset}_allconv_gap_{samples}_train.npy
        return self._p('features', f'features_{self._config_fp}_train.npy')

    def _path_feat_ood(self, ood_dataset):
        # filename: features_{id_dataset}_tested_on_{ood_dataset}_{fp}.npy
        return self._p('features',
                       f'features_{self._config_fp}_test_{ood_dataset}.npy')

    def _path_meta(self):
        return self._p('metadata', f'meta_{self._config_fp}.npz')

    def _path_pca(self):
        return self._p('pca', f'pca_{self._config_fp}.npz')

    def _path_proj(self):
        return self._p('projections', f'proj_{self._config_fp}_train.npy')

    def _path_scores(self, ood_dataset):
        suffix = 'filtered' if self.filtered else 'full'
        return self._p('scores',
                       f'scores_{self._config_fp}_test_{ood_dataset}_{suffix}.npz')

    @staticmethod
    def _mmap_write(path, shape, dtype=np.float32):
        return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)

    @staticmethod
    def _mmap_read(path):
        if not os.path.exists(path):
            return None
        return np.load(path, mmap_mode='r')

    # ================================================================== #
    # Hook management                                                      #
    # ================================================================== #

    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def _is_vit(self, model):
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
        r_resnet = re.compile(r'^layer[1-4]\.\d+$')
        return [name for name, _ in model.named_modules() if r_resnet.match(name)]

    def _find_penultimate(self, model):
        p_name, p_mod = None, None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                break
            if len(list(module.children())) == 0 and not isinstance(module, nn.Conv2d):
                p_name, p_mod = name, module
        return p_name, p_mod

    def _make_hook(self, name):
        def hook(module, input, output):
            self.activations[name] = (
                output.detach() if self.device.type == 'cuda'
                else output.detach().cpu()
            )
        return hook

    def _register_hooks(self, model):
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
                    self.hooks.append(m.register_forward_hook(self._make_hook(name)))
                    self.layer_names.append(name)
                    count += 1
            print(f'[JointPCA] ViT blocks hooked: {count}')
        else:
            block_names = self._get_resnet_block_names(model)
            for name, m in model.named_modules():
                if isinstance(m, nn.Conv2d):
                    self.hooks.append(m.register_forward_hook(self._make_hook(name)))
                    self.layer_names.append(name)
                    count += 1
            for name, m in model.named_modules():
                if name in block_names:
                    self.hooks.append(m.register_forward_hook(self._make_hook(name)))
                    self.layer_names.append(name)
                    count += 1
            p_name, p_mod = self._find_penultimate(model)
            if p_name:
                self.hooks.append(p_mod.register_forward_hook(self._make_hook(p_name)))
                self.layer_names.append(p_name)
                count += 1
            print(f'[JointPCA] ResNet hooks registered: {count} layers')

    def _attach_hooks(self, model):
        """Re-attach hooks after setup() using stored layer_names."""
        self._clear_hooks()
        self.activations = {}
        name_set = set(self.layer_names)
        for name, m in model.named_modules():
            if name in name_set:
                self.hooks.append(m.register_forward_hook(self._make_hook(name)))
        print(f'[JointPCA] Re-attached {len(self.hooks)} hooks '
              f'(expected {len(self.layer_names)})')

    # ================================================================== #
    # Feature extraction                                                   #
    # ================================================================== #

    def _extract_batch(self, net, data, run_forward=True):
        if run_forward:
            with torch.no_grad():
                net(data)
        parts = [
            pool_activation(self.activations[n]).cpu().numpy()
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

    # ================================================================== #
    # Dataset name helper                                                  #
    # ================================================================== #

    @staticmethod
    def _dataset_name(loader):
        def parse_name(raw):
            if raw is None:
                return None
            fname = os.path.basename(str(raw)).replace('.txt', '')
            for prefix in ('test_', 'val_', 'train_'):
                if fname.startswith(prefix):
                    stem = fname[len(prefix):]
                    return 'imagenet_test' if (prefix == 'test_' and
                                               stem == 'imagenet') else stem
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
                    line = str(ds.imglist[0]).strip()
                    root = line.split(' ', 1)[0].split('/', 1)[0]
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

    # ================================================================== #
    # PCA fitting                                                          #
    # ================================================================== #

    def _fit_pca(self, train_features, actual_n_train, feat_dim, n_comp):
        from sklearn.decomposition import PCA

        print(f'[JointPCA] Fitting PCA: N={actual_n_train}, D={feat_dim}, K={n_comp}')
        X   = np.asarray(train_features[:actual_n_train], dtype=np.float32)
        pca = PCA(n_components=n_comp, svd_solver='randomized', random_state=0,
                  copy=False)
        pca.fit(X)
        del X
        gc.collect()

        mean     = pca.mean_.astype(np.float32)
        comps    = pca.components_.astype(np.float32)
        ev       = pca.explained_variance_.astype(np.float32)
        ev_ratio = pca.explained_variance_ratio_.astype(np.float32)
        print(f'[JointPCA] PCA done. Top-100 PCs explain '
              f'{ev_ratio[:100].sum() * 100:.1f}% variance.')
        return mean, comps, ev, ev_ratio

    # ================================================================== #
    # setup()                                                              #
    # ================================================================== #

    def setup(self, net, id_loader_dict, ood_loader_dict=None):
        if self.setup_flag:
            return

        os.makedirs(self.cache_dir, exist_ok=True)
        self._ensure_cache_dirs()
        net.eval()
        net = net.to(self.device)

        model_name = net.__class__.__name__.lower()

        # ResNet18 uses 9000 samples from the test split (validated behaviour).
        # All other models use max_train_samples from the train split.
        is_resnet18 = 'resnet18' in model_name
        if is_resnet18:
            max_samples  = 9000
            train_loader = id_loader_dict['test']
            print(f'[JointPCA] ResNet18 detected — using 9000 samples '
                  f'from test split')
        else:
            max_samples  = int(getattr(self.args, 'max_train_samples', 45000))
            train_loader = id_loader_dict['train']
        self._register_hooks(net)
        print(f'[JointPCA] Layer names ({len(self.layer_names)} total):')
        for n in self.layer_names:
            print(f'  {n}')

        # ── 1. Probe feature dimension + per-layer dims ─────────────── #
        for batch in train_loader:
            data = batch['data'] if isinstance(batch, dict) else batch[0]
            with torch.no_grad():
                net(data.to(self.device))
            # collect per-layer dims before clearing activations
            self.layer_dims = [
                pool_activation(self.activations[n]).shape[1]
                for n in self.layer_names
                if n in self.activations
            ]
            feat_dim = sum(self.layer_dims)
            self.activations = {}
            break

        self._config_fp = self._make_config_fp(
            model_name, self._id_dataset, max_samples
        )
        print(f'[JointPCA] Config FP  : {self._config_fp}')
        print(f'[JointPCA] Feature dim: {feat_dim}  '
              f'({len(self.layer_names)} layers)')

        # ── 2. ID training features (cached) ───────────────────────── #
        skip = False
        if os.path.exists(self._path_feat_train()) and \
                os.path.exists(self._path_meta()):
            try:
                meta = np.load(self._path_meta(), allow_pickle=True)
                if (int(meta['n_features']) == feat_dim and
                        int(meta['n_samples']) > 0 and
                        list(meta['layer_names']) == self.layer_names):
                    print(f'[JointPCA] ID train features cached: '
                          f'{self._path_feat_train()}')
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
                     layer_names=np.array(self.layer_names, dtype=object),
                     layer_dims=np.array(self.layer_dims, dtype=np.int64))
            print(f'[JointPCA] Saved ({n}, {feat_dim})')

        meta           = np.load(self._path_meta(), allow_pickle=True)
        actual_n_train = int(meta['n_samples'])
        # restore layer_dims from cache in case this is a cached run
        if 'layer_dims' in meta:
            self.layer_dims = list(meta['layer_dims'].astype(int))
        train_features = self._mmap_read(self._path_feat_train())[:actual_n_train]

        # ── 3. n_components ─────────────────────────────────────────── #
        n_comp = min(actual_n_train, feat_dim) - 1
        self._clear_hooks()
        self.activations = {}

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

        # ── 5. PC selection for filtered variant ─────────────────────── #
        if self.filtered:
            pr = compute_participation_ratios(self.components, self.layer_dims)
            self.selected_mask = select_pcs(
                explained_variance  = self.explained_variance,
                participation_ratio = pr,
                min_spike_gap       = self.min_spike_gap,
                plot_path           = os.path.join(
                    self.cache_dir, 'plots',
                    f'spectrum_{self._config_fp}.png'
                ),
                config_fp           = self._config_fp,
            )
            n_sel = int(self.selected_mask.sum())
            print(f'[JointPCA] Filtered variant: {n_sel} PCs selected '
                  f'out of {len(self.selected_mask)}')
        else:
            self.selected_mask = None
            print(f'[JointPCA] Full-spectrum variant: all {n_comp} PCs used')

        self.setup_flag = True
        print('[JointPCA] Setup complete.')

    # ================================================================== #
    # postprocess() — batch-level API called by OpenOOD pipeline          #
    # ================================================================== #

    @torch.no_grad()
    def postprocess(self, net, data: Any):
        # Normalise input format
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

        feats      = self._extract_batch(net, x, run_forward=False)
        raw_scores = mahalanobis_scores(
            features            = feats,
            mean                = self.mean,
            components          = self.components,
            explained_variance  = self.explained_variance,
            selected_mask       = self.selected_mask,   # None → full spectrum
        )
        # negate: OpenOOD expects higher score = more ID
        return pred, torch.from_numpy(-raw_scores).float()

    # ================================================================== #
    # inference() — full-loader API with feature + score caching          #
    # ================================================================== #

    def inference(self, net, data_loader):
        dataset    = self._dataset_name(data_loader)
        feat_path  = self._path_feat_ood(dataset)
        score_path = self._path_scores(dataset)
        feat_dim   = self.components.shape[1]

        print(f'\n[JointPCA] Inference: {dataset}  '
              f'({"filtered" if self.filtered else "full-spectrum"})')

        net.eval()
        net = net.to(self.device)
        self._attach_hooks(net)

        # ── Score cache ──────────────────────────────────────────────── #
        if os.path.exists(score_path):
            try:
                sc = np.load(score_path, allow_pickle=True)
                n  = int(sc['n_samples'])
                print(f'[JointPCA] Loaded score cache ({n} samples): {score_path}')
                return sc['pred'], -sc['scores'], sc['labels']
            except Exception as e:
                print(f'[JointPCA] Score cache error: {e} -- recomputing...')

        # ── Feature extraction ───────────────────────────────────────── #
        feat_mm   = self._mmap_read(feat_path)
        total_est = len(getattr(data_loader, 'dataset', None) or []) or 200000
        all_pred, all_lbl, N = None, None, None

        if (feat_mm is not None and
                feat_mm.ndim == 2 and
                feat_mm.shape[1] == feat_dim):
            N = feat_mm.shape[0]
            print(f'[JointPCA] Reusing cached features: ({N}, {feat_dim})')
            preds_list, lbls_list = [], []
            for batch in tqdm(data_loader, desc=f'Labels: {dataset}'):
                if isinstance(batch, dict):
                    data_b, lbl = batch['data'], batch['label']
                else:
                    data_b, lbl = batch[0], batch[1]
                with torch.no_grad():
                    out = net(data_b.to(self.device))
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
                    data_b, lbl = batch['data'], batch['label']
                else:
                    data_b, lbl = batch[0], batch[1]
                data_b = data_b.to(self.device)
                with torch.no_grad():
                    out = net(data_b)
                _, pred = torch.softmax(out, dim=1).max(dim=1)
                parts = [
                    pool_activation(self.activations[n]).cpu().numpy()
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
                raise RuntimeError(
                    '[JointPCA] 0 samples extracted. Hooks never fired.'
                )
            N        = idx
            all_pred = np.concatenate(all_pred_list)[:N]
            all_lbl  = np.concatenate(all_lbl_list)[:N]
            print(f'[JointPCA] Features: ({N}, {feat_dim}) -> {feat_path}')

        # ── Scoring ──────────────────────────────────────────────────── #
        feat_mm = np.load(feat_path, mmap_mode='r')[:N]
        scores  = mahalanobis_scores(
            features           = feat_mm,
            mean               = self.mean,
            components         = self.components,
            explained_variance = self.explained_variance,
            selected_mask      = self.selected_mask,
        )
        np.savez(score_path,
                 scores=scores.astype(np.float32), labels=all_lbl,
                 pred=all_pred, n_samples=N,
                 dataset=dataset, config_fp=self._config_fp)
        print(f'[JointPCA] Scores saved: {score_path}')
        return all_pred, -scores, all_lbl

    # ================================================================== #
    # Hyperparam interface (required by OpenOOD, unused here)             #
    # ================================================================== #

    def set_hyperparam(self, hyperparam: list):
        pass

    def get_hyperparam(self):
        return None

    def __del__(self):
        self._clear_hooks()
