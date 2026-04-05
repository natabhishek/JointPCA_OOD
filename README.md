# JointPCA — OOD Detection via Multi-Layer PCA

This repository contains the implementation of **JointPCA**, an out-of-distribution (OOD) detection postprocessor designed to plug into the [OpenOOD](https://github.com/Jingkang50/OpenOOD) benchmark framework.

Supported backbones: **ResNet-50** and **ViT** .

## Files

| File | Description |
|---|---|
| `jointpca_postprocessor.py` | Main postprocessor class |
| `jointpca.yml` | Config file for the OpenOOD pipeline |
| `README.md` | This file |

---

## Method

**Feature extraction.** Forward hooks capture activations from multiple layers simultaneously. For ResNet-50, hooks are placed on all `Conv2d` layers, residual block outputs, and the penultimate layer; spatial maps are reduced via global average pooling. For ViT, hooks capture every encoder block output, split into the CLS token and a patch-GAP vector. All per-layer vectors are concatenated into a single joint feature.

**PCA.** Sklearn's randomised PCA is fit on up to `max_train_samples` ID training features using the full rank $K = \min(N, D)$. The fit is cached to disk after the first run.

**PC selection (clip-to-mean).** Not all PCs are informative — the full spectrum contains both noisy components and diminishing-return signal. We automatically select a meaningful band using the log₁₀ eigenvalue spectrum. The left boundary **T1** excludes noise: for ResNet-50, the spectrum has a pronounced spike of low-variance noisy PCs on the far left, well separated from the signal; T1 is placed at the valley just after this spike, discarding those noisy components. For ViT no such spike exists, so T1 is simply the leftmost PC — nothing is excluded on the left. The right boundary **T2** is the mean of log₁₀(λ) over all PCs from T1 onwards, clipping away the long tail of very high-variance components that add noise without discriminative value. The selected PCs are those whose eigenvalue falls in the band T1 ≤ log₁₀(λ) ≤ T2. See *Verifying T1/T2 placement* in the Cache layout section for how to visually confirm the boundaries are correct.

**Scoring.** Mahalanobis distance computed on the selected PCs only:

$$\text{score}(z) = \sum_{i \in \text{selected}} \frac{y_i^2}{\lambda_i}, \quad y_i = u_i^\top (z - \mu)$$

Higher score = more OOD.

---

## Setup

### 1. Install OpenOOD

```bash
git clone https://github.com/Jingkang50/OpenOOD.git
cd OpenOOD
pip install -e .
```

Follow the [OpenOOD README](https://github.com/Jingkang50/OpenOOD) to download the relevant benchmark datasets and obtain pre-trained checkpoints for **ResNet-50** and **ViT** on your target ID dataset (e.g. ImageNet-1K).

### 2. Copy the postprocessor

```bash
cp jointpca_postprocessor.py openood/postprocessors/jointpca_postprocessor.py
```

### 3. Register JointPCA in OpenOOD

Open `openood/postprocessors/__init__.py` and add two lines:

```python
# With the other imports at the top:
from .jointpca_postprocessor import JointPCAPostprocessor

# Inside the postprocessor dictionary (look for postprocessors_dict or similar):
'jointpca': JointPCAPostprocessor,
```

The dictionary block looks like this — add the `jointpca` entry alongside the existing ones:

```python
postprocessors_dict = {
    'msp': MSPPostprocessor,
    'odin': ODINPostprocessor,
    # ... other methods ...
    'jointpca': JointPCAPostprocessor,   # <-- add this line
}
```

### 4. Copy the config

```bash
cp jointpca.yml configs/postprocessors/jointpca.yml
```

---

## Running

JointPCA uses the standard OpenOOD `main.py` entry point. Combine your dataset config, network config, pipeline config, and the JointPCA postprocessor config:

**ResNet-50 on ImageNet-1K**

```bash
python main.py \
  --config configs/datasets/imagenet/imagenet.yml \
            configs/datasets/imagenet/imagenet_ood.yml \
            configs/networks/resnet50.yml \
            configs/pipelines/test/test_ood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/jointpca.yml \
  --network.checkpoint path/to/resnet50_imagenet1k.ckpt \
  --network.pretrained True
```

**ViT on ImageNet-1K**

```bash
python main.py \
  --config configs/datasets/imagenet/imagenet.yml \
            configs/datasets/imagenet/imagenet_ood.yml \
            configs/networks/vit_b16.yml \
            configs/pipelines/test/test_ood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/jointpca.yml \
  --network.checkpoint path/to/vit_b16_imagenet1k.ckpt \
  --network.pretrained True
```

The only parameter exposed in `jointpca.yml` is `max_train_samples`. The default is 45 000; reduce it if memory is limited.

```yaml
postprocessor_args:
  max_train_samples: 45000   # adjust as needed
```

---

## Expected console output

During `setup()`, JointPCA prints the complete list of hooked layer names so you can verify the correct layers are captured:

```
[JointPCA] CNN hooks registered: 54 layers
[JointPCA] Layer names (54 total):
  conv1
  layer1.0.conv1
  layer1.0.conv2
  layer1.0.conv3
  layer1.0
  ...
  layer4.2.conv3
  layer4.2
  avgpool
[JointPCA] Feature dim: 102400
[JointPCA] Fitting PCA: N=45000, D=102400, K=45000
[JointPCA] PCA done. Top-100 PCs explain 61.3% variance.
[JointPCA] Setup complete.
```

On subsequent runs, cached artefacts are loaded and setup completes in seconds:

```
[JointPCA] ID train features cached: ./jointpca_cache/features/features_train_resnet50_45k.npy
[JointPCA] PCA cached: ./jointpca_cache/pca/pca_resnet50_45k.npz
[JointPCA] Projections cached: ./jointpca_cache/projections/projections_train_resnet50_45k.npy
[JointPCA] Setup complete.
```

---

## Dependencies

All dependencies are already required by OpenOOD:

| Package | Purpose |
|---|---|
| `torch` | Model inference and hook API |
| `scikit-learn` | Randomised PCA |
| `numpy` | Memory-mapped feature storage |
| `tqdm` | Progress bars |

---

## Cache layout

All intermediate artefacts are stored as memory-mapped NumPy arrays under `./jointpca_cache/`:

**Verifying T1/T2 placement.** Every time `setup()` runs, the spectrum plot is saved automatically to `plots/`. The default `min_spike_gap = 5.0` works correctly for both ResNet-50 (skips the noise spike on the far left) and ViT (no spike present, so T1 falls back to leftmost automatically) — try it as-is first. If the plot shows T1 cutting into signal PCs or missing the spike, adjust `self.min_spike_gap` in `__init__`. A larger value makes spike detection stricter; a smaller value makes it more permissive.

Open the saved plot and check:
- Red T1 line: sits at the valley between noise spike and signal (ResNet-50), or at the leftmost eigenvalue (ViT).
- Blue dashed T2 line: sits near the centre of the signal distribution.
- Blue shaded region: a reasonable signal band between T1 and T2.

```
jointpca_cache/
  features/
    features_train_{config_fp}.npy           # ID training features  (N, D)
    features_{dataset}_{config_fp}.npy       # OOD / test features   (N, D)
  pca/
    pca_{config_fp}.npz                      # mean, components, eigenvalues
  projections/
    projections_train_{config_fp}.npy        # train projections      (N, K)
  scores/
    scores_{dataset}_{config_fp}_clipmd.npz  # per-sample scores
  metadata/
    metadata_{config_fp}.npz                 # n_samples, n_features, layer_names
  plots/
    spectrum_{config_fp}.png                 # eigenvalue spectrum with T1/T2 marked
```

The config fingerprint `{config_fp}` encodes the model class name and `max_train_samples`. Changing either value automatically triggers a fresh extraction and PCA fit. Delete individual cache files to force recomputation of a specific stage.
