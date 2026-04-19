# JointPCA — Out-of-Distribution Detection via Joint Multi-Layer PCA

This repository contains the official implementation of **JointPCA**, submitted to NeurIPS 2026 (anonymous review).

JointPCA is integrated directly into the [OpenOOD v1.5](https://github.com/Jingkang50/OpenOOD) benchmark framework, ensuring fully standardised and reproducible evaluation against all existing baselines.

---

## Results

Evaluated using OpenOOD v1.5's standardised pipeline. All scores are AUROC (↑).

| Backbone / ID dataset | Setting | All PCs | Filtered | Best OpenOOD baseline | OpenOOD method |
|---|---|---|---|---|---|
| ResNet-50 / ImageNet-1K | Near-OOD | 99.95 | **100** | 95.22 | CombOOD |
| ResNet-50 / ImageNet-1K | Far-OOD | 99.98 | **100** | 97.55 | AdaSCALE-A |
| ViT / ImageNet-1K | Near-OOD | 99.22 | **99.78** | 81.71 | RMDS++ |
| ViT / ImageNet-1K | Far-OOD | 99.77 | **99.82** | 93.65 | MDS++ |
| ResNet-18 / ImageNet-200 | Near-OOD | **100** | **100** | 95.74 | CombOOD |
| ResNet-18 / ImageNet-200 | Far-OOD | **100** | **100** | 95.01 | ASH |
| ResNet-18 / CIFAR-100 | Near-OOD | **100** | **100** | 88.30 | MSP |
| ResNet-18 / CIFAR-100 | Far-OOD | **100** | **100** | 91.12 | MDS |
| ResNet-18 / CIFAR-10 | Near-OOD | **100** | **100** | 94.86 | RotPred |
| ResNet-18 / CIFAR-10 | Far-OOD | **100** | **100** | 98.18 | RotPred |

Near-OOD and Far-OOD dataset splits follow the OpenOOD v1.5 benchmark definitions for each ID dataset. Full details are provided in the paper.

**All PCs** = full-spectrum Mahalanobis (primary method, no hyperparameters).  
**Filtered** = spectral restriction to [T1, T2] using the participation ratio criterion (see DEVELOPMENT.md).

---

## Method

**Feature extraction.** Forward hooks capture activations from multiple layers simultaneously. For ResNet, hooks are placed on all Conv2d layers, residual block outputs, and the penultimate layer; spatial maps are pooled via global average pooling. For ViT, hooks capture every encoder block output, decomposed into the CLS token and a patch-GAP vector. All per-layer vectors are concatenated into a single joint feature `z(x)`.

**PCA.** Sklearn's randomised PCA is fit on up to `max_train_samples` ID training features. The full rank `K = min(N, D) - 1` is used. The fit is cached to disk after the first run.

**Scoring.** A spectrally restricted Mahalanobis distance:

$$d^2(x) = \sum_{\alpha \in I} \frac{q_\alpha(x)^2}{\lambda_\alpha}, \qquad q_\alpha(x) = u_\alpha^\top (z(x) - \mu)$$

where $u_\alpha$, $\lambda_\alpha$ are PCA eigenvectors and eigenvalues, and $I$ is the selected spectral interval.

**Primary variant (All PCs).** $I$ is the full spectrum. This is the main JointPCA method. No hyperparameters, no tuning.

**Filtered variant.** $I = [T_1, T_2]$ where $T_1$ excludes noisy low-variance components (ResNet only) and $T_2$ is the eigenvalue of the PC with maximum *participation ratio* $\mathcal{N}_\alpha$ — the mode most broadly shared across layers. See DEVELOPMENT.md for the full protocol.

---

## Setup

### 1. Install OpenOOD

```bash
git clone https://github.com/Jingkang50/OpenOOD.git
cd OpenOOD
pip install -e .
```

**Datasets.** Follow the OpenOOD README to download benchmark datasets using the scripts in `scripts/download/`.

**Checkpoints.** OpenOOD provides pre-trained checkpoints via `scripts/download/`. You can also use torchvision or timm weights — any checkpoint trained on the ID dataset works. Set the path in your network config yml or pass `--network.checkpoint <path>` on the command line.

### 2. Copy the postprocessor files

```bash
cp jointpca_postprocessor.py openood/postprocessors/
cp jointpca_utils.py         openood/postprocessors/
cp jointpca.yml              configs/postprocessors/
```

### 3. Register JointPCA

Open `openood/postprocessors/__init__.py` and add two lines — one import at the top of the file, and one entry in the postprocessors dictionary:

```python
# At the top with the other imports:
from .jointpca_postprocessor import JointPCAPostprocessor

# Inside postprocessors_dict, alongside the existing entries:
'jointpca': JointPCAPostprocessor,
```

---

## Running

JointPCA uses the standard OpenOOD `main.py` entry point.

**ResNet-50 on ImageNet-1K — full spectrum (primary method)**

```bash
python main.py \
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

**ResNet-18 on CIFAR-10**

```bash
python main.py \
  --config \
    configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

**ResNet-18 on CIFAR-100 / ImageNet-200**

Replace the dataset and network configs accordingly (e.g. `cifar100.yml`, `imagenet200.yml`).

**ViT on ImageNet-1K**

```bash
python main.py \
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/vit_b16.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

**Filtered variant**

Set `filtered: true` in `configs/postprocessors/jointpca.yml` before running. A spectrum plot will be saved to `results/jointpca_cache/plots/` for verification. See DEVELOPMENT.md.

---

## Compute

The PCA fitting step is the most memory-intensive part of the pipeline. For context, all experiments were run on a Mac Mini M4 Pro with 24 GB unified memory, without a GPU — the full pipeline ran successfully on CPU alone. GPU acceleration is supported when available and will speed up feature extraction significantly. Further details on runtime and memory usage are provided in the paper.

The only parameter exposed in `jointpca.yml` is `max_train_samples` (default 45 000). Reduce this if memory is limited.

---

## Dependencies

All dependencies are already required by OpenOOD:

| Package | Purpose |
|---|---|
| `torch` | Model inference and hook API |
| `scikit-learn` | Randomised PCA |
| `numpy` | Memory-mapped feature storage |
| `scipy` | Peak detection (filtered variant only) |
| `tqdm` | Progress bars |
| `matplotlib` | Spectrum plot (filtered variant only) |

---

## Cache

All intermediate artefacts (features, PCA, projections, scores) are stored under `results/jointpca_cache/` as memory-mapped NumPy arrays. The config fingerprint encodes the model class name and `max_train_samples` — changing either triggers fresh extraction and fitting. Delete individual subdirectories to force recomputation of a specific stage. Subsequent runs load from cache and complete setup in seconds.

```
results/jointpca_cache/
  features/     ID training and OOD test features
  pca/          mean, components, eigenvalues
  scores/       per-sample scores (full and filtered variants stored separately)
  metadata/     n_samples, n_features, layer_names, layer_dims
  plots/        eigenvalue spectrum with T1/T2 marked (filtered variant only)
```
**Scoring.** A spectrally restricted Mahalanobis distance:

$$d^2(x) = \sum_{\alpha \in I} \frac{q_\alpha(x)^2}{\lambda_\alpha}, \qquad q_\alpha(x) = u_\alpha^\top (z(x) - \mu)$$

where $u_\alpha$, $\lambda_\alpha$ are PCA eigenvectors and eigenvalues, and $I$ is the selected spectral interval.

**Primary variant (All PCs).** $I$ is the full spectrum. This is the main JointPCA method. No hyperparameters, no tuning.

**Filtered variant.** $I = [T_1, T_2]$ where $T_1$ excludes noisy low-variance components (ResNet only) and $T_2$ is the eigenvalue of the PC with maximum *participation ratio* $\mathcal{N}_\alpha$ — the mode most broadly shared across layers. See DEVELOPMENT.md for the full protocol.

---

## Repository structure

The only files you need from this repository are:

```
jointpca_postprocessor.py   — postprocessor class
jointpca_utils.py           — pure utility functions (pooling, scoring, PC selection)
jointpca.yml                — pipeline config
```

One existing OpenOOD file is modified (two lines added):

```
openood/postprocessors/__init__.py
```

---

## Setup

### 1. Install OpenOOD

```bash
git clone https://github.com/Jingkang50/OpenOOD.git
cd OpenOOD
pip install -e .
```

**Datasets.** Follow the OpenOOD README to download benchmark datasets using the scripts in `scripts/download/`.

**Checkpoints.** OpenOOD provides pre-trained checkpoints via `scripts/download/`. You can also use torchvision or timm weights — any checkpoint trained on the ID dataset works. Set the path in your network config yml or pass `--network.checkpoint <path>` on the command line.

### 2. Copy the postprocessor files

```bash
cp jointpca_postprocessor.py openood/postprocessors/
cp jointpca_utils.py         openood/postprocessors/
cp jointpca.yml              configs/postprocessors/
```

### 3. Register JointPCA

Open `openood/postprocessors/__init__.py` and add two lines:

```python
# At the top with the other imports:
from .jointpca_postprocessor import JointPCAPostprocessor

# Inside the postprocessors_dict:
'jointpca': JointPCAPostprocessor,
```

---

## Running

JointPCA uses the standard OpenOOD `main.py` entry point.

**ResNet-50 on ImageNet-1K — full spectrum (primary method)**

```bash
python main.py \
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

**ResNet-18 on CIFAR-10**

```bash
python main.py \
  --config \
    configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

**ResNet-18 on CIFAR-100 / ImageNet-200**

Replace the dataset and network configs accordingly (e.g. `cifar100.yml`, `imagenet200.yml`).

**ViT on ImageNet-1K**

```bash
python main.py \
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/vit_b16.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

**Filtered variant**

Set `filtered: true` in `configs/postprocessors/jointpca.yml` before running. A spectrum plot will be saved to `results/jointpca_cache/plots/` for verification. See DEVELOPMENT.md.

---

## Expected console output

```
[JointPCA] GPU: NVIDIA A100-SXM4-80GB
[JointPCA] ResNet hooks registered: 54 layers
[JointPCA] Feature dim: 102400  (54 layers)
[JointPCA] Fitting PCA: N=45000, D=102400, K=44999
[JointPCA] PCA done. Top-100 PCs explain 61.3% variance.
[JointPCA] Full-spectrum variant: all 44999 PCs used
[JointPCA] Setup complete.
```

On subsequent runs, cached artefacts load in seconds:

```
[JointPCA] ID train features cached: results/jointpca_cache/features/features_train_resnet50_45k.npy
[JointPCA] PCA cached: results/jointpca_cache/pca/pca_resnet50_45k.npz
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
| `scipy` | Peak detection (filtered variant only) |
| `tqdm` | Progress bars |
| `matplotlib` | Spectrum plot (filtered variant only) |

---

## Cache layout

All artefacts are stored under `results/jointpca_cache/`. The config fingerprint `{fp}` encodes the model class name and `max_train_samples`. Changing either value triggers fresh extraction and PCA fitting. Delete individual subdirectories to force recomputation of a specific stage.

```
results/jointpca_cache/
  features/
    features_train_{fp}.npy          ID training features   (N, D)
    features_{dataset}_{fp}.npy      OOD / test features    (N, D)
  pca/
    pca_{fp}.npz                     mean, components, eigenvalues
  scores/
    scores_{dataset}_{fp}_full.npy       scores — full-spectrum variant
    scores_{dataset}_{fp}_filtered.npy   scores — filtered variant
  metadata/
    metadata_{fp}.npz                n_samples, n_features, layer_names, layer_dims
  plots/
    spectrum_{fp}.png                eigenvalue spectrum with T1/T2 (filtered only)
```pip install -e .
```

**Datasets.** The OpenOOD README describes how to download benchmark datasets using the scripts in the `scripts/download/` folder of the repository. Follow the instructions there for ImageNet-1K and the associated OOD splits.

**Model checkpoints.** OpenOOD provides pre-trained checkpoints downloadable via the scripts in `scripts/download/`. That said, checkpoints are flexible — you are not required to use OpenOOD's own weights. ResNet-50 weights can be loaded from torchvision (e.g. `torchvision.models.resnet50(pretrained=True)`), and ViT weights can be obtained from Hugging Face or timm. What matters is that the model was trained on your ID dataset (ImageNet-1K in our case) and that the checkpoint path is correctly set in your network config yml (e.g. `configs/networks/resnet50.yml`) or passed via `--network.checkpoint` on the command line.

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
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

**ViT on ImageNet-1K**

```bash
python main.py \
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/vit_b16.yml \
    configs/pipelines/test/test_ood.yml \
    configs/postprocessors/jointpca.yml \
    configs/preprocessors/base_preprocessor.yml
```

The only parameter exposed in `jointpca.yml` is `max_train_samples`. The default is 45 000; reduce it if memory is limited.

```yaml
postprocessor_args:
  max_train_samples: 45000   # adjust as needed
```

**Checkpoint names.** The checkpoint path is read from your network config yml (e.g. `configs/networks/resnet50.yml`). You can also pass it on the command line with `--network.checkpoint results/checkpoints/your_filename.ckpt` — this simply overrides the yml and does no harm. Either way, the filename must match exactly what is on disk; OpenOOD does not search for partial matches, so a wrong name will cause a clean failure.

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
