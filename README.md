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
cp jointpca_postprocessor.py openood/postprocessors/   # postprocessor class
cp jointpca_utils.py         openood/postprocessors/   # pooling, scoring, PC selection
cp jointpca.yml              configs/postprocessors/   # pipeline config
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

All experiments were run on a Mac Mini M4 Pro with 24 GB unified memory, without a GPU. The largest configuration (ResNet-50, 45 000 training samples) produces a feature matrix of shape $45000 \times 43712$. Sklearn's randomised SVD operates on the Gram matrix $A^\top A$ of shape $43712 \times 43712$ rather than forming the full covariance, keeping memory tractable. Further runtime details are in the paper.

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

All intermediate artefacts (features, PCA, projections, scores) are stored under `results/jointpca_cache/` as memory-mapped NumPy arrays. The first run performs feature extraction and PCA fitting, which is the expensive step. On all subsequent runs, cached artefacts are detected automatically and loaded from disk — setup completes in seconds regardless of dataset size. The config fingerprint encodes the model class name and `max_train_samples`; changing either triggers fresh extraction and fitting. Delete individual subdirectories to force recomputation of a specific stage.

```
results/jointpca_cache/
  features/     ID training and OOD test features
  pca/          mean, components, eigenvalues
  scores/       per-sample scores (full and filtered variants stored separately)
  metadata/     n_samples, n_features, layer_names, layer_dims
  plots/        eigenvalue spectrum with T1/T2 marked (filtered variant only)
```
