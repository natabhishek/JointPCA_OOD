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

Follow the OpenOOD README to download benchmark datasets and pre-trained checkpoints using the scripts in `scripts/download/`. For ViT on ImageNet-1K, download a ViT-B/16 checkpoint pretrained on ImageNet-1K separately (e.g. from [timm](https://github.com/huggingface/pytorch-image-models) or Hugging Face) and place it in `results/checkpoints/`.

### 2. Copy the postprocessor files

```bash
cp jointpca_postprocessor.py openood/postprocessors/   # postprocessor class
cp jointpca_utils.py         openood/postprocessors/   # pooling, scoring, PC selection
cp jointpca.yml              configs/postprocessors/   # pipeline config
```

### 3. Register JointPCA

In `openood/postprocessors/__init__.py`, add the import at the top alongside the other imports:

```python
from .jointpca_postprocessor import JointPCAPostprocessor
```

In `openood/postprocessors/utils.py`, add the import at the top and one entry inside the `postprocessors` dictionary:

```python
# At the top with the other imports:
from .jointpca_postprocessor import JointPCAPostprocessor

# Inside the postprocessors dictionary:
'jointpca': JointPCAPostprocessor,
```

---

## Running

JointPCA uses the standard OpenOOD `main.py` entry point.

**CPU-only machines.** Add `map_location='cpu'` to the three `torch.load` calls in `openood/networks/utils.py`:

```python
# 1. Inside the dict/list checkpoint loop:
subnet.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=False)

# 2. Main load block:
net.load_state_dict(torch.load(network_config.checkpoint, map_location='cpu'), strict=False)

# 3. Retry block after RuntimeError:
loaded_pth = torch.load(network_config.checkpoint, map_location='cpu')
```

The commands below use the standard OpenOOD v1.5 checkpoints. Download them first:

```bash
python scripts/download/download.py --contents checkpoints --checkpoints ood_v1.5
```

---

**ResNet-18 on CIFAR-10**

```bash
python main.py \
  --config \
    configs/datasets/cifar10/cifar10.yml \
    configs/datasets/cifar10/cifar10_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/jointpca.yml \
  --network.checkpoint results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s1/last_epoch100_acc0.9490.ckpt \
  --num_gpus 0
```

**ResNet-18 on CIFAR-100**

```bash
python main.py \
  --config \
    configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/jointpca.yml \
  --network.checkpoint results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s1/last_epoch100_acc0.7710.ckpt \
  --num_gpus 0
```

**ResNet-18 on ImageNet-200**

```bash
python main.py \
  --config \
    configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/jointpca.yml \
  --network.checkpoint results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s1/last_epoch90_acc0.8530.ckpt \
  --num_gpus 0
```

**ResNet-50 on ImageNet-1K**

Download the torchvision pretrained checkpoint first:

```bash
mkdir -p checkpoints
wget https://download.pytorch.org/models/resnet50-0676ba61.pth -O checkpoints/resnet50-0676ba61.pth
```

The network config (`configs/networks/resnet50.yml`) already points to this path — no `--network.checkpoint` flag needed:

```bash
python main.py \
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/resnet50.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/jointpca.yml \
  --num_gpus 0
```

**ViT on ImageNet-1K**

For fair comparison with the OpenOOD leaderboard, we use the standard torchvision ViT-B/16 checkpoint trained at image size 224. Download it:

```bash
mkdir -p checkpoints
wget https://download.pytorch.org/models/vit_b_16-c867db91.pth -O checkpoints/vit_b_16-c867db91.pth
```

Edit `configs/networks/vit_b16.yml` — update the checkpoint path and image size:

```yaml
checkpoint: ./checkpoints/vit_b_16-c867db91.pth
image_size: 224
```

Also update the image size in `configs/datasets/imagenet/imagenet_ood.yml` (or whichever OOD config you use) to match:

```yaml
image_size: 224
```

Then run:

```bash
python main.py \
  --config \
    configs/datasets/imagenet/imagenet.yml \
    configs/datasets/imagenet/imagenet_ood.yml \
    configs/networks/vit_b16.yml \
    configs/pipelines/test/test_ood.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/jointpca.yml \
  --num_gpus 0
```

**Filtered variant**

Set `filtered: true` in `configs/postprocessors/jointpca.yml` before running any of the above. A spectrum plot will be saved to `results/jointpca_cache/plots/` for verification. See DEVELOPMENT.md.

---

## Compute

All experiments were run on a Mac Mini M4 Pro with 24 GB unified memory, without a GPU. The largest configuration (ResNet-50, 45 000 training samples) produces a feature matrix $A$ of shape $45000 \times 43712$. PCA is performed on the $43712 \times 43712$ sample covariance matrix $A^\top A$, keeping memory tractable on consumer hardware. Further runtime details are in the paper.

**On the choice of ID split for PCA fitting.** JointPCA fits PCA on the ID test split rather than the training split. This is valid because both splits are drawn from the same distribution — the number of samples that matter is small relative to the dataset size, and we verified that using the same number of samples from the training split produces identical results. The practical reason for using the test split is that ImageNet-1K training images (~150 GB) are not available via the OpenOOD download script and require a separate registration at image-net.org, while the test split is readily available.

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
