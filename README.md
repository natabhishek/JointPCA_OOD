# JointPCA — OOD Detection via Multi-Layer PCA

This repository contains the implementation of **JointPCA**, an out-of-distribution (OOD) detection postprocessor designed to plug into the [OpenOOD](https://github.com/Jingkang50/OpenOOD) benchmark framework.

Supported backbones: **ResNet-50** and **ViT**.

---

## Method

### Feature extraction

Forward hooks are registered on the model before any inference runs. The hooked layers depend on the architecture:

**ResNet-50 (CNN)**  
Hooks are placed on:
1. Every `Conv2d` layer (all kernel sizes)
2. Every residual block output — matched by the pattern `layer{1-4}.{block_idx}` (e.g. `layer1.0`, `layer3.2`)
3. The penultimate layer (the last non-linear, non-classifier layer, typically the post-GAP feature vector)

Spatial feature maps `(B, C, H, W)` are reduced to `(B, C)` by **global average pooling** before concatenation.

**ViT (Vision Transformer)**  
Hooks are placed on every encoder block — detected automatically by checking for `nn.MultiheadAttention` + `nn.LayerNorm` as direct children. From each block output `(B, L, D)`:
- The **CLS token** `[:, 0, :]` → `(B, D)`
- The **patch-GAP** `[:, 1:, :].mean(dim=1)` → `(B, D)`

These two are concatenated, giving `(B, 2D)` per block.

All per-layer vectors are concatenated along the feature axis into a single joint feature vector of dimension $D_\text{joint}$.

### PCA

Sklearn's **randomised PCA** (`svd_solver='randomized'`) is fit on up to `max_train_samples` ID training features. The full rank is used: $K = \min(N, D_\text{joint})$. The mean vector, principal components, and per-component eigenvalues are cached to disk after the first run.

### Scoring

Each test sample is projected onto the PCA subspace and scored with the **squared Mahalanobis distance** from the ID training mean:

$$\text{score}(z) = \sum_{i=1}^{K} \frac{y_i^2}{\lambda_i}, \quad y_i = u_i^\top (z - \mu)$$

where $\lambda_i$ is the $i$-th eigenvalue (variance along the $i$-th PC). A **higher** score indicates the sample is further from the ID distribution (more likely OOD). OpenOOD receives the negated score so that higher = more in-distribution, matching its convention.

### Caching

All intermediate artefacts are stored as memory-mapped NumPy arrays under `./jointpca_cache/`:

```
jointpca_cache/
  features/
    features_train_{config_fp}.npy       # ID training features  (N, D)
    features_{dataset}_{config_fp}.npy   # OOD / test features   (N, D)
  pca/
    pca_{config_fp}.npz                  # mean, components, eigenvalues
  projections/
    projections_train_{config_fp}.npy    # train projections      (N, K)
  scores/
    scores_{dataset}_{config_fp}_md.npz  # per-sample MD scores
  metadata/
    metadata_{config_fp}.npz             # n_samples, n_features, layer_names
```

The config fingerprint `{config_fp}` encodes the model class name and `max_train_samples`. Changing either value automatically triggers a fresh extraction and PCA fit. Delete individual cache files to force recomputation of a specific stage.

---

## Setup

### 1. Install OpenOOD

Clone and install the OpenOOD framework:

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
