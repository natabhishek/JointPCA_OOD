# JointPCA — OOD Detection via Multi-Layer PCA

This repository contains the implementation of **JointPCA**, an out-of-distribution (OOD) detection method that extracts activations from multiple network layers simultaneously, concatenates them into a single joint feature vector, fits PCA, and scores test samples using the **Mahalanobis distance** in the learned PCA subspace.

The code is designed as a drop-in postprocessor for the [OpenOOD](https://github.com/Jingkang50/OpenOOD) benchmark framework.

---

## Files

| File | Description |
|---|---|
| `jointpca_postprocessor.py` | Main postprocessor class |
| `jointpca.yml` | Config file for the OpenOOD pipeline |
| `README.md` | This file |

---

## Method overview

**Feature extraction.**  
Forward hooks are registered on the model before inference. For **ResNet50 / RegNet / WideResNet** (CNN), hooks are placed on every `Conv2d` layer, every residual block output, and the penultimate layer; spatial feature maps are reduced to vectors via global average pooling. For **ViT** models, hooks capture the output of every encoder block, and each activation is split into the CLS token and the patch-GAP, which are concatenated. All per-layer vectors are then concatenated into a single high-dimensional joint feature.

**PCA.**  
Sklearn's randomised PCA is fit on up to `max_train_samples` ID training features, using the full rank (min(N, D) components). The mean, principal components, and per-component eigenvalues are cached to disk.

**Scoring.**  
Each test sample is projected onto the PCA subspace and scored with the squared Mahalanobis distance from the ID training mean:

$$\text{score}(z) = \sum_{i=1}^{K} \frac{y_i^2}{\lambda_i}$$

where $y_i = u_i^\top (z - \mu)$ is the projection onto the $i$-th principal component and $\lambda_i$ is the corresponding eigenvalue. A **lower** score means the sample is closer to the ID distribution; **higher** score means OOD.

**Caching.**  
All intermediate artefacts (raw features, PCA model, train projections, per-dataset scores) are stored as memory-mapped NumPy arrays in `./jointpca_cache/`. Subsequent runs on the same model and dataset skip all recomputation automatically.

---

## Prerequisites: OpenOOD setup

JointPCA runs inside the OpenOOD framework. Follow the steps below to get OpenOOD installed and configured before integrating JointPCA.

### 1. Clone OpenOOD

```bash
git clone https://github.com/Jingkang50/OpenOOD.git
cd OpenOOD
```

The official OpenOOD documentation lives at the repository above. For general questions about datasets, networks, or the pipeline, refer to the [OpenOOD README](https://github.com/Jingkang50/OpenOOD/blob/main/README.md) and the [OpenOOD wiki](https://github.com/Jingkang50/OpenOOD/wiki).

### 2. Install dependencies

```bash
pip install -e .
# or, for a quick install without cloning:
pip install git+https://github.com/Jingkang50/OpenOOD
```

Ensure `scikit-learn`, `scipy`, `tqdm`, and `torch` are available (they are listed in OpenOOD's `setup.py`).

### 3. Download data

OpenOOD's evaluator can download benchmark splits automatically. For large-scale experiments (ImageNet), download the ImageNet-1K training images from the [official source](https://image-net.org/) and point the config to the correct path.  
Pre-downloaded benchmark image lists and OOD data are available via the [OpenOOD downloading script](https://github.com/Jingkang50/OpenOOD#data).

### 4. Download a pre-trained checkpoint

OpenOOD provides pre-trained ResNet-18, ResNet-50, and ViT checkpoints for CIFAR-10, CIFAR-100, ImageNet-200, and ImageNet-1K. Download them from the links in the [OpenOOD README](https://github.com/Jingkang50/OpenOOD#model-checkpoints) and place them under `results/checkpoints/`.

---

## Integrating JointPCA into OpenOOD

Three small edits are needed inside the OpenOOD source tree. All edits are additive; nothing existing is removed.

### Step 1 — Copy the postprocessor file

```bash
cp jointpca_postprocessor.py openood/postprocessors/jointpca_postprocessor.py
```

### Step 2 — Register the postprocessor in `openood/postprocessors/__init__.py`

Open `openood/postprocessors/__init__.py` and add the following two lines alongside the existing imports and the postprocessor dictionary entry:

```python
# At the top of the file, with the other imports:
from .jointpca_postprocessor import JointPCAPostprocessor

# Inside the postprocessors dictionary (postprocessor_dict or similar mapping):
'jointpca': JointPCAPostprocessor,
```

The exact variable name for the dictionary differs slightly between OpenOOD versions. Look for a block such as:

```python
postprocessors_dict = {
    'msp': MSPPostprocessor,
    'odin': ODINPostprocessor,
    ...
}
```

and add `'jointpca': JointPCAPostprocessor` there.

### Step 3 — Copy the config file

```bash
cp jointpca.yml configs/postprocessors/jointpca.yml
```

---

## Running JointPCA

JointPCA follows the standard OpenOOD command-line interface. The required config files are:
- A **dataset** config (ID dataset + OOD splits)
- A **network** config (architecture)
- A **pipeline** config (e.g. `test_ood.yml`)
- A **preprocessor** config (image transforms)
- The **JointPCA postprocessor** config (`jointpca.yml`)

### Example: ResNet-50 on ImageNet-1K

```bash
python main.py \
  --config configs/datasets/imagenet/imagenet.yml \
            configs/datasets/imagenet/imagenet_ood.yml \
            configs/networks/resnet50.yml \
            configs/pipelines/test/test_ood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/jointpca.yml \
  --network.checkpoint results/checkpoints/resnet50_imagenet1k.ckpt \
  --network.pretrained True
```

### Example: ViT on ImageNet-1K

```bash
python main.py \
  --config configs/datasets/imagenet/imagenet.yml \
            configs/datasets/imagenet/imagenet_ood.yml \
            configs/networks/vit_b16.yml \
            configs/pipelines/test/test_ood.yml \
            configs/preprocessors/base_preprocessor.yml \
            configs/postprocessors/jointpca.yml \
  --network.checkpoint results/checkpoints/vit_b16_imagenet1k.ckpt \
  --network.pretrained True
```

### Adjusting the number of training samples

Edit `max_train_samples` in `jointpca.yml`. The default is 45 000, which balances PCA quality and memory use. For CIFAR-scale experiments, 10 000–20 000 is usually sufficient.

```yaml
postprocessor_args:
  max_train_samples: 20000
```

---

## Cache layout

All cached artefacts are written to `./jointpca_cache/` relative to the working directory.

```
jointpca_cache/
  features/
    features_train_{config_fp}.npy      # ID training features   (N, D)
    features_{dataset}_{config_fp}.npy  # OOD / test features    (N, D)
  pca/
    pca_{config_fp}.npz                 # mean, components, eigenvalues
  projections/
    projections_train_{config_fp}.npy   # train projections       (N, K)
  scores/
    scores_{dataset}_{config_fp}_md.npz # per-sample MD scores
  metadata/
    metadata_{config_fp}.npz            # n_samples, n_features, layer_names
```

Delete any of these files to force recomputation of that stage. The config fingerprint `{config_fp}` encodes the model class name and `max_train_samples`, so changing either one automatically triggers a fresh run.

---

## Expected output

During `setup()` the postprocessor prints the full list of hooked layer names, which lets you verify that the correct layers are being captured:

```
[JointPCA] CNN hooks registered: 54 layers
[JointPCA] Layer names (54 total):
  layer1.0.conv1
  layer1.0.conv2
  ...
  layer1.0
  layer1.1
  ...
  avgpool
[JointPCA] Feature dim: 102400
[JointPCA] Fitting PCA: N=45000, D=102400, K=45000
[JointPCA] PCA done. Top-100 PCs explain 61.3% variance.
[JointPCA] Setup complete.
```

During `inference()` a progress bar is shown for feature extraction and the final AUROC / FPR95 metrics are printed by OpenOOD's evaluator.

---

## Dependencies

All of the following are already required by OpenOOD:

- `torch >= 1.13`
- `torchvision`
- `scikit-learn`
- `scipy`
- `numpy`
- `tqdm`

No additional packages are needed.

---

## Citation

If you use this code, please cite both the OpenOOD benchmark and this work:

```bibtex
@article{zhang2023openood,
  title     = {OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author    = {Zhang, Jingyang and others},
  year      = {2023}
}
```

```bibtex
@inproceedings{yang2022openood,
  title     = {OpenOOD: Benchmarking Generalized Out-of-Distribution Detection},
  author    = {Yang, Jingkang and others},
  booktitle = {NeurIPS},
  year      = {2022}
}
```
