<p align="center">
  <img src="https://img.shields.io/badge/MICCAI-2025_(Oral)-blueviolet.svg?style=flat-square">
  <img src="https://img.shields.io/badge/PyTorch_Lightning-2.6+-ee4c2c.svg?style=flat-square&logo=pytorch-lightning">
  <img src="https://img.shields.io/badge/python-3.12-blue.svg?style=flat-square">
  <img src="https://img.shields.io/badge/license-MIT-green.svg?style=flat-square">
</p>

<h1 align="center">TESLA: Test-time Reference-free Through-plane Super-resolution</h1>

<p align="center">
  <b>Official PyTorch Lightning implementation</b> of <br>
  <i>"TESLA: Test-time Reference-free Through-plane Super-resolution for Multi-contrast Brain MRI"</i> <br>
  <br>
  Accepted at <strong>MICCAI 2025 (Oral)</strong>
  <br><br>
  <a href="https://huggingface.co/yoonseokchoi/tesla-ckpts">Pretrained Weights</a> &bull;
  <a href="https://huggingface.co/datasets/yoonseokchoi/tesla-ixi">Dataset</a> &bull;
  <a href="#-setup">Setup</a> &bull;
  <a href="#-training">Training</a> &bull;
  <a href="#-inference">Inference</a>
</p>

---

![TESLA Architecture](assets/tesla_architecture.jpg)

## Overview

TESLA is a **test-time reference-free** through-plane super-resolution framework for multi-contrast brain MRI. During training, TESLA leverages cross-modal information (e.g., T1-weighted) to learn content-style disentangled representations via AdaIN-based generators. At test time, only the degraded target image is required — no reference image is needed.

The training pipeline consists of four sequential stages:

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | **ContentNet** | Self-reconstruction pre-training on the reference domain (T1/PD) to learn content-style disentanglement |
| 2 | **ProTPSR 4&rarr;2** | Progressive through-plane SR: reconstruct 2x resolution from 4x degraded T2 |
| 3 | **ProTPSR 2&rarr;1** | Progressive through-plane SR: reconstruct HR from the output of Stage 2 |
| 4 | **TESLA** | Full cross-modal SR with frozen ContentNet + ProTPSR, PatchNCE contrastive loss, and PatchGAN discriminator |

---

## Repository Structure

```
TESLA/
├── configs/
│   ├── tesla_train.yaml          # Training configuration (all stages)
│   └── tesla_inference.yaml      # Inference configuration
├── assets/
│   └── tesla_architecture.jpg    # Architecture figure
│
├── lightning_module.py           # Lightning modules (ContentNet, ProTPSR, TESLA)
├── networks_tesla.py             # AdaINGen, PatchGAN, PatchSampleF (TESLA stage)
├── networks_contentnet.py        # AdaINGen, PatchGAN (ContentNet stage)
├── data_module.py                # PyTorch Lightning DataModule
├── customdataset_h5_tesla.py     # HDF5 dataset loader
├── utils.py                      # Weight init, schedulers, utilities
│
├── train_lightning.py            # Training entry point (all 4 stages)
├── inference_lightning.py        # Inference entry point (all stages)
├── visualize_results.ipynb       # Jupyter notebook for result visualization
│
├── environment.yaml              # Conda environment specification
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yoonseokchoi-ai/TESLA.git
cd TESLA
```

### 2. Create Conda Environment

```bash
# Create and activate the environment
conda env create -f environment.yaml
conda activate tesla
```

> **Note:** The default `environment.yaml` installs PyTorch with CUDA support. If your NVIDIA driver supports a specific CUDA version, you can install PyTorch separately:
> ```bash
> # Example: Install PyTorch for CUDA 12.9
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
> ```
> Check your driver's CUDA version with `nvidia-smi`.

### 3. Download Dataset

TESLA uses the [IXI dataset](https://brain-development.org/ixi-dataset/) preprocessed into HDF5 format.

```bash
# Install git-lfs (required for large files)
git lfs install

# Clone dataset into ./data
git clone https://huggingface.co/datasets/yoonseokchoi/tesla-ixi data
cd data && git lfs pull && cd ..
```

After downloading, the data directory should look like:

```
data/
└── ixi/
    ├── train/
    │   └── output_data_1.2mm_2.4mm_4.8mm_50.h5    # 4,000 training slices
    └── test/
        └── output_data_1.2mm_2.4mm_4.8mm_50.h5     # 1,000 test slices
```

Each `.h5` file contains the following keys with shape `(N, 1, 128, 256)`:

| Key | Description |
|-----|-------------|
| `data_A` | HR T1-weighted (reference domain) |
| `data_PD` | HR PD-weighted (reference domain) |
| `data_B_HR` | HR T2-weighted (target, ground truth) |
| `data_B_21` | Interpolated LR T2 (2x &rarr; 1x) |
| `data_B_41` | Interpolated LR T2 (4x &rarr; 1x) |
| `data_B_2fold` | Downsampled LR T2 (2x) |
| `data_B_4fold` | Downsampled LR T2 (4x) |
| `data_B_SR_2to1` | Pre-computed SR T2 (2x &rarr; 1x) |
| `data_B_SR_4to2` | Pre-computed SR T2 (4x &rarr; 2x) |

### 4. Download Pretrained Weights (Optional)

```bash
git clone https://huggingface.co/yoonseokchoi/tesla-ckpts ckpts
cd ckpts && git lfs pull && cd ..
```

```
ckpts/
├── contentnet/
│   └── lightning/
│       └── last.ckpt             # ContentNet (SSIM 0.9982)
├── progressive/
│   └── lightning/
│       ├── prog_4to2/
│       │   └── last.ckpt         # ProTPSR 4→2 (SSIM 0.9407)
│       └── prog_2to1/
│           └── last.ckpt         # ProTPSR 2→1 (SSIM 0.9205)
└── tesla/
    └── lightning/
        └── last.ckpt             # TESLA (SSIM 0.9198)
```

---

## Training

TESLA training follows a **four-stage sequential pipeline**. All stages are launched via the same entry point (`train_lightning.py`) with different `--stage` flags. Training progress is logged to [Weights & Biases](https://wandb.ai/).

> **GPU Selection:** Set `CUDA_VISIBLE_DEVICES` to select a GPU.
> **Config Overrides:** Any config value can be overridden via CLI with dot-notation (e.g., `--training.batch_size 4`).

### Stage 1: ContentNet Pre-training

ContentNet learns **content-style disentanglement** on the reference domain (T1 or PD) via self-reconstruction. The content encoder captures structural information while the style encoder captures contrast-specific features. This frozen ContentNet is later used in TESLA for cross-modal contrastive learning (PatchNCE).

```bash
CUDA_VISIBLE_DEVICES=0 python train_lightning.py \
    --config configs/tesla_train.yaml \
    --stage contentnet
```

### Stage 2: ProTPSR 4x &rarr; 2x

The first progressive through-plane SR stage. An AdaINGen network learns to reconstruct **2x resolution T2** (`x_b_21`) from the **4x degraded T2** (`x_b_41`). Training uses L1 + SSIM reconstruction losses and a data consistency loss that ensures the downsampled output matches the degraded input.

```bash
CUDA_VISIBLE_DEVICES=0 python train_lightning.py \
    --config configs/tesla_train.yaml \
    --stage prog_4to2
```

### Stage 3: ProTPSR 2x &rarr; 1x

The second progressive SR stage. A new AdaINGen takes the **frozen Stage 2 output** as input and reconstructs **HR T2** (`x_b_hr`). The Stage 2 model is loaded and frozen automatically.

```bash
CUDA_VISIBLE_DEVICES=0 python train_lightning.py \
    --config configs/tesla_train.yaml \
    --stage prog_2to1 \
    --progressive.prog_4to2_ckpt_path ckpts/progressive/lightning/prog_4to2/last.ckpt
```

### Stage 4: TESLA

The full TESLA model. The frozen **ProTPSR pipeline** (Stage 2 + 3) generates the SR input online from `x_b_41`, then `gen_b` (AdaINGen) refines it with:
- **PatchGAN** adversarial loss with Canny edge conditioning
- **PatchNCE** cross-modal contrastive loss between frozen ContentNet (T1) and gen_b (T2) feature spaces
- **L1 + SSIM** reconstruction and data consistency losses

```bash
CUDA_VISIBLE_DEVICES=0 python train_lightning.py \
    --config configs/tesla_train.yaml \
    --stage tesla \
    --contentnet_ckpt_path ckpts/contentnet/lightning/last.ckpt \
    --progressive.prog_4to2_ckpt_path ckpts/progressive/lightning/prog_4to2/last.ckpt \
    --progressive.prog_2to1_ckpt_path ckpts/progressive/lightning/prog_2to1/last.ckpt
```

### Training Configuration

All training parameters are defined in [`configs/tesla_train.yaml`](configs/tesla_train.yaml). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 100 | Number of training epochs |
| `training.batch_size` | 8 | Batch size |
| `training.gen_lr` | 0.0001 | Generator learning rate |
| `training.dis_lr` | 0.0001 | Discriminator learning rate |
| `data.crf_domain` | `t1` | Edge conditioning domain (`t1`/`pd`/`t2`/`srt2`/`none`) |
| `data.sr_scale` | 4 | Super-resolution scale factor |

---

## Inference

Inference is configured separately via [`configs/tesla_inference.yaml`](configs/tesla_inference.yaml), which specifies per-stage checkpoint paths, GPU, batch size, and output directory.

```bash
# Run TESLA inference (full pipeline)
CUDA_VISIBLE_DEVICES=0 python inference_lightning.py \
    --config configs/tesla_inference.yaml \
    --stage tesla
```

You can also run inference for individual stages:

```bash
# ContentNet
python inference_lightning.py --config configs/tesla_inference.yaml --stage contentnet

# ProTPSR stages
python inference_lightning.py --config configs/tesla_inference.yaml --stage prog_4to2
python inference_lightning.py --config configs/tesla_inference.yaml --stage prog_2to1
```

### Outputs

Inference produces two files in the output directory:

**`inference_tesla.h5`** — Image results:

| Key | Shape | Description |
|-----|-------|-------------|
| `x_lr` | `(N, 1, 128, 256)` | Original 4x degraded input (interpolated to HR size) |
| `x_protpsr` | `(N, 1, 128, 256)` | ProTPSR output (progressive SR) |
| `x_sr` | `(N, 1, 128, 256)` | TESLA final SR output |
| `x_hr` | `(N, 1, 128, 256)` | HR ground truth |
| `x_a` | `(N, 1, 128, 256)` | Reference T1 image |

**`inference_tesla.json`** — Quantitative report:

```json
{
  "config": {
    "stage": "tesla",
    "ckpt_path": { "tesla": "ckpts/tesla/lightning/last.ckpt", ... },
    "dataset": "IXI",
    "data_dir": "./data/ixi/",
    "inference_date": "2025-03-25 12:00:00"
  },
  "summary": {
    "n_samples": 1000,
    "ssim_mean": 0.9198,
    "ssim_std": 0.0234,
    "psnr_mean": 30.67,
    "psnr_std": 2.41
  },
  "per_sample": [
    { "index": 0, "ssim": 0.9234, "psnr": 31.45 },
    ...
  ]
}
```

### Visualization

Open [`visualize_results.ipynb`](visualize_results.ipynb) to visualize inference results:

- Per-sample SSIM / PSNR distribution histograms
- Side-by-side comparison: `x_lr` (degraded) | `x_protpsr` (ProTPSR) | `x_sr` (TESLA) | `x_hr` (GT) | Error map | `x_a` (Reference)
- Best / median / worst sample visualization sorted by SSIM

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{choi2025tesla,
  title={TESLA: Test-time Reference-free Through-plane Super-resolution for Multi-contrast Brain MRI},
  author={Choi, Yoonseok and others},
  booktitle={Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year={2025}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
