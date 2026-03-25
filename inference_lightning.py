"""
TESLA Inference with PyTorch Lightning.

Usage:
  # TESLA (full pipeline):
  python inference_lightning.py --config configs/tesla_inference.yaml --stage tesla

  # ContentNet:
  python inference_lightning.py --config configs/tesla_inference.yaml --stage contentnet

  # Progressive stages:
  python inference_lightning.py --config configs/tesla_inference.yaml --stage prog_2to1

  # Override ckpt or output:
  python inference_lightning.py --config configs/tesla_inference.yaml --stage tesla \
    --ckpt_path ckpts/tesla/lightning/last.ckpt --output_dir outputs/custom/
"""

import os
import json
import argparse
import yaml
import h5py
import numpy as np
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric
from tqdm import tqdm

from lightning_module import TESLALightningModule, ContentNetLightningModule, ProgressiveReconModule
from customdataset_h5_tesla import IXI_Dataset


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_dataset_config(config: dict):
    """Build flat config namespace for IXI_Dataset."""

    class _Cfg:
        pass

    cfg = _Cfg()
    cfg.dataset = config["data"]["dataset"]
    cfg.ixi_h5_1_2mm_dir = config["data"]["ixi_h5_dir"]
    cfg.nb_train_imgs = config["data"]["nb_train_imgs"]
    cfg.nb_test_imgs = config["data"]["nb_test_imgs"]
    cfg.crf_domain = config["data"]["crf_domain"]
    cfg.sr_scale = config["data"]["sr_scale"]
    cfg.hr_pd = config["data"].get("hr_pd", False)
    return cfg


def build_train_config(config: dict) -> dict:
    """Build a training-compatible config dict for loading Lightning modules.

    Inference config is minimal, so we fill in required fields with defaults
    that match the training config structure.
    """
    train_cfg = {
        "model": config["model"],
        "data": config["data"],
        "stage": config.get("stage", "tesla"),
        "contentnet_ckpt_path": config.get("ckpt_path", {}).get("contentnet"),
        "progressive": config.get("progressive", {
            "prog_4to2_ckpt_path": config.get("ckpt_path", {}).get("prog_4to2"),
            "prog_2to1_ckpt_path": config.get("ckpt_path", {}).get("prog_2to1"),
        }),
        "training": {
            "gen_lr": 0.0001, "dis_lr": 0.0001,
            "beta1": 0.5, "beta2": 0.999, "weight_decay": 0.0001,
            "lr_policy": "constant", "step_size": 30, "gamma": 0.5,
            "patience": 10, "factor": 0.1, "init": "kaiming",
            "batch_size": config.get("batch_size", 8),
            "epochs": 1, "workers": config.get("num_workers", 4),
            "generator_steps": 1, "discriminator_steps": 1,
        },
        "loss": {
            "gan_w": 0, "recon_l1_x_w": 0, "recon_l1_s_w": 0,
            "recon_l1_c_w": 0, "recon_l1_cyc_w": 0, "recon_ssim_x_w": 0,
            "recon_ssim_c_w": 0, "recon_ssim_cyc_w": 0, "recon_x_cyc_w": 0,
            "recon_patchnce_w": 0, "dc_l1_w": 0, "dc_ssim_w": 0,
        },
        "dc": {"dc_avg": False, "dc_monai": True, "dc_monai_method": "area"},
        "nce": {
            "nce": False, "nce_idt": False, "nce_layers": [0, 1, 2],
            "nce_T": 0.07, "lambda_NCE": 0, "num_patches": 256,
            "netF": "mlp_sample", "netF_nc": 256,
            "nce_includes_all_negatives_from_minibatch": False,
        },
    }
    return train_cfg


def main():
    parser = argparse.ArgumentParser(description="TESLA Lightning Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to inference YAML config")
    parser.add_argument("--stage", type=str, default="tesla",
                        help="Stage: contentnet | prog_4to2 | prog_2to1 | tesla")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Override checkpoint path")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--gpu", type=str, default=None, help="Override GPU id")
    args = parser.parse_args()

    _STAGE_ALIASES = {"1": "contentnet", "2": "tesla"}
    stage = _STAGE_ALIASES.get(args.stage, args.stage)

    config = load_config(args.config)

    # Resolve ckpt path: CLI override > config per-stage path
    ckpt_paths = config.get("ckpt_path", {})
    ckpt_path = args.ckpt_path or ckpt_paths.get(stage)
    if not ckpt_path:
        raise ValueError(f"No checkpoint path for stage '{stage}'. Set in config or pass --ckpt_path.")

    output_dir = args.output_dir or config.get("output_dir", "outputs/tesla_inference/")
    batch_size = args.batch_size or config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)
    gpu = args.gpu or str(config.get("gpu", "0"))

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build training-compatible config for Lightning module loading
    train_cfg = build_train_config(config)
    train_cfg["stage"] = stage

    # Load model
    if stage == "contentnet":
        model = ContentNetLightningModule.load_from_checkpoint(ckpt_path, config=train_cfg, strict=False)
    elif stage in ("prog_4to2", "prog_2to1"):
        model = ProgressiveReconModule.load_from_checkpoint(ckpt_path, config=train_cfg, strict=False)
    elif stage == "tesla":
        model = TESLALightningModule.load_from_checkpoint(ckpt_path, config=train_cfg, strict=False)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    model = model.to(device)
    model.eval()

    # Dataset
    ds_config = build_dataset_config(config)
    test_dataset = IXI_Dataset(config=ds_config, mode="test")
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Metrics
    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    psnr_metric = PSNRMetric(max_val=1.0)

    # Per-sample metric storage
    sample_ssim = []
    sample_psnr = []

    # Image storage
    # Paper notation: x_lr = degraded input, x_protpsr = progressive SR output, x_sr = TESLA output
    all_x_sr = []
    all_x_hr = []
    all_x_lr = []         # original degraded input (e.g. x_b_41)
    all_x_protpsr = []    # progressive SR output (prog_4to2 → prog_2to1)
    all_x_a = []          # reference (T1/PD)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Stage: {stage}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Output: {output_dir}")
    print(f"GPU: {gpu}, Batch size: {batch_size}")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            x_b_hr = batch["data_B_HR"].to(device)
            bs = x_b_hr.size(0)

            if stage == "contentnet":
                x_a = batch["data_PD"].to(device) if config["data"].get("hr_pd", False) else batch["data_A"].to(device)
                c_a, s_a = model.gen_a.encode(x_a)
                x_a_recon = model.gen_a.decode(c_a[-1], s_a)
                x_pred, x_gt = x_a_recon, x_a
                all_x_lr.append(x_a.cpu().numpy())

            elif stage == "prog_4to2":
                x_b_41 = batch["data_B_41"].to(device)
                c, s = model.gen.encode(x_b_41)
                x_recon = model.gen.decode(c[-1], s)
                x_pred, x_gt = x_recon, batch["data_B_21"].to(device)
                all_x_lr.append(x_b_41.cpu().numpy())

            elif stage == "prog_2to1":
                x_b_41 = batch["data_B_41"].to(device)
                c, s = model.gen_4to2.encode(x_b_41)
                x_4to2 = model.gen_4to2.decode(c[-1], s)
                c, s = model.gen.encode(x_4to2)
                x_recon = model.gen.decode(c[-1], s)
                x_pred, x_gt = x_recon, x_b_hr
                all_x_lr.append(x_b_41.cpu().numpy())

            elif stage == "tesla":
                x_b_41 = batch["data_B_41"].to(device)
                x_b_protpsr = model._progressive_forward(x_b_41)
                c_b, s_b = model.gen_b.encode(x_b_protpsr)
                x_b_recon = model.gen_b.decode(c_b[-1], s_b)
                x_pred, x_gt = x_b_recon, x_b_hr
                all_x_lr.append(x_b_41.cpu().numpy())
                all_x_protpsr.append(x_b_protpsr.cpu().numpy())
                if "data_A" in batch:
                    all_x_a.append(batch["data_A"].numpy())

            # Per-sample metrics
            ssim_val = ssim_metric(y_pred=x_pred, y=x_gt)  # (B,)
            psnr_val = psnr_metric(y_pred=x_pred, y=x_gt)  # (B,)

            for i in range(bs):
                sample_ssim.append(ssim_val[i].item())
                sample_psnr.append(psnr_val[i].item())

            all_x_sr.append(x_pred.cpu().numpy())
            all_x_hr.append(x_gt.cpu().numpy())

    ssim_arr = np.array(sample_ssim)
    psnr_arr = np.array(sample_psnr)

    print(f"\n{'='*50}")
    print(f"  Inference Results ({stage})")
    print(f"  Samples : {len(ssim_arr)}")
    print(f"  SSIM    : {ssim_arr.mean():.4f} +/- {ssim_arr.std():.4f}")
    print(f"  PSNR    : {psnr_arr.mean():.2f} +/- {psnr_arr.std():.2f} dB")
    print(f"{'='*50}")

    # ---- Save H5 ----
    h5_path = os.path.join(output_dir, f"inference_{stage}.h5")
    all_x_sr = np.concatenate(all_x_sr, axis=0)
    all_x_hr = np.concatenate(all_x_hr, axis=0)
    all_x_lr = np.concatenate(all_x_lr, axis=0)

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("x_sr", data=all_x_sr)
        f.create_dataset("x_hr", data=all_x_hr)
        f.create_dataset("x_lr", data=all_x_lr)
        if all_x_protpsr:
            f.create_dataset("x_protpsr", data=np.concatenate(all_x_protpsr, axis=0))
        if all_x_a:
            f.create_dataset("x_a", data=np.concatenate(all_x_a, axis=0))

    print(f"H5 saved to {h5_path}")

    # ---- Save JSON report ----
    json_path = os.path.join(output_dir, f"inference_{stage}.json")

    report = {
        "config": {
            "stage": stage,
            "ckpt_path": {stage: ckpt_path},
            "dataset": config["data"]["dataset"],
            "data_dir": config["data"]["ixi_h5_dir"],
            "nb_test_imgs": config["data"]["nb_test_imgs"],
            "sr_scale": config["data"]["sr_scale"],
            "crf_domain": config["data"]["crf_domain"],
            "batch_size": batch_size,
            "gpu": gpu,
            "inference_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "summary": {
            "n_samples": len(ssim_arr),
            "ssim_mean": round(float(ssim_arr.mean()), 6),
            "ssim_std": round(float(ssim_arr.std()), 6),
            "ssim_min": round(float(ssim_arr.min()), 6),
            "ssim_max": round(float(ssim_arr.max()), 6),
            "ssim_median": round(float(np.median(ssim_arr)), 6),
            "psnr_mean": round(float(psnr_arr.mean()), 4),
            "psnr_std": round(float(psnr_arr.std()), 4),
            "psnr_min": round(float(psnr_arr.min()), 4),
            "psnr_max": round(float(psnr_arr.max()), 4),
            "psnr_median": round(float(np.median(psnr_arr)), 4),
        },
        "per_sample": [
            {"index": i, "ssim": round(float(ssim_arr[i]), 6), "psnr": round(float(psnr_arr[i]), 4)}
            for i in range(len(ssim_arr))
        ],
    }

    # Add all ckpt paths for tesla stage
    if stage == "tesla" and isinstance(ckpt_paths, dict):
        report["config"]["ckpt_path"] = {k: v for k, v in ckpt_paths.items() if v}

    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"JSON report saved to {json_path}")


if __name__ == "__main__":
    main()
