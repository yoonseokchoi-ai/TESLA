"""
TESLA Training with PyTorch Lightning + W&B Logging.

Usage:
  # ContentNet pre-training:
  python train_lightning.py --config configs/tesla_train.yaml --stage contentnet

  # Progressive through-plane SR (4x→2x):
  python train_lightning.py --config configs/tesla_train.yaml --stage prog_4to2

  # Progressive through-plane SR (2x→1x, requires frozen prog_4to2):
  python train_lightning.py --config configs/tesla_train.yaml --stage prog_2to1

  # TESLA training (loads frozen ContentNet):
  python train_lightning.py --config configs/tesla_train.yaml --stage tesla

  # Override any config value via CLI:
  python train_lightning.py --config configs/tesla_train.yaml --stage tesla --training.batch_size 4 --training.epochs 50
"""

import os
import argparse
import yaml

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from lightning_module import TESLALightningModule, ContentNetLightningModule, ProgressiveReconModule
from data_module import TESLADataModule


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def apply_cli_overrides(config: dict, overrides: list) -> dict:
    """Apply dot-notation CLI overrides like --training.batch_size 4."""
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
            value = overrides[i + 1]
            i += 2
        else:
            value = "true"
            i += 1

        # Convert value type
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False

        # Navigate nested keys
        keys = key.split(".")
        d = config
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    return config


def main():
    parser = argparse.ArgumentParser(description="TESLA Lightning Training")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--stage", type=str, default=None,
                        help="Training stage: contentnet | prog_4to2 | prog_2to1 | tesla (or legacy 1/2)")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to Lightning checkpoint to resume from")

    args, unknown = parser.parse_known_args()

    # Load config
    config = load_config(args.config)

    # CLI overrides
    if unknown:
        config = apply_cli_overrides(config, unknown)

    # Stage override & normalize legacy integer values
    if args.stage is not None:
        config["stage"] = args.stage

    stage = str(config.get("stage", "tesla"))
    _STAGE_ALIASES = {"1": "contentnet", "2": "tesla"}
    stage = _STAGE_ALIASES.get(stage, stage)
    config["stage"] = stage

    # GPU setup — respect CUDA_VISIBLE_DEVICES if already set externally
    gpu_str = os.environ.get("CUDA_VISIBLE_DEVICES", str(config.get("gpu", "0")))
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    num_gpus = len(gpu_str.split(","))
    strategy = config.get("strategy", "auto")
    if num_gpus > 1 and strategy == "auto":
        strategy = "ddp"

    # Seed
    pl.seed_everything(42, workers=True)

    # ---- W&B Logger ----
    wandb_cfg = config.get("wandb", {})
    run_name = wandb_cfg.get("run_name", None)
    if run_name is None:
        run_name = f"tesla_{stage}"

    offline = wandb_cfg.get("offline", False)
    wandb_logger = WandbLogger(
        project = wandb_cfg.get("project", "tesla-sr"),
        name    = run_name,
        entity  = wandb_cfg.get("entity", None),
        config  = config,
        save_dir= "logs/",
        offline = offline,
    )

    # ---- DataModule ----
    data_module = TESLADataModule(config)

    # ---- LightningModule ----
    if stage == "contentnet":
        model = ContentNetLightningModule(config)
        ckpt_dir = os.path.join(config["dirs"]["ckpt_dir_ContentNet"], "lightning")
    elif stage in ("prog_4to2", "prog_2to1"):
        model = ProgressiveReconModule(config)
        ckpt_dir = os.path.join(config["dirs"]["ckpt_dir_progressive"], "lightning", stage)
    elif stage == "tesla":
        model = TESLALightningModule(config)
        ckpt_dir = os.path.join(config["dirs"]["ckpt_dir_tesla"], "lightning")
    else:
        raise ValueError(f"Unknown stage: {stage}. Use: contentnet | prog_4to2 | prog_2to1 | tesla")

    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- Callbacks ----
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=f"{stage}" + "-{epoch:04d}-{val_ssim:.4f}",
        monitor="val_ssim",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=config["training"].get("model_save_step", 5),
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # ---- Trainer ----
    trainer = pl.Trainer(
        max_epochs              = config["training"]["epochs"],
        accelerator             = "gpu",
        devices                 = num_gpus,
        strategy                = strategy,
        logger                  = wandb_logger,
        callbacks               = [checkpoint_cb, lr_monitor],
        check_val_every_n_epoch = 1,
        log_every_n_steps       = 1,
        enable_progress_bar     = True,
        deterministic           = False,
        precision               = "32-true",
    )

    # ---- Train ----
    trainer.fit(model, datamodule=data_module, ckpt_path=args.resume_ckpt)

    print(f"\nTraining complete! Best checkpoint: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
