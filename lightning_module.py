import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

from monai.losses.ssim_loss import SSIMLoss
from monai.metrics.regression import SSIMMetric
from monai.metrics import PSNRMetric
from monai.transforms import Resize

from networks_tesla import AdaINGen, PatchGAN_Dis, define_F
from networks_contentnet import AdaINGen as ContentNet_AdaINGen
from utils import weights_init


# ---------------------------------------------------------------------------
# PatchNCE Loss (from trainer_h5_tesla.py)
# ---------------------------------------------------------------------------
class PatchNCELoss(nn.Module):
    def __init__(self, nce_T=0.07, batch_size=8, nce_includes_all_negatives_from_minibatch=False):
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        l_pos = torch.bmm(feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1)

        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T
        loss = self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )
        return loss


# ---------------------------------------------------------------------------
# Helper: flat config namespace for existing modules
# ---------------------------------------------------------------------------
class _FlatCfg:
    """Minimal namespace object that existing network constructors expect."""

    def __init__(self, config: dict):
        model = config["model"]
        training = config["training"]
        nce_cfg = config["nce"]
        loss_cfg = config["loss"]
        dc_cfg = config["dc"]
        data_cfg = config["data"]

        # Model - generator
        self.gen_dim = model["gen_dim"]
        self.gen_mlp_dim = model["gen_mlp_dim"]
        self.gen_style_dim = model["gen_style_dim"]
        self.gen_activ = model["gen_activ"]
        self.gen_n_downsample = model["gen_n_downsample"]
        self.gen_n_res = model["gen_n_res"]
        self.gen_pad_type = model["gen_pad_type"]
        self.input_ch_a = model["input_ch_a"]
        self.input_ch_b = model["input_ch_b"]

        # Model - discriminator
        self.dis_dim = model["dis_dim"]
        self.dis_norm = model["dis_norm"]
        self.dis_activ = model["dis_activ"]
        self.dis_n_layer = model["dis_n_layer"]
        self.dis_gan_type = model["dis_gan_type"]
        self.dis_num_scales = model["dis_num_scales"]
        self.dis_pad_type = model["dis_pad_type"]

        # Training
        self.gen_lr = training["gen_lr"]
        self.dis_lr = training["dis_lr"]
        self.beta1 = training["beta1"]
        self.beta2 = training["beta2"]
        self.weight_decay = training["weight_decay"]
        self.lr_policy = training["lr_policy"]
        self.step_size = training["step_size"]
        self.gamma = training["gamma"]
        self.patience = training["patience"]
        self.factor = training["factor"]
        self.init = training["init"]
        self.batch_size = training["batch_size"]
        self.epochs = training["epochs"]
        self.workers = training["workers"]
        self.generator_steps = training["generator_steps"]
        self.discriminator_steps = training["discriminator_steps"]

        # Loss weights
        self.gan_w = loss_cfg["gan_w"]
        self.recon_l1_x_w = loss_cfg["recon_l1_x_w"]
        self.recon_l1_s_w = loss_cfg["recon_l1_s_w"]
        self.recon_l1_c_w = loss_cfg["recon_l1_c_w"]
        self.recon_l1_cyc_w = loss_cfg["recon_l1_cyc_w"]
        self.recon_ssim_x_w = loss_cfg["recon_ssim_x_w"]
        self.recon_ssim_c_w = loss_cfg["recon_ssim_c_w"]
        self.recon_ssim_cyc_w = loss_cfg["recon_ssim_cyc_w"]
        self.recon_x_cyc_w = loss_cfg["recon_x_cyc_w"]
        self.recon_patchnce_w = loss_cfg["recon_patchnce_w"]
        self.dc_l1_w = loss_cfg["dc_l1_w"]
        self.dc_ssim_w = loss_cfg["dc_ssim_w"]

        # Data consistency
        self.dc_avg = dc_cfg["dc_avg"]
        self.dc_monai = dc_cfg["dc_monai"]
        self.dc_monai_method = dc_cfg["dc_monai_method"]

        # NCE
        self.nce = nce_cfg["nce"]
        self.nce_idt = nce_cfg["nce_idt"]
        self.nce_layers = nce_cfg["nce_layers"]
        self.nce_T = nce_cfg["nce_T"]
        self.lambda_NCE = nce_cfg["lambda_NCE"]
        self.num_patches = nce_cfg["num_patches"]
        self.netF = nce_cfg["netF"]
        self.netF_nc = nce_cfg["netF_nc"]
        self.nce_includes_all_negatives_from_minibatch = nce_cfg[
            "nce_includes_all_negatives_from_minibatch"
        ]

        # Data
        self.dataset = data_cfg["dataset"]
        self.hr_pd = data_cfg.get("hr_pd", False)
        self.sr_scale = data_cfg.get("sr_scale", 4)
        self.crf_domain = data_cfg.get("crf_domain", "t1")

        # Misc (needed by some existing functions)
        self.device = config.get("device", "0")
        self.test_epochs = 100
        self.train_dataset = True  # flag for ContentNet loading


# ===========================================================================
# Stage 2: TESLA Lightning Module
# ===========================================================================
class TESLALightningModule(pl.LightningModule):
    """PyTorch Lightning module for TESLA (Stage 2) training."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.cfg = _FlatCfg(config)
        self.automatic_optimization = False

        # ---- Build networks ----
        self.gen_b = AdaINGen(self.cfg.input_ch_b, self.cfg)
        self.dis_b = PatchGAN_Dis()
        self.net_F = define_F(
            input_nc=self.cfg.input_ch_b,
            netF=self.cfg.netF,
            norm="batch",
            use_dropout=False,
            init_type="normal",
            init_gain=0.02,
            no_antialias=False,
            gpu_ids=self.cfg.device,
            opt=self.cfg,
        )

        # ---- ContentNet (frozen, loaded from stage1 ckpt) ----
        self.contentnet_gen_a = ContentNet_AdaINGen(self.cfg.input_ch_a, self.cfg)
        self._load_and_freeze_contentnet()

        # ---- Progressive SR (frozen, loaded from progressive ckpts) ----
        prog_cfg = config.get("progressive", {})
        self.prog_4to2 = AdaINGen(self.cfg.input_ch_b, self.cfg)
        self.prog_2to1 = AdaINGen(self.cfg.input_ch_b, self.cfg)
        self._load_and_freeze_progressive(
            prog_4to2_path=prog_cfg.get("prog_4to2_ckpt_path"),
            prog_2to1_path=prog_cfg.get("prog_2to1_ckpt_path"),
        )

        # ---- Losses ----
        self.ganloss = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.SSIMloss = SSIMLoss(spatial_dims=2, data_range=1)

        self.PatchNCEloss = nn.ModuleList()
        for _ in self.cfg.nce_layers:
            self.PatchNCEloss.append(
                PatchNCELoss(
                    nce_T=self.cfg.nce_T,
                    batch_size=self.cfg.batch_size,
                    nce_includes_all_negatives_from_minibatch=self.cfg.nce_includes_all_negatives_from_minibatch,
                )
            )

        # ---- Metrics ----
        self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
        self.psnr_metric = PSNRMetric(max_val=1.0)

        # ---- Weight init ----
        self.gen_b.apply(weights_init(self.cfg.init))
        self.dis_b.apply(weights_init("gaussian"))
        # netF is initialized inside define_F

        # ---- Epoch-level accumulators ----
        self._train_loss_accum = {}
        self._train_loss_count = 0
        self._val_ssim_sum = 0.0
        self._val_psnr_sum = 0.0
        self._val_n_samples = 0
        self._val_images = []

        # W&B config
        wandb_cfg = config.get("wandb", {})
        self.num_display = wandb_cfg.get("num_display_images", 4)
        self.log_images_every = wandb_cfg.get("log_images_every_n_epochs", 1)

    # ---------------------------------------------------------------
    # ContentNet loading & freezing
    # ---------------------------------------------------------------
    def _load_and_freeze_contentnet(self):
        ckpt_path = self.config.get("contentnet_ckpt_path", None)
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                # Lightning checkpoint
                state = {k.replace("gen_a.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("gen_a.")}
                self.contentnet_gen_a.load_state_dict(state)
            elif "a" in ckpt:
                self.contentnet_gen_a.load_state_dict(ckpt["a"])
            elif "model" in ckpt:
                self.contentnet_gen_a.load_state_dict(ckpt["model"])
            print(f"[TESLA] Loaded ContentNet from {ckpt_path}")
        else:
            print(f"[TESLA] WARNING: ContentNet ckpt not found at {ckpt_path}")

        # Freeze ContentNet
        self.contentnet_gen_a.eval()
        for p in self.contentnet_gen_a.parameters():
            p.requires_grad = False

    def _load_and_freeze_progressive(self, prog_4to2_path, prog_2to1_path):
        for name, model, path in [
            ("prog_4to2", self.prog_4to2, prog_4to2_path),
            ("prog_2to1", self.prog_2to1, prog_2to1_path),
        ]:
            if path and os.path.isfile(path):
                ckpt = torch.load(path, map_location="cpu")
                if "state_dict" in ckpt:
                    state = {k.replace("gen.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("gen.")}
                    model.load_state_dict(state)
                else:
                    model.load_state_dict(ckpt)
                print(f"[TESLA] Loaded frozen {name} from {path}")
            else:
                print(f"[TESLA] WARNING: {name} ckpt not found at {path}")
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

    def _progressive_forward(self, x_b_41):
        """Frozen progressive pipeline: x_b_41 → prog_4to2 → prog_2to1 → x_b_sr."""
        with torch.no_grad():
            c, s = self.prog_4to2.encode(x_b_41)
            x_4to2 = self.prog_4to2.decode(c[-1], s)
            c, s = self.prog_2to1.encode(x_4to2)
            x_2to1 = self.prog_2to1.decode(c[-1], s)
        return x_2to1

    # ---------------------------------------------------------------
    # Optimizers
    # ---------------------------------------------------------------
    def configure_optimizers(self):
        cfg = self.cfg
        opt_d = torch.optim.Adam(
            [p for p in self.dis_b.parameters() if p.requires_grad],
            lr=cfg.dis_lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        opt_g = torch.optim.Adam(
            list(self.gen_b.parameters()) + list(self.net_F.parameters()),
            lr=cfg.gen_lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

        sched_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=cfg.step_size, gamma=cfg.gamma)
        sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=cfg.step_size, gamma=cfg.gamma)

        return (
            [opt_d, opt_g],
            [{"scheduler": sched_d, "interval": "epoch"}, {"scheduler": sched_g, "interval": "epoch"}],
        )

    # ---------------------------------------------------------------
    # Training step (manual optimization)
    # ---------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        cfg = self.cfg

        # Unpack batch (TESLA paper notation: x_b = target T2, x_a = reference T1/PD)
        x_b_sr_2to1 = self._progressive_forward(batch["data_B_41"])  # Progressive SR input
        x_b_hr = batch["data_B_HR"]             # HR ground truth T2
        x_b_4fold = batch["data_B_4fold"]        # 4x downsampled T2 (for data consistency)
        canny_edge = batch["data_cdt_edge"].float()
        x_a = batch["data_PD"] if cfg.hr_pd else batch["data_A"]  # Reference image

        # PatchGAN patch size
        patch = (1, x_b_sr_2to1.shape[-2] // 2**4, x_b_sr_2to1.shape[-1] // 2**4)
        real_label = torch.ones(x_b_sr_2to1.size(0), *patch, device=self.device)
        fake_label = torch.zeros(x_b_sr_2to1.size(0), *patch, device=self.device)

        # ==================== Discriminator ====================
        for _ in range(cfg.discriminator_steps):
            c_b, s_b = self.gen_b.encode(x_b_sr_2to1)
            x_b_recon = self.gen_b.decode(c_b[-1], s_b)

            pred_real = self.dis_b(img=x_b_sr_2to1, img_condition=canny_edge)
            pred_fake = self.dis_b(img=x_b_recon.detach(), img_condition=canny_edge)

            loss_d = cfg.gan_w * 0.5 * (self.ganloss(pred_real, real_label) + self.ganloss(pred_fake, fake_label))

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

        # ==================== Generator ====================
        for _ in range(cfg.generator_steps):
            c_b, s_b_prime = self.gen_b.encode(x_b_sr_2to1)
            x_b_recon = self.gen_b.decode(c_b[-1], s_b_prime)
            c_b_recon, s_b_recon = self.gen_b.encode(x_b_recon)

            # Cycle reconstruction (optional)
            x_b_cyc = self.gen_b.decode(c_b_recon[-1], s_b_recon) if cfg.recon_x_cyc_w > 0 else None

            # Data consistency: ||downsample(x_SR) - x_LR||
            src_shape = x_b_recon.shape
            tgt_shape = x_b_4fold.shape
            if cfg.dc_avg:
                dc_1to4 = x_b_recon.reshape(-1, 1, tgt_shape[-2], int(src_shape[-2] / tgt_shape[-2]), src_shape[-1]).mean(axis=3)
            elif cfg.dc_monai:
                resize_4fold = Resize(spatial_size=(tgt_shape[-2], src_shape[-1]), mode=cfg.dc_monai_method)
                dc_1to4 = torch.zeros_like(x_b_4fold)
                for b in range(x_b_recon.shape[0]):
                    dc_1to4[b] = resize_4fold(x_b_recon[b])

            loss_dc_l1 = cfg.dc_l1_w * self.L1loss(dc_1to4, x_b_4fold)
            loss_dc_ssim = cfg.dc_ssim_w * self.SSIMloss(dc_1to4, x_b_4fold)

            # Adversarial loss
            pred_fake = self.dis_b(img=x_b_recon, img_condition=canny_edge)
            loss_adv = cfg.gan_w * self.ganloss(pred_fake, real_label)

            # Reconstruction losses
            loss_l1_x = cfg.recon_l1_x_w * self.L1loss(x_b_recon, x_b_hr)
            loss_l1_s = cfg.recon_l1_s_w * self.L1loss(s_b_recon, s_b_prime)
            loss_l1_c = cfg.recon_l1_c_w * self.L1loss(c_b_recon[-1], c_b[-1])
            loss_l1_cyc = cfg.recon_l1_cyc_w * self.L1loss(x_b_cyc, x_b_sr_2to1) if cfg.recon_x_cyc_w > 0 else 0

            loss_ssim_x = cfg.recon_ssim_x_w * self.SSIMloss(x_b_recon, x_b_hr)
            loss_ssim_c = cfg.recon_ssim_c_w * self.SSIMloss(c_b_recon[-1], c_b[-1])
            loss_ssim_cyc = cfg.recon_ssim_cyc_w * self.SSIMloss(x_b_cyc, x_b_sr_2to1) if cfg.recon_x_cyc_w > 0 else 0

            # PatchNCE loss (cross-modal contrastive learning)
            loss_nce = torch.tensor(0.0, device=self.device)
            if cfg.nce:
                loss_nce = cfg.recon_patchnce_w * self._compute_nce_loss(x_a, x_b_sr_2to1, nce_idt=False)
            if cfg.nce_idt and cfg.recon_patchnce_w > 0:
                loss_nce_src = cfg.recon_patchnce_w * self._compute_nce_loss(x_a, x_b_sr_2to1, nce_idt=False)
                loss_nce_idt = cfg.recon_patchnce_w * self._compute_nce_loss(x_b_sr_2to1, x_b_sr_2to1, nce_idt=True)
                loss_nce = 0.5 * (loss_nce_src + loss_nce_idt)

            # Total generator loss
            loss_g = (
                loss_adv + loss_l1_x + loss_l1_s + loss_l1_c + loss_l1_cyc
                + loss_ssim_x + loss_ssim_c + loss_ssim_cyc
                + loss_nce + loss_dc_l1 + loss_dc_ssim
            )

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

        # ---- Accumulate epoch losses ----
        loss_dict = {
            "D_loss/total": loss_d,
            "G_loss/total": loss_g,
            "G_loss/adv": loss_adv,
            "G_loss/L1_x": loss_l1_x,
            "G_loss/L1_s": loss_l1_s,
            "G_loss/L1_c": loss_l1_c,
            "G_loss/SSIM_x": loss_ssim_x,
            "G_loss/SSIM_c": loss_ssim_c,
            "G_loss/PatchNCE": loss_nce,
            "G_loss/DC_L1": loss_dc_l1,
            "G_loss/DC_SSIM": loss_dc_ssim,
        }
        self._accumulate_losses(loss_dict)

    def _compute_nce_loss(self, x_a, x_b, nce_idt=False):
        """Compute PatchNCE loss: cross-modal contrastive learning between reference (x_a) and target (x_b)."""
        feat_q, _ = self.gen_b.encode(x_b)
        n_layers = len(feat_q)

        if nce_idt:
            feat_k, _ = self.gen_b.encode(x_a)
        else:
            self.contentnet_gen_a.eval()
            feat_k, _ = self.contentnet_gen_a.encode(x_a)

        feat_k_pool, sample_ids = self.net_F(feat_k, self.cfg.num_patches, None)
        feat_q_pool, _ = self.net_F(feat_q, self.cfg.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.PatchNCEloss):
            loss = crit(f_q, f_k) * self.cfg.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    def _accumulate_losses(self, loss_dict):
        self._train_loss_count += 1
        for k, v in loss_dict.items():
            val = v.detach() if isinstance(v, torch.Tensor) else v
            if k not in self._train_loss_accum:
                self._train_loss_accum[k] = val
            else:
                self._train_loss_accum[k] += val

    # ---------------------------------------------------------------
    # End of training epoch — log averaged losses
    # ---------------------------------------------------------------
    def on_train_epoch_start(self):
        self._train_loss_accum = {}
        self._train_loss_count = 0

    def on_train_epoch_end(self):
        if self._train_loss_count == 0:
            return
        for k, v in self._train_loss_accum.items():
            avg = v / self._train_loss_count
            self.log(k, avg, prog_bar=(k in ("G_loss/total", "D_loss/total")), sync_dist=True)

        # Log learning rates
        opt_d, opt_g = self.optimizers()
        self.log("lr/discriminator", opt_d.param_groups[0]["lr"])
        self.log("lr/generator", opt_g.param_groups[0]["lr"])

        # Step schedulers
        for sched in self.lr_schedulers():
            sched.step()

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------
    def on_validation_epoch_start(self):
        self._val_ssim_sum = 0.0
        self._val_psnr_sum = 0.0
        self._val_n_samples = 0
        self._val_images = []

    def validation_step(self, batch, batch_idx):
        x_b_sr_2to1 = self._progressive_forward(batch["data_B_41"])
        x_b_hr = batch["data_B_HR"]
        x_a = batch["data_PD"] if self.cfg.hr_pd else batch["data_A"]

        # Forward: encode LR input → content + style → decode → SR output
        c_b, s_b = self.gen_b.encode(x_b_sr_2to1)
        x_b_recon = self.gen_b.decode(c_b[-1], s_b)

        # Metrics
        ssim_val = self.ssim_metric(y_pred=x_b_recon, y=x_b_hr)
        psnr_val = self.psnr_metric(y_pred=x_b_recon, y=x_b_hr)

        self._val_ssim_sum += torch.sum(ssim_val).item()
        self._val_psnr_sum += torch.sum(psnr_val).item()
        self._val_n_samples += x_b_hr.size(0)

        # Collect images for logging (only from first batch)
        if batch_idx == 0:
            n = min(self.num_display, x_b_hr.size(0))
            self._val_images = {
                "x_a_ref": x_a[:n].cpu(),
                "x_b_lr": x_b_sr_2to1[:n].cpu(),
                "x_b_sr": x_b_recon[:n].cpu(),
                "x_b_hr": x_b_hr[:n].cpu(),
            }

    def on_validation_epoch_end(self):
        if self._val_n_samples == 0:
            return

        mean_ssim = self._val_ssim_sum / self._val_n_samples
        mean_psnr = self._val_psnr_sum / self._val_n_samples

        self.log("val_ssim", mean_ssim, prog_bar=True, sync_dist=True)
        self.log("val_psnr", mean_psnr, prog_bar=True, sync_dist=True)

        # Log images to W&B
        if (
            self._val_images
            and self.logger
            and (self.current_epoch + 1) % self.log_images_every == 0
        ):
            self._log_wandb_images()

    def _log_wandb_images(self):
        """Log x_a(ref) | x_b_lr | x_b_sr | x_b_hr | error as a single image grid to W&B."""
        imgs = self._val_images
        x_a_ref = imgs["x_a_ref"]   # (N, 1, H, W)
        x_b_lr = imgs["x_b_lr"]     # (N, 1, H, W)
        x_b_sr = imgs["x_b_sr"]     # (N, 1, H, W)
        x_b_hr = imgs["x_b_hr"]     # (N, 1, H, W)

        error_map = torch.abs(x_b_sr - x_b_hr)

        rows = []
        n = x_b_lr.size(0)
        for i in range(n):
            row = torch.cat([x_a_ref[i], x_b_lr[i], x_b_sr[i], x_b_hr[i], error_map[i]], dim=-1)
            rows.append(row)
        grid = torch.cat(rows, dim=-2).clamp(0, 1)

        grid_np = grid.squeeze(0).numpy()

        caption = "x_a(Ref) | x_b_LR | x_b_SR | x_b_HR | |SR-HR|"
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {"val/images": wandb.Image(grid_np, caption=caption)},
                step=self.global_step,
            )

    # ---------------------------------------------------------------
    # Test step (same as validation)
    # ---------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()


# ===========================================================================
# Progressive Through-plane SR (ProTPSR) Lightning Module
# ===========================================================================
class ProgressiveReconModule(pl.LightningModule):
    """PyTorch Lightning module for progressive through-plane SR (4x→2x→1x).

    Stages:
      - prog_4to2: Train AdaINGen to reconstruct x_b_21 from x_b_41 (4x→2x).
      - prog_2to1: Freeze prog_4to2, train a second AdaINGen to reconstruct
                    x_b_hr from prog_4to2(x_b_41) (2x→1x).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.cfg = _FlatCfg(config)
        self.automatic_optimization = False

        stage = str(config.get("stage", "prog_4to2"))
        assert stage in ("prog_4to2", "prog_2to1"), f"Invalid progressive stage: {stage}"
        self.prog_stage = stage

        prog_cfg       = config.get("progressive", {})
        self.l1_w      = prog_cfg.get("l1_w", 10.0)
        self.ssim_w    = prog_cfg.get("ssim_w", 1.0)
        self.dc_l1_w   = prog_cfg.get("dc_l1_w", 1.0)
        self.dc_ssim_w = prog_cfg.get("dc_ssim_w", 0.1)

        # Current-stage generator
        self.gen = AdaINGen(self.cfg.input_ch_b, self.cfg)
        self.gen.apply(weights_init(self.cfg.init))

        # Frozen upstream generator for prog_2to1
        self.gen_4to2 = None
        if self.prog_stage == "prog_2to1":
            self.gen_4to2 = AdaINGen(self.cfg.input_ch_b, self.cfg)
            self._load_and_freeze_4to2(prog_cfg.get("prog_4to2_ckpt_path"))

        # Losses & metrics
        self.L1loss      = nn.L1Loss()
        self.SSIMloss    = SSIMLoss(spatial_dims=2, data_range=1)
        self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
        self.psnr_metric = PSNRMetric(max_val=1.0)

        # Epoch accumulators
        self._train_loss_accum = {}
        self._train_loss_count = 0
        self._val_ssim_sum = 0.0
        self._val_psnr_sum = 0.0
        self._val_n_samples = 0
        self._val_images = []

        wandb_cfg = config.get("wandb", {})
        self.num_display = wandb_cfg.get("num_display_images", 4)
        self.log_images_every = wandb_cfg.get("log_images_every_n_epochs", 1)

    def _load_and_freeze_4to2(self, ckpt_path):
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            # Support both Lightning checkpoint and raw state_dict
            if "state_dict" in ckpt:
                state = {k.replace("gen.", ""): v for k, v in ckpt["state_dict"].items() if k.startswith("gen.")}
                self.gen_4to2.load_state_dict(state)
            else:
                self.gen_4to2.load_state_dict(ckpt)
            print(f"[ProTPSR] Loaded frozen prog_4to2 from {ckpt_path}")
        else:
            print(f"[ProTPSR] WARNING: prog_4to2 ckpt not found at {ckpt_path}")
        self.gen_4to2.eval()
        for p in self.gen_4to2.parameters():
            p.requires_grad = False

    def _get_input_target(self, batch):
        """Return (input, target, degraded_for_dc) based on progressive stage."""
        if self.prog_stage == "prog_4to2":
            # Input: 4x degraded (bilinear upsampled to 2x size), Target: 2x data
            x_input  = batch["data_B_41"]       # (B, 1, 32, 256) → upsample to (64, 256)
            x_target = batch["data_B_21"]      # (B, 1, 64, 256)
            x_dc_ref = batch["data_B_4fold"]   # (B, 1, 32, 256) for data consistency
            return x_input, x_target, x_dc_ref
        else:
            # prog_2to1: frozen 4to2 output → train 2to1
            x_b_41 = batch["data_B_41"]
            with torch.no_grad():
                c, s = self.gen_4to2.encode(x_b_41)
                x_input = self.gen_4to2.decode(c[-1], s)  # prog_4to2 output
            x_target = batch["data_B_HR"]       # (B, 1, 128, 256)
            x_dc_ref = batch["data_B_2fold"]    # (B, 1, 64, 256) for data consistency
            return x_input, x_target, x_dc_ref

    def configure_optimizers(self):
        cfg = self.cfg
        opt_g = torch.optim.Adam(
            [p for p in self.gen.parameters() if p.requires_grad],
            lr=cfg.gen_lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=cfg.step_size, gamma=cfg.gamma)
        return [opt_g], [{"scheduler": sched_g, "interval": "epoch"}]

    def training_step(self, batch, batch_idx):
        opt_g = self.optimizers()
        x_input, x_target, x_dc_ref = self._get_input_target(batch)

        c, s = self.gen.encode(x_input)
        x_recon = self.gen.decode(c[-1], s)

        # Reconstruction losses
        loss_l1 = self.l1_w * self.L1loss(x_recon, x_target)
        loss_ssim = self.ssim_w * self.SSIMloss(x_recon, x_target)

        # Data consistency: downsample SR output → compare with degraded input
        dc_size = (x_dc_ref.shape[-2], x_dc_ref.shape[-1])
        resize_dc = Resize(spatial_size=dc_size, mode=self.cfg.dc_monai_method)
        dc_down = torch.zeros_like(x_dc_ref)
        for b in range(x_recon.shape[0]):
            dc_down[b] = resize_dc(x_recon[b])
        loss_dc_l1 = self.dc_l1_w * self.L1loss(dc_down, x_dc_ref)
        loss_dc_ssim = self.dc_ssim_w * self.SSIMloss(dc_down, x_dc_ref)

        loss_g = loss_l1 + loss_ssim + loss_dc_l1 + loss_dc_ssim

        opt_g.zero_grad()
        self.manual_backward(loss_g)
        opt_g.step()

        loss_dict = {
            "G_loss/total": loss_g,
            "G_loss/L1": loss_l1,
            "G_loss/SSIM": loss_ssim,
            "G_loss/DC_L1": loss_dc_l1,
            "G_loss/DC_SSIM": loss_dc_ssim,
        }
        self._accumulate_losses(loss_dict)

    def _accumulate_losses(self, loss_dict):
        self._train_loss_count += 1
        for k, v in loss_dict.items():
            val = v.detach() if isinstance(v, torch.Tensor) else v
            if k not in self._train_loss_accum:
                self._train_loss_accum[k] = val
            else:
                self._train_loss_accum[k] += val

    def on_train_epoch_start(self):
        self._train_loss_accum = {}
        self._train_loss_count = 0

    def on_train_epoch_end(self):
        if self._train_loss_count == 0:
            return
        for k, v in self._train_loss_accum.items():
            avg = v / self._train_loss_count
            self.log(k, avg, prog_bar=(k == "G_loss/total"), sync_dist=True)
        opt_g = self.optimizers()
        self.log("lr/generator", opt_g.param_groups[0]["lr"])
        self.lr_schedulers().step()

    def on_validation_epoch_start(self):
        self._val_ssim_sum = 0.0
        self._val_psnr_sum = 0.0
        self._val_n_samples = 0
        self._val_images = []

    def validation_step(self, batch, batch_idx):
        x_input, x_target, _ = self._get_input_target(batch)
        c, s = self.gen.encode(x_input)
        x_recon = self.gen.decode(c[-1], s)

        ssim_val = self.ssim_metric(y_pred=x_recon, y=x_target)
        psnr_val = self.psnr_metric(y_pred=x_recon, y=x_target)
        self._val_ssim_sum += torch.sum(ssim_val).item()
        self._val_psnr_sum += torch.sum(psnr_val).item()
        self._val_n_samples += x_target.size(0)

        if batch_idx == 0:
            n = min(self.num_display, x_target.size(0))
            self._val_images = {
                "x_input": x_input[:n].cpu(),
                "x_recon": x_recon[:n].cpu(),
                "x_target": x_target[:n].cpu(),
            }

    def on_validation_epoch_end(self):
        if self._val_n_samples == 0:
            return
        mean_ssim = self._val_ssim_sum / self._val_n_samples
        mean_psnr = self._val_psnr_sum / self._val_n_samples
        self.log("val_ssim", mean_ssim, prog_bar=True, sync_dist=True)
        self.log("val_psnr", mean_psnr, prog_bar=True, sync_dist=True)

        if (
            self._val_images
            and self.logger
            and (self.current_epoch + 1) % self.log_images_every == 0
        ):
            self._log_wandb_images()

    def _log_wandb_images(self):
        imgs = self._val_images
        x_in = imgs["x_input"]
        x_sr = imgs["x_recon"]
        x_gt = imgs["x_target"]
        error_map = torch.abs(x_sr - x_gt)

        # Resize input to match target height for visualization
        if x_in.shape[-2] != x_gt.shape[-2]:
            resize_vis = Resize(spatial_size=(x_gt.shape[-2], x_gt.shape[-1]), mode="bilinear")
            x_in = torch.stack([resize_vis(x_in[i]) for i in range(x_in.size(0))])

        rows = []
        for i in range(x_in.size(0)):
            row = torch.cat([x_in[i], x_sr[i], x_gt[i], error_map[i]], dim=-1)
            rows.append(row)
        grid = torch.cat(rows, dim=-2).clamp(0, 1)
        grid_np = grid.squeeze(0).numpy()

        caption = f"Input | SR | Target | |SR-Target| ({self.prog_stage})"
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {"val/images": wandb.Image(grid_np, caption=caption)},
                step=self.global_step,
            )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        return self.on_validation_epoch_end()


# ===========================================================================
# Stage 1: ContentNet Lightning Module
# ===========================================================================
class ContentNetLightningModule(pl.LightningModule):
    """PyTorch Lightning module for ContentNet (Stage 1) pre-training."""

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.cfg = _FlatCfg(config)
        self.automatic_optimization = False

        # ---- Build networks ----
        self.gen_a = ContentNet_AdaINGen(self.cfg.input_ch_a, self.cfg)
        self.dis_a = PatchGAN_Dis()

        # ---- Losses ----
        self.ganloss = nn.MSELoss()
        self.L1loss = nn.L1Loss()
        self.SSIMloss = SSIMLoss(spatial_dims=2, data_range=1)

        # ---- Metrics ----
        self.ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
        self.psnr_metric = PSNRMetric(max_val=1.0)

        # ---- Weight init ----
        self.gen_a.apply(weights_init(self.cfg.init))
        self.dis_a.apply(weights_init("gaussian"))

        # ---- Epoch accumulators ----
        self._train_loss_accum = {}
        self._train_loss_count = 0
        self._val_ssim_sum = 0.0
        self._val_psnr_sum = 0.0
        self._val_n_samples = 0
        self._val_images = []

        wandb_cfg = config.get("wandb", {})
        self.num_display = wandb_cfg.get("num_display_images", 4)
        self.log_images_every = wandb_cfg.get("log_images_every_n_epochs", 1)

    def configure_optimizers(self):
        cfg = self.cfg
        opt_d = torch.optim.Adam(
            [p for p in self.dis_a.parameters() if p.requires_grad],
            lr=cfg.dis_lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        opt_g = torch.optim.Adam(
            [p for p in self.gen_a.parameters() if p.requires_grad],
            lr=cfg.gen_lr,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )
        sched_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=cfg.step_size, gamma=cfg.gamma)
        sched_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=cfg.step_size, gamma=cfg.gamma)
        return (
            [opt_d, opt_g],
            [{"scheduler": sched_d, "interval": "epoch"}, {"scheduler": sched_g, "interval": "epoch"}],
        )

    def training_step(self, batch, batch_idx):
        opt_d, opt_g = self.optimizers()
        cfg = self.cfg

        # ContentNet pre-training on reference domain (T1 or PD)
        x_a = batch["data_PD"] if cfg.hr_pd else batch["data_A"]

        if "data_cdt_edge" in batch:
            canny_edge = batch["data_cdt_edge"].float()
        else:
            canny_edge = torch.zeros_like(x_a)

        patch = (1, x_a.shape[-2] // 2**4, x_a.shape[-1] // 2**4)
        real_label = torch.ones(x_a.size(0), *patch, device=self.device)
        fake_label = torch.zeros(x_a.size(0), *patch, device=self.device)

        # ==================== Discriminator ====================
        for _ in range(cfg.discriminator_steps):
            c_a, s_a = self.gen_a.encode(x_a)
            x_a_recon = self.gen_a.decode(c_a[-1], s_a)

            pred_real = self.dis_a(img=x_a, img_condition=canny_edge)
            pred_fake = self.dis_a(img=x_a_recon.detach(), img_condition=canny_edge)

            loss_d = cfg.gan_w * 0.5 * (self.ganloss(pred_real, real_label) + self.ganloss(pred_fake, fake_label))

            opt_d.zero_grad()
            self.manual_backward(loss_d)
            opt_d.step()

        # ==================== Generator ====================
        for _ in range(cfg.generator_steps):
            c_a, s_a_prime = self.gen_a.encode(x_a)
            x_a_recon = self.gen_a.decode(c_a[-1], s_a_prime)
            c_a_recon, s_a_recon = self.gen_a.encode(x_a_recon)

            x_a_cyc = self.gen_a.decode(c_a_recon[-1], s_a_recon) if cfg.recon_x_cyc_w > 0 else None

            # Adversarial
            pred_fake = self.dis_a(img=x_a_recon, img_condition=canny_edge)
            loss_adv = cfg.gan_w * self.ganloss(pred_fake, real_label)

            # Reconstruction losses
            loss_l1_x = cfg.recon_l1_x_w * self.L1loss(x_a_recon, x_a)
            loss_l1_s = cfg.recon_l1_s_w * self.L1loss(s_a_recon, s_a_prime)
            loss_l1_c = cfg.recon_l1_c_w * self.L1loss(c_a_recon[-1], c_a[-1])
            loss_l1_cyc = cfg.recon_l1_cyc_w * self.L1loss(x_a_cyc, x_a) if cfg.recon_x_cyc_w > 0 else 0

            loss_ssim_x = cfg.recon_ssim_x_w * self.SSIMloss(x_a_recon, x_a)
            loss_ssim_c = cfg.recon_ssim_c_w * self.SSIMloss(c_a_recon[-1], c_a[-1])
            loss_ssim_cyc = cfg.recon_ssim_cyc_w * self.SSIMloss(x_a_cyc, x_a) if cfg.recon_x_cyc_w > 0 else 0

            loss_g = loss_adv + loss_l1_x + loss_l1_s + loss_l1_c + loss_l1_cyc + loss_ssim_x + loss_ssim_c + loss_ssim_cyc

            opt_g.zero_grad()
            self.manual_backward(loss_g)
            opt_g.step()

        loss_dict = {
            "D_loss/total": loss_d,
            "G_loss/total": loss_g,
            "G_loss/adv": loss_adv,
            "G_loss/L1_x": loss_l1_x,
            "G_loss/L1_s": loss_l1_s,
            "G_loss/L1_c": loss_l1_c,
            "G_loss/SSIM_x": loss_ssim_x,
            "G_loss/SSIM_c": loss_ssim_c,
        }
        self._accumulate_losses(loss_dict)

    def _accumulate_losses(self, loss_dict):
        self._train_loss_count += 1
        for k, v in loss_dict.items():
            val = v.detach() if isinstance(v, torch.Tensor) else v
            if k not in self._train_loss_accum:
                self._train_loss_accum[k] = val
            else:
                self._train_loss_accum[k] += val

    def on_train_epoch_start(self):
        self._train_loss_accum = {}
        self._train_loss_count = 0

    def on_train_epoch_end(self):
        if self._train_loss_count == 0:
            return
        for k, v in self._train_loss_accum.items():
            avg = v / self._train_loss_count
            self.log(k, avg, prog_bar=(k in ("G_loss/total", "D_loss/total")), sync_dist=True)

        opt_d, opt_g = self.optimizers()
        self.log("lr/discriminator", opt_d.param_groups[0]["lr"])
        self.log("lr/generator", opt_g.param_groups[0]["lr"])

        for sched in self.lr_schedulers():
            sched.step()

    def on_validation_epoch_start(self):
        self._val_ssim_sum = 0.0
        self._val_psnr_sum = 0.0
        self._val_n_samples = 0
        self._val_images = []

    def validation_step(self, batch, batch_idx):
        x_a = batch["data_PD"] if self.cfg.hr_pd else batch["data_A"]

        c_a, s_a = self.gen_a.encode(x_a)
        x_a_recon = self.gen_a.decode(c_a[-1], s_a)

        ssim_val = self.ssim_metric(y_pred=x_a_recon, y=x_a)
        psnr_val = self.psnr_metric(y_pred=x_a_recon, y=x_a)

        self._val_ssim_sum += torch.sum(ssim_val).item()
        self._val_psnr_sum += torch.sum(psnr_val).item()
        self._val_n_samples += x_a.size(0)

        if batch_idx == 0:
            n = min(self.num_display, x_a.size(0))
            self._val_images = {
                "x_a": x_a[:n].cpu(),
                "x_a_recon": x_a_recon[:n].cpu(),
            }

    def on_validation_epoch_end(self):
        if self._val_n_samples == 0:
            return

        mean_ssim = self._val_ssim_sum / self._val_n_samples
        mean_psnr = self._val_psnr_sum / self._val_n_samples

        self.log("val_ssim", mean_ssim, prog_bar=True, sync_dist=True)
        self.log("val_psnr", mean_psnr, prog_bar=True, sync_dist=True)

        if (
            self._val_images
            and self.logger
            and (self.current_epoch + 1) % self.log_images_every == 0
        ):
            self._log_wandb_images()

    def _log_wandb_images(self):
        imgs = self._val_images
        x_a = imgs["x_a"]
        x_a_recon = imgs["x_a_recon"]
        error_map = torch.abs(x_a_recon - x_a)

        rows = []
        n = x_a.size(0)
        for i in range(n):
            row = torch.cat([x_a[i], x_a_recon[i], error_map[i]], dim=-1)
            rows.append(row)
        grid = torch.cat(rows, dim=-2).clamp(0, 1)
        grid_np = grid.squeeze(0).numpy()

        caption = "x_a | x_a_recon | |Recon-Input|"
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log(
                {"val/images": wandb.Image(grid_np, caption=caption)},
                step=self.global_step,
            )
