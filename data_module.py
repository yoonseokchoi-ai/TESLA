import pytorch_lightning as pl
from torch.utils.data import DataLoader
from customdataset_h5_tesla import IXI_Dataset


class TESLADataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule wrapping the existing IXI_Dataset."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Create train/val/test datasets using the existing IXI_Dataset class."""
        # Build a flat config namespace that IXI_Dataset expects
        ds_config = self._build_dataset_config()

        if stage == "fit" or stage is None:
            self.train_dataset = IXI_Dataset(config=ds_config, mode="train")
            self.val_dataset = IXI_Dataset(config=ds_config, mode="test")

        if stage == "test" or stage is None:
            self.test_dataset = IXI_Dataset(config=ds_config, mode="test")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            num_workers=self.config["training"]["workers"],
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["workers"],
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=False,
            num_workers=self.config["training"]["workers"],
            pin_memory=True,
            drop_last=True,
        )

    def _build_dataset_config(self):
        """Convert nested YAML config to a flat namespace object that IXI_Dataset expects."""

        class _Cfg:
            pass

        cfg = _Cfg()
        cfg.dataset = self.config["data"]["dataset"]
        cfg.ixi_h5_1_2mm_dir = self.config["data"]["ixi_h5_dir"]
        cfg.nb_train_imgs = self.config["data"]["nb_train_imgs"]
        cfg.nb_test_imgs = self.config["data"]["nb_test_imgs"]
        cfg.crf_domain = self.config["data"]["crf_domain"]
        cfg.sr_scale = self.config["data"]["sr_scale"]
        cfg.hr_pd = self.config["data"].get("hr_pd", False)
        return cfg
