import os
import h5py
import torch
import numpy as np
from torch.utils import data
from monai.transforms import Resize
from skimage.feature import canny
    
class IXI_Dataset(data.Dataset):
    """IXI dataset for multi-contrast MRI super-resolution (TESLA).

    Loads T1 (data_A), PD, T2-HR (data_B_HR), and degraded T2 variants from HDF5.
    """

    def __init__(self, config, mode):
        self.mode = mode
        self.config = config
        self.ixi_h5_1_2mm_dir = config.ixi_h5_1_2mm_dir

        if self.mode == "train":
            self.train_data_dir, self.nb_train_imgs = self.preprocess(config)
        elif self.mode == "test":
            self.test_data_dir, self.nb_test_imgs = self.preprocess(config)

    def preprocess(self, config):
        """Determine H5 file path and number of images for train/test split."""
        data_dir = os.path.join(self.ixi_h5_1_2mm_dir, self.mode, "output_data_1.2mm_2.4mm_4.8mm_50.h5")
        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"

        if self.mode == 'train':
            if config.nb_train_imgs is not None:
                assert isinstance(config.nb_train_imgs, int) and config.nb_train_imgs > 0
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_train_imgs <= len(f["data_B_HR"])
                return data_dir, config.nb_train_imgs
            else:
                with h5py.File(data_dir, "r") as f:
                    return data_dir, len(f["data_B_HR"])

        elif self.mode == 'test':
            if config.nb_test_imgs is not None:
                assert isinstance(config.nb_test_imgs, int) and config.nb_test_imgs > 0
                with h5py.File(data_dir, "r") as f:
                    assert config.nb_test_imgs <= len(f["data_B_HR"])
                return data_dir, config.nb_test_imgs
            else:
                with h5py.File(data_dir, "r") as f:
                    return data_dir, len(f["data_B_HR"])
     
        
    def apply_canny(self, img):
        """Apply Canny edge detection. Input shape: (1, H, W) → Output: list of (H, W) bool arrays."""
        edge = canny(img[0], sigma=1, low_threshold=0.1, high_threshold=0.2)
        return [edge]

    def __getitem__(self, index):
        """Fetch a single sample. Returns dict of tensors (no batch dim)."""
        data_dir = self.train_data_dir if self.mode == 'train' else self.test_data_dir

        os.environ["HDF5_USE_FILE_LOCKING"] = "TRUE"
        with h5py.File(data_dir, "r") as f:
            A         = np.array(f["data_A"][index],         dtype=np.float32).copy()
            PD        = np.array(f["data_PD"][index],        dtype=np.float32).copy()
            B_41      = np.array(f["data_B_41"][index],      dtype=np.float32).copy()
            B_21      = np.array(f["data_B_21"][index],      dtype=np.float32).copy()
            B_HR      = np.array(f["data_B_HR"][index],      dtype=np.float32).copy()
            B_SR_2to1 = np.array(f["data_B_SR_2to1"][index], dtype=np.float32).copy()

        # Downsample HR for data consistency loss
        resize_2fold = Resize(spatial_size=(64, 256), mode="area")
        resize_4fold = Resize(spatial_size=(32, 256), mode="area")
        B_2fold = resize_2fold(B_HR)
        B_4fold = resize_4fold(B_HR)

        result = {
            "data_A":         torch.from_numpy(A),
            "data_PD":        torch.from_numpy(PD),
            "data_B_HR":      torch.from_numpy(B_HR),
            "data_B_41":      torch.from_numpy(B_41),
            "data_B_21":      torch.from_numpy(B_21),
            "data_B_4fold":   torch.from_numpy(np.asarray(B_4fold)),
            "data_B_2fold":   torch.from_numpy(np.asarray(B_2fold)),
            "data_B_SR_2to1": torch.from_numpy(B_SR_2to1),
        }

        # Canny edge conditioning (train only)
        if self.mode == 'train':
            if self.config.crf_domain == "t1":
                cdt_edge = self.apply_canny(A)
            elif self.config.crf_domain == "pd":
                cdt_edge = self.apply_canny(PD)
            elif self.config.crf_domain == "t2":
                cdt_edge = self.apply_canny(B_HR)
            elif self.config.crf_domain == "srt2":
                cdt_edge = self.apply_canny(B_SR_2to1)
            elif self.config.crf_domain == "none":
                cdt_edge = np.zeros_like(A)
            result["data_cdt_edge"] = torch.from_numpy(np.where(np.array(cdt_edge), 1, 0))

        return result


    def __len__(self):
        if self.mode == 'train':
            return self.nb_train_imgs
        elif self.mode == 'test':
            return self.nb_test_imgs