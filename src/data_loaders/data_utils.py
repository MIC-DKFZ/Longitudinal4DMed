import argparse
import torch
from typing import Tuple
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np

from .acdc_loader import ACDCDataset
from .isles_loader import ISLESDataset


class BGTransformWrapper(Dataset):
    """
    Wraps any longitudinal dataset with batchgenerators spatial + intensity
    augmentations.  All T frames (context + target) are stacked into a single
    multi-channel volume so the SAME spatial transform is applied to every frame.
    its a bit hacky, but it works. 
    """

    def __init__(self, ds: Dataset, tfm) -> None:
        self.dataset = ds
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        s = self.dataset[idx]
        t = s["target_img"]
        c = s["context"]
        # convert to numpy if needed
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        if isinstance(c, torch.Tensor):
            c = c.numpy()
        combined = np.concatenate([t, c], axis=0).astype(np.float32)  # (T+1, C, D, H, W)
        T1, C, D, H, W = combined.shape
        # reshape to (1, T1*C, D, H, W) — same spatial transform for all frames
        out_ch = self.tfm(data=combined.reshape(1, T1 * C, D, H, W))["data"]
        out = out_ch.reshape(T1, C, D, H, W)
        return {
            "target_img": torch.from_numpy(out[[0]]),
            "context":    torch.from_numpy(out[1:]),
            "target_seg":   s["target_seg"],
            "context_seg":  s["context_seg"],
            "target_time":  s["target_time"],
            "context_time": s["context_time"],
        }


def build_aug_transform(image_size: tuple):
    """Build the batchgenerators augmentation pipeline (training only)."""
    from batchgenerators.transforms.spatial_transforms import SpatialTransform
    from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform
    from batchgenerators.transforms.color_transforms import (
        GammaTransform,
        BrightnessMultiplicativeTransform,
        ClipValueRange,
    )
    from batchgenerators.transforms.abstract_transforms import Compose

    return Compose([
        SpatialTransform(
            patch_size=image_size,
            do_elastic_deform=True, alpha=(0., 500.), sigma=(6., 10.),
            do_rotation=True,
            angle_x=(-15 / 180 * np.pi, 15 / 180 * np.pi),
            angle_y=(-15 / 180 * np.pi, 15 / 180 * np.pi),
            angle_z=(-15 / 180 * np.pi, 15 / 180 * np.pi),
            do_scale=False, scale=(0.9, 1.1),
            border_mode_data='nearest',
            order_data=1,
            random_crop=False,
        ),
        GaussianNoiseTransform(noise_variance=(0.0, 0.02), p_per_sample=0.3),
        GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, p_per_sample=0.3),
        BrightnessMultiplicativeTransform(multiplier_range=(0.7, 1.3), per_channel=False, p_per_sample=0.3),
        ClipValueRange(min=0.0, max=1.0),
    ])




class DummyTemporalDataset(Dataset):
    """
    Minimal toy dataset so the script runs out of the box.

    Replace this with your own Dataset that returns:
        x:     [T, C, D, H, W]
        times: [T]
    """

    def __init__(
            self,
            length: int = 8,
            T: int = 4,
            C: int = 1,
            D: int = 32,
            H: int = 32,
            W: int = 32,
    ) -> None:
        super().__init__()
        self.length = length
        self.T = T
        self.C = C
        self.D = D
        self.H = H
        self.W = W

        # fixed normalized times for the toy example
        self._times = torch.linspace(0.0, 1.0, T)

    def __len__(self) -> int:
        return self.length


    def __getitem__(self, idx):
        # synthetic sequence: (T, C, D, H, W)
        x = torch.randn(self.T+1, self.C, self.D, self.H, self.W).clip(0, 1)

        return {
            "target_img": x[[-1]],  # (1, C, D, H, W)
            "context": x[:-1],  # (T-1, C, D, H, W)
            "target_seg": torch.zeros_like(x[[-1]]),
            "context_seg": torch.zeros_like(x[:-1]),
            "target_time": torch.tensor([1.0], dtype=torch.float32),
            "context_time": torch.linspace(0.0, 1.0, self.T+1)[:-1],
        }

    def _get_data_shape(self) -> Tuple[int, int, int, int, int]:
        return (self.T, self.C, self.D, self.H, self.W)

def build_dataloader(args: argparse.Namespace, train_test_val='trn') -> DataLoader:
    if args.dummy:
        dataset = DummyTemporalDataset()
    else:
        if args.dataset == 'acdc':
            data_dir = os.getenv("DATA_DIR", "./data/")
            local_data_path_folder = 'ACDC'
            data_dir = os.path.join(data_dir, local_data_path_folder)
            dataset = ACDCDataset(
                data_dir=data_dir,
                split=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        elif args.dataset == 'isles':
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = ISLESDataset(
                data_dir=data_dir,
                train_test_val=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        elif args.dataset == 'lumiere':
            from .lumiere_loader import LumiereDataset
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = LumiereDataset(
                data_dir=data_dir,
                split=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        elif args.dataset == 'oasis':
            from .oasis_loader import OASISDataset
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = OASISDataset(
                data_dir=data_dir,
                split=train_test_val,
                num_to_keep_context=5,
                **vars(args)
            )
        else:
            raise NotImplementedError(
            "Provide your own dataset or run with --use-dummy-data to test the pipeline."
            )

    if train_test_val == 'trn' and getattr(args, 'augmentation', False):
        image_size = dataset._get_data_shape()[2:]
        dataset = BGTransformWrapper(dataset, build_aug_transform(image_size))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader
