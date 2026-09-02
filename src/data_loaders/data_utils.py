import argparse
import sys
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

    def _get_data_shape(self):
        return self.dataset._get_data_shape()

    def __getitem__(self, idx: int) -> dict:
        s = self.dataset[idx]

        """
        target seg and context seg must go through saqme spatial transform as images. 
        therefore we pass them through the seg= arg. 
        Note that the seg shape may not be uniform across datasets. 
        """

        def _to_numpy(x):
            return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

        t, c = _to_numpy(s["target_img"]), _to_numpy(s["context"])
        t_seg, c_seg = _to_numpy(s["target_seg"]), _to_numpy(s["context_seg"])

        combined = np.concatenate([t, c], axis=0).astype(np.float32)  # (T+1, C, D, H, W)
        T1, C, D, H, W = combined.shape
        
        t_seg_shape, c_seg_shape = t_seg.shape, c_seg.shape
        t_seg_flat = t_seg.astype(np.float32).reshape(-1, D, H, W)
        c_seg_flat = c_seg.astype(np.float32).reshape(-1, D, H, W)
        seg_combined = np.concatenate([t_seg_flat, c_seg_flat], axis=0)[None]  # (1, K, D, H, W)
        k_t = t_seg_flat.shape[0]

        result = self.tfm(data=combined.reshape(1, T1 * C, D, H, W), seg=seg_combined)
        out = result["data"].reshape(T1, C, D, H, W)
        seg_out = result["seg"][0]  # (K, D, H, W)
        target_seg = seg_out[:k_t].reshape(t_seg_shape)
        context_seg = seg_out[k_t:].reshape(c_seg_shape)

        return {
            "target_img": torch.from_numpy(out[[0]]),
            "context":    torch.from_numpy(out[1:]),
            "target_seg":   torch.from_numpy(target_seg.astype(np.float32)),
            "context_seg":  torch.from_numpy(context_seg.astype(np.float32)),
            "target_time":  s["target_time"],
            "context_time": s["context_time"],
        }


def _to_single_frame_seg(seg_arr, spatial_shape):
    """Collapse a seg with extra leading dims (channel, time) to one (D, H, W)
    anchor frame -- first non-empty slice, else frame 0. ACDC's seg is
    already (D, H, W) (no-op); Lumiere's is (T, 1, D, H, W)."""
    seg_arr = np.asarray(seg_arr)
    while seg_arr.ndim > len(spatial_shape):
        flat = seg_arr.reshape(-1, *seg_arr.shape[-len(spatial_shape):])
        nonempty = next((f for f in flat if f.sum() > 0), flat[0])
        seg_arr = nonempty
    return seg_arr


def _to_tensor(x):
    """Underlying datasets disagree on numpy vs torch (Lumiere: numpy, ACDC:
    torch) -- normalize every key to tensor so augmented and pass-through
    samples return identical types, or default_collate breaks when a batch
    mixes them."""
    return torch.as_tensor(np.asarray(x)) if not isinstance(x, torch.Tensor) else x


def _import_laugen():
    # laugen is a sibling of src/, not on sys.path when cwd=src
    _repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    import laugen
    return laugen


class LaugenSharedDeformWrapper(Dataset):
    # TODO: review
    """Wraps a dataset with LAUGEN's `apply_shared_deform_to_sequence`: one
    field per sample, applied identically to every real frame. `aug_prob`
    gates it per-sample (like batchgenerators' `p_per_sample`)."""

    def __init__(self, ds: Dataset, aug_prob: float = 0.5, intensity: float = 12.0,
                 growth_max: float = 4.0, blur_range=(1.0, 2.0), n_blobs: int = 1,
                 use_blob_localization: bool = True, seg_threshold: float = 1.0) -> None:
        self.dataset = ds
        self.aug_prob = aug_prob
        self.deform_kwargs = dict(intensity=intensity, growth_max=growth_max,
                                   blur_range=blur_range, n_blobs=n_blobs, apply_prob=1.0,
                                   fill_holes=True, fill_holes_axis=0,
                                   use_blob_localization=use_blob_localization,
                                   seg_threshold=seg_threshold)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_data_shape(self):
        return self.dataset._get_data_shape()

    def __getitem__(self, idx: int) -> dict:
        s = self.dataset[idx]
        s = {k: _to_tensor(v) for k, v in s.items()}
        if np.random.rand() >= self.aug_prob:
            return s

        apply_shared_deform_to_sequence = _import_laugen().apply_shared_deform_to_sequence

        t, c = s["target_img"].numpy(), s["context"].numpy()
        t_seg, c_seg = s["target_seg"].numpy(), s["context_seg"].numpy()

        combined = np.concatenate([t, c], axis=0).astype(np.float32)  # (T1, C, D, H, W)
        T1, C, D, H, W = combined.shape
        assert C == 1, "LaugenSharedDeformWrapper assumes single-channel images"

        anchor_seg = c_seg if c_seg is not None and np.asarray(c_seg).sum() > 0 else t_seg
        anchor_seg = _to_single_frame_seg(anchor_seg, (D, H, W))
        warped = apply_shared_deform_to_sequence(combined[:, 0], anchor_seg, target_shape=(D, H, W),
                                                  **self.deform_kwargs)
        out = warped[:, None]  # (T1, 1, D, H, W)

        return {
            "target_img": torch.from_numpy(out[[0]]),
            "context":    torch.from_numpy(out[1:]),
            "target_seg":   s["target_seg"],
            "context_seg":  s["context_seg"],
            "target_time":  s["target_time"],
            "context_time": s["context_time"],
        }


class SemiSynthAugWrapper(Dataset):
    # TODO: review
    """Wraps a dataset with LAUGEN's `deform_structure`: replaces the whole
    sample with a synthetic growth sequence from one real frame, instead of
    warping real frames in place (that's `LaugenSharedDeformWrapper`).
    `aug_ratio` gates it per-sample. Generic across datasets (no per-dataset
    subclass), unlike SADM's `semisynth/mixin.py` origin."""

    def __init__(self, ds: Dataset, aug_ratio: float = 0.5, intensity: float = 20.0,
                 growth_max: float = 5.0, n_blobs: int = 2, time_nonlinear: bool = True,
                 global_noise_std: float = 0.0, global_noise_sigma: float = 8.0,
                 bias_field: bool = False, bias_coeff: float = 0.5, bias_std: float = 4.0,
                 intensity_noise_std: float = 0.02, seg_intensity: bool = True,
                 seg_intensity_alpha: float = 0.15, seg_intensity_sigma: float = 6.0) -> None:
        self.dataset = ds
        self.aug_ratio = aug_ratio
        self.bias_field = bias_field
        self.bias_coeff = bias_coeff
        self.bias_std = bias_std
        self.intensity_noise_std = intensity_noise_std
        self.seg_intensity = seg_intensity
        self.seg_intensity_alpha = seg_intensity_alpha
        self.seg_intensity_sigma = seg_intensity_sigma
        self.deform_kwargs = dict(intensity=intensity, growth_max=growth_max, n_blobs=n_blobs,
                                   time_nonlinear=time_nonlinear, global_noise_std=global_noise_std,
                                   global_noise_sigma=global_noise_sigma,
                                   fill_holes=True, fill_holes_axis=0)

    def __len__(self) -> int:
        return len(self.dataset)

    def _get_data_shape(self):
        return self.dataset._get_data_shape()

    def __getitem__(self, idx: int) -> dict:
        s = self.dataset[idx]
        s = {k: _to_tensor(v) for k, v in s.items()}
        if np.random.rand() >= self.aug_ratio:
            return s

        laugen = _import_laugen()

        ctx = s["context"].numpy()          # (Tc, C, *spatial)
        t_seg, c_seg = s["target_seg"].numpy(), s["context_seg"].numpy()
        n_context, C = ctx.shape[0], ctx.shape[1]
        assert C == 1, "SemiSynthAugWrapper assumes single-channel images"

        # source frame: fullest context frame (robust to Lumiere-style
        # zero-padded context, a no-op for ACDC's already-dense context)
        frame_sums = ctx[:, 0].reshape(n_context, -1).sum(axis=1)
        src_idx = int(np.argmax(frame_sums))
        src = ctx[src_idx, 0]  # (*spatial)

        anchor_seg = c_seg if np.asarray(c_seg).sum() > 0 else t_seg
        anchor_seg = _to_single_frame_seg(anchor_seg, src.shape)
        seg = (anchor_seg > 0.5).astype(np.float32) if np.any(anchor_seg > 0) \
            else (src > src.mean()).astype(np.float32)

        # non-uniform time sampling: t=0 at the source frame, rest from (0, 1].
        n_total = n_context + 1
        t_rest = np.sort(np.random.uniform(0.05, 1.0, n_total - 1)).astype(np.float32)
        time_points = np.concatenate([np.array([0.0], dtype=np.float32), t_rest])

        seq = laugen.deform_structure(src, seg, time_points, target_shape=src.shape,
                                       **self.deform_kwargs)  # (n_total, *spatial)

        if self.seg_intensity:
            seq = laugen.apply_seg_intensity(seq, seg, alpha=self.seg_intensity_alpha,
                                              smooth_sigma=self.seg_intensity_sigma)
        if self.bias_field:
            seq = laugen.apply_bias_field(seq, seg, coeff=self.bias_coeff, std_dev=self.bias_std)
        if self.intensity_noise_std > 0.0:
            seq = seq + np.random.randn(*seq.shape).astype(np.float32) * self.intensity_noise_std

        context = seq[:-1, None]   # (Tc, 1, *spatial)
        target = seq[[-1], None]   # (1,  1, *spatial)

        return {
            "target_img":   torch.from_numpy(target),
            "context":      torch.from_numpy(context),
            "target_seg":   torch.ones_like(s["target_seg"]),
            "context_seg":  torch.ones_like(s["context_seg"]),
            "target_time":  torch.from_numpy(time_points[[-1]]),
            "context_time": torch.from_numpy(time_points[:-1]),
        }


def build_aug_transform(image_size: tuple, spatial_only: bool = False):
    """Build the batchgenerators augmentation pipeline (training only).

    Matches SADM/Synthwave's own pipeline (utils/load_data_and_network.py)
    exactly -- this repo's previous params (elastic deform on, +-15 deg
    rotation, no scale, added Gaussian noise, wider/more-frequent gamma and
    brightness) were substantially more aggressive on every axis than SADM's,
    a likely cause of CRONOS/TFM not beating the last-context-frame baseline
    on ACDC here despite the same architecture training fine in SADM.

    Gamma/Brightness use per_channel=False, applied once across all stacked
    T frames -- context/target intensity relationship isn't scrambled.
    spatial_only=True drops them, keeping only SpatialTransform, for ablation.
    """
    from batchgenerators.transforms.spatial_transforms import SpatialTransform
    from batchgenerators.transforms.color_transforms import (
        GammaTransform,
        BrightnessMultiplicativeTransform,
        ClipValueRange,
    )
    from batchgenerators.transforms.abstract_transforms import Compose

    transforms = [
        SpatialTransform(
            patch_size=image_size,
            do_elastic_deform=False, alpha=(0., 500.), sigma=(6., 10.),
            do_rotation=True,
            angle_x=(-5 / 180 * np.pi, 5 / 180 * np.pi),
            angle_y=(-5 / 180 * np.pi, 5 / 180 * np.pi),
            angle_z=(-5 / 180 * np.pi, 5 / 180 * np.pi),
            do_scale=True, scale=(0.9, 1.1),
            border_mode_data='nearest',
            order_data=1,
            random_crop=True,
        ),
    ]
    if not spatial_only:
        transforms += [
            GammaTransform(gamma_range=(0.8, 1.25), invert_image=False, p_per_sample=0.15),
            BrightnessMultiplicativeTransform(multiplier_range=(0.9, 1.1), per_channel=False, p_per_sample=0.15),
        ]
    transforms.append(ClipValueRange(min=0.0, max=1.0))
    return Compose(transforms)




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
        kwargs = dict(vars(args))
        kwargs.setdefault('num_to_keep_context', 5)
        if args.dataset == 'acdc':
            data_dir = os.getenv("DATA_DIR", "./data/")
            local_data_path_folder = 'ACDC'
            data_dir = os.path.join(data_dir, local_data_path_folder)
            dataset = ACDCDataset(
                data_dir=data_dir,
                split=train_test_val,
                **kwargs
            )
        elif args.dataset == 'isles':
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = ISLESDataset(
                data_dir=data_dir,
                train_test_val=train_test_val,
                **kwargs
            )
        elif args.dataset == 'lumiere':
            from .lumiere_loader import LumiereDataset
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = LumiereDataset(
                data_dir=data_dir,
                train_test_val=train_test_val,
                **kwargs
            )
        elif args.dataset == 'oasis':
            from .oasis_loader import OASISDataset
            data_dir = os.getenv("DATA_DIR", "./data/")
            dataset = OASISDataset(
                data_dir=data_dir,
                train_test_val=train_test_val,
                **kwargs
            )
        elif args.dataset == 'aimi':
            from .aimi_loader import AimiDataset
            data_dir = os.getenv("DATASET_LOCATION_AIMI", "./data/aimi")
            dataset = AimiDataset(
                data_dir=data_dir,
                train_test_val=train_test_val,
                **kwargs
            )
        else:
            raise NotImplementedError(
            "Provide your own dataset or run with --use-dummy-data to test the pipeline."
            )

    if train_test_val == 'trn' and getattr(args, 'semisynth_augmentation', False):
        dataset = SemiSynthAugWrapper(
            dataset,
            aug_ratio=getattr(args, 'semisynth_aug_ratio', 0.5),
            intensity=getattr(args, 'semisynth_intensity', 20.0),
            growth_max=getattr(args, 'semisynth_growth_max', 5.0),
            n_blobs=getattr(args, 'semisynth_n_blobs', 2),
            bias_field=getattr(args, 'semisynth_bias_field', False),
            seg_intensity=getattr(args, 'semisynth_seg_intensity', True),
        )

    if train_test_val == 'trn' and getattr(args, 'laugen_augmentation', False):
        dataset = LaugenSharedDeformWrapper(
            dataset,
            aug_prob=getattr(args, 'laugen_aug_prob', 0.5),
            intensity=getattr(args, 'laugen_intensity', 12.0),
            growth_max=getattr(args, 'laugen_growth_max', 4.0),
            n_blobs=getattr(args, 'laugen_n_blobs', 1),
            use_blob_localization=getattr(args, 'laugen_use_blobs', True),
            seg_threshold=getattr(args, 'laugen_seg_threshold', 1.0),
        )

    if train_test_val == 'trn' and getattr(args, 'augmentation', False):
        image_size = dataset._get_data_shape()[2:]
        spatial_only = getattr(args, 'augmentation_spatial_only', False)
        dataset = BGTransformWrapper(dataset, build_aug_transform(image_size, spatial_only=spatial_only))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(train_test_val == 'trn'),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader
