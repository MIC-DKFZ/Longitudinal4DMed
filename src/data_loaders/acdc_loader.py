import os
import random

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ACDCDataset(Dataset):
    """
    ACDC Dataset for temporal medical image generation.
    Adapted from https://github.com/ubc-tea/SADM-Longitudinal-Medical-Image-Generation
    Added val split and deterministic validation sampling
    """
    def __init__(self, data_dir, split="trn", **kwargs):
        self.hparams = kwargs
        val_split = self.hparams.get('val_split', 0)
        if self.hparams['debug'] == True:
            # safeguard
            if split == 'tst':
                split = 'trn'

        if split == 'tst':
            # For the test set, load only the test data.
            self.data = np.transpose(
                np.load(os.path.join(data_dir, f"{split}_dat.npy")), (0, 1, 4, 3, 2)
            )[:, :, None]
        else:
            # Merge training and validation npy files.
            trn_data = np.load(os.path.join(data_dir, "trn_dat.npy"))
            val_data = np.load(os.path.join(data_dir, "val_dat.npy"))
            merged = np.concatenate((trn_data, val_data), axis=0)

            # Apply the transpose and add a singleton dimension.
            merged = np.transpose(merged, (0, 1, 4, 3, 2))[:, :, None]

            # Apply the bootstrap split via remainder.
            indices = np.arange(merged.shape[0])
            if split == 'trn':
                self.data = merged[indices % 5 != val_split]
            elif split == 'val':
                self.data = merged[indices % 5 == val_split]

        if split == 'tst':
            self.seg = np.load(os.path.join(data_dir, f"{split}_seg.npy"))
        else:
            trn_seg = np.load(os.path.join(data_dir, "trn_seg.npy"))
            val_seg = np.load(os.path.join(data_dir, "val_seg.npy"))
            merged_seg = np.concatenate((trn_seg, val_seg), axis=0)
            indices = np.arange(merged_seg.shape[0])
            if split == 'trn':
                self.seg = merged_seg[indices % 5 != val_split]
            elif split == 'val':
                self.seg = merged_seg[indices % 5 == val_split]
        self.frames = self.data.shape[1]
        # we only slightly change this in order to keep the old stuff
        self.num_to_keep = kwargs.get("num_to_keep_context", 11)
        self.code_original = False
        self.distance = self.num_to_keep

        if self.hparams['debug'] == True:
            self.data = self.data[:6]
            self.seg = self.seg[:6]  # also slice seg!
        if split == 'trn':
            self.precompute_random = False
            self.indices_random = None
        else:
            if self.num_to_keep >= self.frames - 1:
                self.precompute_random = False
                self.indices_random = None
            else:
                self.precompute_random = True
                self.precompute_randomness()
        # we keep it like that, for now

    def __len__(self):
        return len(self.data)

    def _create_missing_mask(self):
        target_idx = np.random.randint(max(self.distance, 1) + 1, self.frames)
        # Create the missing_mask with random entries (length = target_idx - distance)
        last_context = target_idx - self.distance
        missing_mask = np.zeros(target_idx, dtype=np.float32)
        missing_bits = np.random.randint(0, 2, size=last_context).tolist()
        missing_mask[:last_context] = missing_bits

        # todo: the option that exactly self.distance frames are missing
        # todo: the option that all up to distance are there!
        if not any(missing_mask):
            missing_mask[np.random.randint(last_context)] = 1

            # missing_mask[np.random.randint(len(missing_mask))] = 1
        # Add a 1 for the first frame
        return missing_mask, target_idx

    def precompute_randomness(self):
        """Precompute the random target index and missing mask for each sample."""
        print("Precomputing random target index and missing mask for deterministic validation and test")
        self.indices_random = []
        N = len(self.data)
        for idx in range(N):
            missing_mask, target_idx = self._create_missing_mask()
            self.indices_random.append({
                'target_idx': target_idx,
                'missing_mask': missing_mask
            })

    def _get_data_shape(self):
        # we can make this more elegant, but this is fine I guess? maybe just move this to the train
        T_all, C, D, H, W = self.data.shape[1:]
        return (int(T_all-1), C, D, H, W)

    def __getitem__(self, idx):
        seg_a = None
        seg_b = None
        seg = self.seg[idx]
        if self.seg is not None:
            seg_a, seg_b = seg[..., 0], seg[..., -1]

        # functionality to get the dense context
        if self.num_to_keep >= self.frames - 1:
            x_prev = self.data[idx, :-1]
            x = self.data[idx, -1]
            time_vector = np.linspace(0, 1, x_prev.shape[0] + 1, dtype=np.float32)
            return {"target_img": torch.from_numpy(x.astype(np.float32)[np.newaxis, ...]),
                    "context": torch.from_numpy(x_prev.astype(np.float32)),
                    "target_seg": seg_a, "context_seg": seg_b,
                    "target_time": time_vector[[-1]], "context_time": time_vector[:-1]}  #seg_a, seg_b
        elif self.code_original:  # the original code
            target_idx = np.random.randint(1, self.frames)
            missing_mask = [1]
            if target_idx > 1:
                if target_idx > 2:
                    missing_mask = np.append(missing_mask, np.random.randint(0, 2, size=(target_idx - 2,)))
                missing_mask = np.append(missing_mask, [1])
            missing_mask = np.append(missing_mask, np.zeros(self.frames - len(missing_mask))).astype(np.float32)

            x_prev = np.clip(self.data[idx, :-1] * missing_mask[:-1, None, None, None, None], 0., 1.)
            x = self.data[idx, target_idx]
            return torch.from_numpy(x), torch.from_numpy(x_prev), seg_a, seg_b

        else:
            # If precomputation is enabled, simply use the stored values,
            # otherwise compute them on the fly.
            # PRECOMPUTATION IS IMPORTANT FOR THE VALIDATION SET TO BE COMPARABLE
            # see: https://arxiv.org/abs/2508.21580
            if self.precompute_random:
                rand_info = self.indices_random[idx]
                target_idx = rand_info['target_idx']
                missing_mask = rand_info['missing_mask']
            else:
                missing_mask, target_idx = self._create_missing_mask()

            x_prev = self.data[idx, :len(missing_mask)] * missing_mask[..., None, None, None, None]
            x_prev_final = np.concatenate(
                (np.zeros((self.frames - 1 - x_prev.shape[0], *x_prev.shape[1:])), x_prev),
                axis=0
            )
            x = self.data[idx, target_idx]

            # add the segmentations as well if we need them downstream or for evals
            seg_a = np.transpose(seg_a, (2, 0, 1))
            seg_b = np.transpose(seg_b, (2, 0, 1))
            time_vector = np.linspace(0, 1, x_prev_final.shape[0] + 1, dtype=np.float32)
            target_time = time_vector[[target_idx]]
            context_time = time_vector[:target_idx]
            context_time = np.where(missing_mask.astype(bool), context_time, -1.0).astype(np.float32)

            # left-pad to match length (self.frames - 1); padded slots also -1
            pad_len = self.frames - 1 - context_time.shape[0]
            if pad_len > 0:
                pad_times = -np.ones(pad_len, dtype=np.float32)
                context_time = np.concatenate((pad_times, context_time), axis=0)

            return {"target_img": torch.from_numpy(x.astype(np.float32)[np.newaxis, ...]),
                    "context": torch.from_numpy(x_prev_final.astype(np.float32)),
                    "target_seg": seg_a, "context_seg": seg_b,
                    "target_time": target_time, "context_time": context_time}



if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR_ACDC", "./data/ACDC/")
    hparams = {
        "num_to_keep_context": 5,
        "debug": True,
        "val_split": 0,
    }
    dataset = ACDCDataset(data_dir=data_dir, split="val", **hparams)
    print(f"Dataset length: {len(dataset)}")
    example = next(iter(dataset))
    print("Sampled one data point successfully.")