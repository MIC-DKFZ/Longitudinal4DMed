import imageio
import numpy as np
import os

import torch

from tfm.data_loaders.acdc_loader import ACDCDataset


def save_gif(volume_t, path, duration=0.1):
    # volume_t: array [T,H,W]
    frames = [(x / x.max() * 255).numpy().astype(np.uint8) for x in volume_t]
    imageio.mimsave(path, frames, duration=duration, loop=0)


if __name__ == '__main__':
    dataset = ACDCDataset
    data_dir = os.getenv("DATA_DIR_ACDC", "./data/ACDC/")
    hparams = {
        "num_to_keep_context": 12,
        "debug": True,
        "val_split": 0,
    }
    dataset = ACDCDataset(data_dir=data_dir, split="val", **hparams)
    sample = next(iter(dataset))
    all_img = torch.concatenate([sample['context'], sample['target_img']], dim=0)
    save_gif(all_img[:, 0, 16], "../results/acdc_example.gif", duration=0.08)
