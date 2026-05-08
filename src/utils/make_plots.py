import imageio
import numpy as np
import os

import torch


def save_gif(volume_t, path, fps: int = 8):
    """Save a (T, H, W) tensor or list of (H, W) arrays as an animated GIF.

    Each frame is normalised independently to [0, 255].
    """
    frames = []
    for x in volume_t:
        if torch.is_tensor(x):
            x = x.detach().cpu().float().numpy()
        x = x.squeeze()
        lo, hi = x.min(), x.max()
        x = ((x - lo) / (hi - lo + 1e-8) * 255).clip(0, 255).astype(np.uint8)
        frames.append(x)
    imageio.mimsave(str(path), frames, fps=fps, loop=0)



if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from data_loaders.acdc_loader import ACDCDataset

    data_dir = os.getenv("DATA_DIR", "./data/ACDC/")
    dataset = ACDCDataset(data_dir=data_dir, split="val", num_to_keep_context=12, debug=True, val_split=0)
    sample = next(iter(dataset))
    all_img = torch.cat([sample['context'], sample['target_img']], dim=0)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir  = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "acdc_example.gif")
    save_gif(all_img[:, 0, all_img.shape[2] // 2, :, :], out_path)
    print(f"Saved {out_path}")
