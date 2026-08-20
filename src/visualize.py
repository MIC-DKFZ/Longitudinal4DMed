#!/usr/bin/env python
"""
Visualize predictions from a saved checkpoint.

Usage:
    python src/visualize.py --checkpoint checkpoints/acdc/20240101_120000_tfm_best.pt --out-dir results/vis
    python src/visualize.py --checkpoint ... --dummy --device cpu   # no real data needed
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import build_model
from data_loaders.data_utils import build_dataloader, DummyTemporalDataset
from utils.validation_utils import get_last_context_image_baseline, _forward_and_reshape, _extract_batch
from utils.make_plots import save_gif


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _central_slice(t: torch.Tensor) -> np.ndarray:
    """Return a normalised 2D central slice from a (C, D, H, W) or (D, H, W) tensor."""
    arr = t.squeeze().detach().cpu().float().numpy()
    while arr.ndim > 2:
        arr = arr[arr.shape[0] // 2]
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def save_prediction_grid(gt, lci, pred, out_path: Path, idx: int) -> None:
    """PNG grid: images (top row) + residuals (bottom row)."""
    gt_s   = _central_slice(gt)
    lci_s  = _central_slice(lci)
    pred_s = _central_slice(pred)

    res_lci  = np.abs(lci_s  - gt_s)
    res_pred = np.abs(pred_s - gt_s)
    rel_max  = max(res_lci.max(), res_pred.max(), 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, img, title in zip(
        axes[0],
        [gt_s, lci_s, pred_s],
        ["Ground truth", "Last context (LCI)", "Prediction"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    axes[1, 0].axis("off")
    for ax, res, title in zip(
        axes[1, 1:],
        [res_lci / rel_max, res_pred / rel_max],
        ["|LCI - GT|", "|Pred - GT|"],
    ):
        im = ax.imshow(res, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    cax = fig.add_axes([0.12, 0.04, 0.76, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Normalised absolute error")

    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / f"sample_{idx:03d}_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Restore model from checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    model_args = argparse.Namespace(**saved_args)
    model_args.device = str(device)
    if args.data_dir:
        os.environ["DATA_DIR"] = args.data_dir

    model = build_model(model_args, device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt.get('epoch', '?')}  loss={ckpt.get('avg_loss', '?'):.4f}")

    # Build dataloader
    if args.dummy:
        from torch.utils.data import DataLoader
        T, C = model_args.in_shape[:2]
        ds = DummyTemporalDataset(length=max(args.num_samples, 4), T=T, C=C,
                                  D=model_args.in_shape[2], H=model_args.in_shape[3], W=model_args.in_shape[4])
        loader = DataLoader(ds, batch_size=2, shuffle=False)
    else:
        if args.dataset:
            model_args.dataset = args.dataset
        model_args.batch_size = 2
        model_args.num_workers = 0
        loader = build_dataloader(model_args, train_test_val="val")

    in_shape = model_args.in_shape

    n_done = 0
    with torch.no_grad():
        for batch in loader:
            if n_done >= args.num_samples:
                break

            batch_x, batch_y, _, _, time_points = _extract_batch(batch, device)
            lci = get_last_context_image_baseline(batch_x)

            pred, batch_y = _forward_and_reshape(model, batch_x, batch_y, time_points, in_shape=in_shape)

            B = batch_x.shape[0]
            for b in range(B):
                if n_done >= args.num_samples:
                    break
                sample_dir = out_dir / f"sample_{n_done:03d}"

                gt_vol   = batch_y[b]   # (C, D, H, W)
                pred_vol = pred[b]
                lci_vol  = lci[b]

                save_prediction_grid(gt_vol, lci_vol, pred_vol, sample_dir, n_done)
                print(f"  grid  -> {sample_dir / f'sample_{n_done:03d}_grid.png'}")

                if args.save_gif:
                    frames = [_central_slice(batch_x[b, t]) for t in range(batch_x.shape[1])]
                    frames.append(_central_slice(pred_vol))
                    gif_path = sample_dir / f"sample_{n_done:03d}_temporal.gif"
                    save_gif(frames, gif_path)
                    print(f"  gif   -> {gif_path}")

                n_done += 1

    print(f"\nDone. {n_done} samples saved to {out_dir}")


def _get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize TFM/CRONOS predictions from a checkpoint.")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    p.add_argument("--out-dir", default="results/vis", help="Output directory.")
    p.add_argument("--dataset", default='isles',
                   help="Dataset name override (acdc / isles / lumiere). Uses saved value if omitted.")
    p.add_argument("--data-dir", default=None, help="Data root (sets DATA_DIR env var).")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--save-gif", action="store_true", help="Also save animated GIFs.")
    p.add_argument("--dummy", action="store_true", help="Use DummyTemporalDataset (no real data).")
    return p.parse_args()


if __name__ == "__main__":
    run(_get_args())
