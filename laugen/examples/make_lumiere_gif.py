"""
Sanity-check GIF for LUMIERE: real cycle vs the same real cycle warped by
one shared field (grad-only, no blob localization) side by side.

Not part of the package, not run in tests/CI. Reuses helpers from
make_shared_field_gif.py (crop/label/arrow drawing).

Usage:
    DATA_DIR=/path/to/data python make_lumiere_gif.py
"""
import os
import sys

import numpy as np
import imageio

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from laugen import apply_shared_deform_to_sequence
from laugen.deform import _sample_blob_field
from make_shared_field_gif import _to_frame, _label, _draw_arrows

from data_loaders.lumiere_loader import LumiereDataset

DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.expanduser("~/PycharmProjects/Synthwave/SADM-Longitudinal-Medical-Image-Generation/data"),
)
FPS = 4


def _pick_patient(ds, min_real_context=4):
    """First patient with >= min_real_context nonzero (non-padded) context frames."""
    for i in range(len(ds)):
        s = ds[i]
        n_real = int((np.asarray(s["context_time"]) > 0).sum())
        if n_real >= min_real_context:
            return i, s
    return 0, ds[0]


def _crop_bounds(seg_vol, margin_frac=0.4, min_margin=8):
    # Lumiere's seg is continuous/z-scored (~[-0.7, 2.6]), not a clean 0/1
    # mask -- >0 mostly picks up noise, real tumor sits above ~0.5.
    fg = (seg_vol > 0.5).sum(axis=-1)
    ys, xs = np.nonzero(fg > 0) if fg.sum() > 0 else (np.array([0]), np.array([0]))
    if len(ys) == 0:
        H, W = seg_vol.shape[:2]
        return 0, H, 0, W
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0)
    half = side / 2 + max(int(side * margin_frac), min_margin)
    H, W = seg_vol.shape[:2]
    return int(max(0, cy - half)), int(min(H, cy + half)), int(max(0, cx - half)), int(min(W, cx + half))


def generate(out_path="assets/lumiere_shared_field.gif", intensity=12.0, growth_max=4.0,
             use_blob_localization=False, upscale=3, arrow_step=8):
    ds = LumiereDataset(data_dir=DATA_DIR, train_test_val="trn", num_to_keep_context=3, val_split=0)
    idx, s = _pick_patient(ds)

    context_time = np.asarray(s["context_time"])
    real_mask = context_time > 0
    context = np.asarray(s["context"])[real_mask, 0]  # (T_real, H, W, D)
    target = np.asarray(s["target_img"])[0, 0]  # (H, W, D)
    cycle = np.concatenate([context, target[None]], axis=0)
    cycle = (cycle - cycle.min()) / (cycle.max() - cycle.min() + 1e-8)

    seg = np.asarray(s["target_seg"])[0, 0]
    depth = int(np.argmax((seg > 0.5).sum(axis=(0, 1))))
    y0, y1, x0, x1 = _crop_bounds(seg[:, :, depth:depth + 1])

    np.random.seed(0)
    warped_cycle = apply_shared_deform_to_sequence(
        cycle, seg, intensity=intensity, growth_max=growth_max, apply_prob=1.0,
        n_blobs=1, blur_range=(1.0, 2.0), fill_holes=True, fill_holes_axis=2,
        use_blob_localization=use_blob_localization, seg_threshold=0.5,
    )
    np.random.seed(0)
    _, full_disp, _ = _sample_blob_field(
        cycle.shape[1:], seg, intensity, growth_max, 1.0, False, 0.0, 8.0, 1, (1.0, 2.0), True, 2,
        use_blob_localization=use_blob_localization, seg_threshold=0.5,
    )
    disp_h, disp_w = full_disp[0, :, :, depth], full_disp[1, :, :, depth]

    header_left = _label(np.full((20, (x1 - x0) * upscale, 3), 0, dtype=np.uint8), "GT", color=(255, 255, 255))
    header_right = _label(np.full((20, (x1 - x0) * upscale, 3), 0, dtype=np.uint8), "Warped", color=(255, 255, 255))
    header = np.concatenate([header_left, np.full((20, 2, 3), 255, dtype=np.uint8), header_right], axis=1)

    frames = []
    for t in range(cycle.shape[0]):
        left = np.stack([_to_frame(cycle[t], y0, y1, x0, x1, upscale, depth=depth)] * 3, axis=-1)
        right = _to_frame(warped_cycle[t], y0, y1, x0, x1, upscale, depth=depth)
        right = _draw_arrows(right, disp_h, disp_w, y0, x0, upscale, step=arrow_step)
        sep = np.full((left.shape[0], 2, 3), 255, dtype=np.uint8)
        row = np.concatenate([left, sep, right], axis=1)
        sep_row = np.full((2, row.shape[1], 3), 255, dtype=np.uint8)
        frames.append(np.concatenate([header, sep_row, row], axis=0))
    frames = frames + frames[-2:0:-1]

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"saved {out_path}, patient_idx={idx}, n_real_frames={cycle.shape[0]}, "
          f"depth={depth}, use_blob_localization={use_blob_localization}")


if __name__ == "__main__":
    generate(out_path="assets/lumiere_blobs.gif", use_blob_localization=True)
    generate(out_path="assets/lumiere_grad_only.gif", use_blob_localization=False)
