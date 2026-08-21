"""
Generate the LAUGEN hero GIF (`assets/hero_acdc.gif`) used at the top of
`laugen/README.md`.

Unlike `demo.py`/`demo.ipynb` (which are data-free, using a synthetic toy blob so
the notebook runs with no bundled data), this script needs a real ACDC frame + its
segmentation mask, so it is not part of the package and not run in tests/CI — it's
a one-off asset-generation script.

Shows LAUGEN applied independently to each of ACDC's 3 anatomical structures (RV
cavity, myocardium, LV cavity), each at 3 deformation sizes (mild/moderate/
aggressive) -- a 3x3 grid, animated over time, from one single input frame.

Data: point ACDC_DATA_DIR at a directory containing trn_dat.npy / trn_seg.npy
(defaults to the SADM/Synthwave repo's own data/ dir -- see laugen memory note
"ACDC data path"). trn_dat.npy: (N, 12, H, W, D) float32, 12 cardiac-cycle frames
per patient. trn_seg.npy: (N, H, W, D, 2) float32, only 2 segmented frames per
patient (index 0 = ED, index -1 = ES); values are continuous (resampling
interpolation) but round cleanly to integer labels 0=background, 1=RV, 2=myo, 3=LV.

Usage:
    python make_hero_gif.py
    ACDC_DATA_DIR=/path/to/data python make_hero_gif.py
"""
import os

import numpy as np
import imageio

from laugen import deform_structure

ACDC_DATA_DIR = os.environ.get(
    "ACDC_DATA_DIR",
    os.path.expanduser(
        "~/PycharmProjects/Synthwave/SADM-Longitudinal-Medical-Image-Generation/data"
    ),
)

# Patient 15: well-balanced RV/myocardium/LV label counts at depth slice 5 (of 32).
# Patient 13 (the original pick) turned out to be an outlier -- its myocardium
# mask produces a large, diffuse deformation instead of the clean thin-ring
# effect every other patient shows (visually confirmed by diffing each
# structure's t=0 vs t=9 frame across several candidates); it's a sample
# issue, not a deform_structure bug.
PATIENT_IDX = 15
DEPTH_SLICE = 5

STRUCTURES = [("RV", 1), ("Myocardium", 2), ("LV", 3)]
SIZES = [("mild", 2.5, 3.0), ("moderate", 6.0, 4.0), ("aggressive", 10.0, 5.0)]
# Sharper boundary falloff than deform_structure's default U(2, 7) -- less halo/
# ghosting around each structure, so the 3 rows read as distinct, localised effects.
BLUR_RANGE = (1.5, 2.5)

N_TIMESTEPS = 10
FPS = 6
OUT_PATH = "assets/hero_acdc.gif"

# Crop margin around the heart (union of RV+myo+LV), as a fraction of the
# bounding box's side length -- generous, since the larger deformation sizes
# push structure content outward and a tight crop would clip that motion.
CROP_MARGIN_FRAC = 0.6
CROP_MIN_MARGIN = 10


def _compute_crop_bounds(seg_labels, depth, margin_frac=CROP_MARGIN_FRAC,
                          min_margin=CROP_MIN_MARGIN):
    """Square crop box around the union of all labeled structures at `depth`,
    padded by `margin_frac` of the box's side length (plus a pixel floor)."""
    mask = seg_labels[:, :, depth] > 0
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
    side = max(y1 - y0, x1 - x0)
    half = side / 2 + max(int(side * margin_frac), min_margin)
    H, W = seg_labels.shape[:2]
    y0c, y1c = int(max(0, cy - half)), int(min(H, cy + half))
    x0c, x1c = int(max(0, cx - half)), int(min(W, cx + half))
    return y0c, y1c, x0c, x1c


def load_frame():
    data = np.load(os.path.join(ACDC_DATA_DIR, "trn_dat.npy"), mmap_mode="r")
    seg = np.load(os.path.join(ACDC_DATA_DIR, "trn_seg.npy"), mmap_mode="r")
    img = np.asarray(data[PATIENT_IDX, 0]).astype(np.float32)  # (H, W, D), ED frame
    seg_labels = np.round(np.asarray(seg[PATIENT_IDX, ..., 0])).astype(np.int32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img, seg_labels


def generate(sizes=SIZES, blur_range=BLUR_RANGE, out_path=OUT_PATH,
             img=None, seg_labels=None, pingpong=True):
    if img is None or seg_labels is None:
        img, seg_labels = load_frame()
    time_points = np.linspace(0, 1, N_TIMESTEPS, dtype=np.float32)
    y0, y1, x0, x1 = _compute_crop_bounds(seg_labels, DEPTH_SLICE)

    sequences = {}
    for struct_name, label in STRUCTURES:
        struct_seg = (seg_labels == label).astype(np.float32)
        for size_name, intensity, growth_max in sizes:
            np.random.seed(0)  # same draw across cells -> differences reflect the knobs, not luck
            seq = deform_structure(img, struct_seg, time_points, intensity=intensity,
                                    growth_max=growth_max, target_shape=img.shape,
                                    apply_prob=1.0, n_blobs=1, blur_range=blur_range,
                                    fill_holes=True, fill_holes_axis=2)
            sequences[(struct_name, size_name)] = seq

    grid_frames = []
    for t in range(N_TIMESTEPS):
        rows = []
        for struct_name, _ in STRUCTURES:
            cols = []
            for size_name, *_ in sizes:
                sl = sequences[(struct_name, size_name)][t, y0:y1, x0:x1, DEPTH_SLICE]
                sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
                cols.append((sl * 255).astype(np.uint8))
            # 2px white separator between size columns
            sep = np.full((cols[0].shape[0], 2), 255, dtype=np.uint8)
            row = cols[0]
            for c in cols[1:]:
                row = np.concatenate([row, sep, c], axis=1)
            rows.append(row)
        sep_row = np.full((2, rows[0].shape[1]), 255, dtype=np.uint8)
        grid = rows[0]
        for r in rows[1:]:
            grid = np.concatenate([grid, sep_row, r], axis=0)
        grid_frames.append(grid)

    if pingpong:
        # forward + reverse (excluding both endpoints once) -> f0..f9,f8..f1 -> loops
        # straight back into f0 with no jump/repeat at the seam.
        grid_frames = grid_frames + grid_frames[-2:0:-1]

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    imageio.mimsave(out_path, grid_frames, fps=FPS, loop=0)
    print(f"saved {out_path}, frame shape {grid_frames[0].shape}, "
          f"rows={[s[0] for s in STRUCTURES]}, cols={[s[0] for s in sizes]}, "
          f"blur_range={blur_range}")


def main():
    generate()


if __name__ == "__main__":
    main()
