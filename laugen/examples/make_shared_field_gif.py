"""
GIF comparing `deform_structure` (single-frame -> synthetic sequence, the
existing hero-GIF approach) against `apply_shared_deform_to_sequence` (one
field sampled once, applied identically to every frame of the REAL cardiac
cycle) side by side.

Not part of the package, not run in tests/CI -- a one-off asset script, same
pattern as make_hero_gif.py (reuses its data loading / crop / calibrated
params).

Usage:
    python make_shared_field_gif.py
    ACDC_DATA_DIR=/path/to/data python make_shared_field_gif.py
"""
import os

import numpy as np
import imageio

from laugen import apply_shared_deform_to_sequence
from laugen.deform import _sample_blob_field
from make_hero_gif import (
    ACDC_DATA_DIR, PATIENT_IDX, DEPTH_SLICE, N_TIMESTEPS, FPS,
    CALIBRATED_PARAMS, load_frame, _compute_crop_bounds,
)

OUT_PATH = "assets/shared_field_real_cycle.gif"


def load_real_cycle():
    data = np.load(os.path.join(ACDC_DATA_DIR, "trn_dat.npy"), mmap_mode="r")
    cycle = np.asarray(data[PATIENT_IDX]).astype(np.float32)  # (T_real, H, W, D)
    cycle = (cycle - cycle.min()) / (cycle.max() - cycle.min() + 1e-8)
    return cycle


def _to_frame(vol, y0, y1, x0, x1, upscale=3, depth=DEPTH_SLICE):
    sl = vol[y0:y1, x0:x1, depth]
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    cell = (sl * 255).astype(np.uint8)
    if upscale and upscale != 1:
        from PIL import Image
        h, w = cell.shape
        cell = np.asarray(Image.fromarray(cell).resize((w * upscale, h * upscale), Image.LANCZOS))
    return cell


def generate(out_path=OUT_PATH, structure="Myocardium", pingpong=True, upscale=3):
    """Real cycle (left) vs the same real cycle warped by one shared field (right).

    Uses the ED-frame segmentation as the anatomy anchor for blob placement
    (real per-frame segs aren't available for the other 10 real frames), and
    `structure`'s calibrated params (see make_hero_gif.CALIBRATED_PARAMS) so
    the deformation magnitude matches what the synthetic-sequence hero GIF uses.
    """
    _, seg_labels = load_frame()
    cycle = load_real_cycle()
    y0, y1, x0, x1 = _compute_crop_bounds(seg_labels, DEPTH_SLICE)

    struct_label = {"RV": 1, "Myocardium": 2, "LV": 3}[structure]
    struct_seg = (seg_labels == struct_label).astype(np.float32)
    params = CALIBRATED_PARAMS[structure]

    np.random.seed(0)
    warped_cycle = apply_shared_deform_to_sequence(
        cycle, struct_seg, intensity=params["intensity"], growth_max=params["growth_max"],
        apply_prob=1.0, n_blobs=params["n_blobs"], blur_range=params["blur_range"],
        fill_holes=True, fill_holes_axis=2,
    )

    frames = []
    for t in range(cycle.shape[0]):
        left = _to_frame(cycle[t], y0, y1, x0, x1, upscale)
        right = _to_frame(warped_cycle[t], y0, y1, x0, x1, upscale)
        sep = np.full((left.shape[0], 2), 255, dtype=np.uint8)
        frames.append(np.concatenate([left, sep, right], axis=1))

    if pingpong:
        frames = frames + frames[-2:0:-1]

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"saved {out_path}, frame shape {frames[0].shape}, structure={structure}, "
          f"patient={PATIENT_IDX}, n_real_frames={cycle.shape[0]}, params={params}")


# Sweep around the Myocardium-calibrated defaults (12/4, see configs/laugen_shared_deform.yaml)
STRENGTH_LEVELS = [
    ("0.5x", 6.0, 3.0),
    ("1x (current)", 12.0, 4.0),
    ("1.5x", 18.0, 5.0),
    ("2x", 24.0, 6.0),
    ("3x", 36.0, 8.0),
]


def generate_strength_sweep(out_path="assets/shared_field_strength_sweep.gif",
                             structure="Myocardium", levels=STRENGTH_LEVELS,
                             pingpong=True, upscale=3):
    """One row per strength level: [original | warped] pair, animated over
    the real cycle -- so the deformation's actual visual magnitude can be
    judged directly instead of guessed from a single strength."""
    _, seg_labels = load_frame()
    cycle = load_real_cycle()
    y0, y1, x0, x1 = _compute_crop_bounds(seg_labels, DEPTH_SLICE)

    struct_label = {"RV": 1, "Myocardium": 2, "LV": 3}[structure]
    struct_seg = (seg_labels == struct_label).astype(np.float32)
    base_blur = CALIBRATED_PARAMS[structure]["blur_range"]
    base_n_blobs = CALIBRATED_PARAMS[structure]["n_blobs"]

    warped_by_level = {}
    for name, intensity, growth_max in levels:
        np.random.seed(0)  # same draw across levels -> differences reflect strength, not luck
        warped_by_level[name] = apply_shared_deform_to_sequence(
            cycle, struct_seg, intensity=intensity, growth_max=growth_max,
            apply_prob=1.0, n_blobs=base_n_blobs, blur_range=base_blur,
            fill_holes=True, fill_holes_axis=2,
        )

    frames = []
    for t in range(cycle.shape[0]):
        rows = []
        for name, *_ in levels:
            left = _to_frame(cycle[t], y0, y1, x0, x1, upscale)
            right = _to_frame(warped_by_level[name][t], y0, y1, x0, x1, upscale)
            sep = np.full((left.shape[0], 2), 255, dtype=np.uint8)
            rows.append(np.concatenate([left, sep, right], axis=1))
        sep_row = np.full((2, rows[0].shape[1]), 255, dtype=np.uint8)
        grid = rows[0]
        for r in rows[1:]:
            grid = np.concatenate([grid, sep_row, r], axis=0)
        frames.append(grid)

    if pingpong:
        frames = frames + frames[-2:0:-1]

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"saved {out_path}, frame shape {frames[0].shape}, structure={structure}, "
          f"levels={[(n, i, g) for n, i, g in levels]}")


def _draw_arrows(cell_gray, disp_h, disp_w, y0, x0, upscale, step=8, color=(255, 60, 60), min_mag=0.6):
    # TODO: review
    """Coarse quiver of the (H, W) displacement field onto a cropped+upscaled
    cell. Arrows use -disp (backward-warp field negated) so they point the
    way content actually moves, not the sampling direction."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(cell_gray).convert("RGB")
    draw = ImageDraw.Draw(img)
    y1, x1 = y0 + cell_gray.shape[0] // upscale, x0 + cell_gray.shape[1] // upscale
    for y in range(y0, y1, step):
        for x in range(x0, x1, step):
            dy, dx = -disp_h[y, x], -disp_w[y, x]
            mag = (dy ** 2 + dx ** 2) ** 0.5
            if mag < min_mag:
                continue
            py, px = (y - y0) * upscale, (x - x0) * upscale
            qy, qx = py + dy * upscale, px + dx * upscale
            draw.line([(px, py), (qx, qy)], fill=color, width=2)
            ang = np.arctan2(qy - py, qx - px)
            for da in (2.6, -2.6):
                hx, hy = qx - 6 * np.cos(ang + da), qy - 6 * np.sin(ang + da)
                draw.line([(qx, qy), (hx, hy)], fill=color, width=2)
    return np.asarray(img)


def _label(cell_rgb, text, xy=(4, 2), color=(255, 255, 0)):
    from PIL import Image, ImageDraw
    img = Image.fromarray(cell_rgb)
    ImageDraw.Draw(img).text(xy, text, fill=color)
    return np.asarray(img)


def generate_strength_sweep_annotated(out_path="assets/shared_field_strength_sweep_annotated.gif",
                                       structure="Myocardium", levels=STRENGTH_LEVELS,
                                       pingpong=True, upscale=3, arrow_step=8):
    """All strength levels in ONE gif, labeled GT vs Warped, with the
    sampled displacement field drawn as arrows on the Warped column so the
    deformation's actual shape (not just its net visual effect) is visible."""
    _, seg_labels = load_frame()
    cycle = load_real_cycle()
    y0, y1, x0, x1 = _compute_crop_bounds(seg_labels, DEPTH_SLICE)

    struct_label = {"RV": 1, "Myocardium": 2, "LV": 3}[structure]
    struct_seg = (seg_labels == struct_label).astype(np.float32)
    base_blur = CALIBRATED_PARAMS[structure]["blur_range"]
    base_n_blobs = CALIBRATED_PARAMS[structure]["n_blobs"]
    dim = cycle.shape[1:]  # (H, W, D)

    warped_by_level = {}
    field_by_level = {}
    for name, intensity, growth_max in levels:
        np.random.seed(0)  # same draw across levels -> differences reflect strength, not luck
        warped_by_level[name] = apply_shared_deform_to_sequence(
            cycle, struct_seg, intensity=intensity, growth_max=growth_max,
            apply_prob=1.0, n_blobs=base_n_blobs, blur_range=base_blur,
            fill_holes=True, fill_holes_axis=2,
        )
        np.random.seed(0)  # re-draw identically, just to also get the raw field for the arrows
        _, full_disp, _ = _sample_blob_field(
            dim, struct_seg, intensity, growth_max, 1.0, False, 0.0, 8.0,
            base_n_blobs, base_blur, True, 2,
        )
        field_by_level[name] = (full_disp[0, :, :, DEPTH_SLICE], full_disp[1, :, :, DEPTH_SLICE])

    header_left = _label(np.full((20, (x1 - x0) * upscale, 3), 0, dtype=np.uint8), "GT", color=(255, 255, 255))
    header_right = _label(np.full((20, (x1 - x0) * upscale, 3), 0, dtype=np.uint8),
                           "Warped (red = field)", color=(255, 255, 255))
    header = np.concatenate([header_left, np.full((20, 2, 3), 255, dtype=np.uint8), header_right], axis=1)

    frames = []
    for t in range(cycle.shape[0]):
        rows = [header]
        for name, *_ in levels:
            left = np.stack([_to_frame(cycle[t], y0, y1, x0, x1, upscale)] * 3, axis=-1)
            left = _label(left, name)
            right = _to_frame(warped_by_level[name][t], y0, y1, x0, x1, upscale)
            disp_h, disp_w = field_by_level[name]
            right = _draw_arrows(right, disp_h, disp_w, y0, x0, upscale, step=arrow_step)
            sep = np.full((left.shape[0], 2, 3), 255, dtype=np.uint8)
            rows.append(np.concatenate([left, sep, right], axis=1))
        sep_row = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
        grid = rows[0]
        for r in rows[1:]:
            grid = np.concatenate([grid, sep_row, r], axis=0)
        frames.append(grid)

    if pingpong:
        frames = frames + frames[-2:0:-1]

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"saved {out_path}, frame shape {frames[0].shape}, structure={structure}, "
          f"levels={[(n, i, g) for n, i, g in levels]}")


# Pure smooth global field, no anatomy guidance. # TODO: review -- not
# comparable to STRENGTH_LEVELS' magnitude 1:1 (gaussian_filter damps std a
# lot; std=300,sigma=8 -> max|disp|~14px, roughly matching "1x" there).
GLOBAL_NOISE_LEVELS = [
    ("weak", 150.0, 8.0),
    ("moderate", 300.0, 8.0),
    ("strong", 600.0, 8.0),
    ("very strong", 1000.0, 8.0),
]


def generate_global_noise_sweep_annotated(out_path="assets/shared_field_global_noise_sweep_annotated.gif",
                                           structure="Myocardium", levels=GLOBAL_NOISE_LEVELS,
                                           pingpong=True, upscale=3, arrow_step=8):
    """Same layout as generate_strength_sweep_annotated, but for the no-blob
    alternative: a smooth global random displacement field (global_noise_std/
    sigma), not anchored to anatomy at all. `structure`'s seg is still loaded
    for the crop bounds, but doesn't otherwise affect this field."""
    _, seg_labels = load_frame()
    cycle = load_real_cycle()
    y0, y1, x0, x1 = _compute_crop_bounds(seg_labels, DEPTH_SLICE)

    struct_label = {"RV": 1, "Myocardium": 2, "LV": 3}[structure]
    struct_seg = (seg_labels == struct_label).astype(np.float32)
    dim = cycle.shape[1:]  # (H, W, D)

    warped_by_level = {}
    field_by_level = {}
    for name, std, sigma in levels:
        np.random.seed(0)
        warped_by_level[name] = apply_shared_deform_to_sequence(
            cycle, struct_seg, intensity=0.0, growth_max=0.0, apply_prob=1.0,
            n_blobs=0, global_noise_std=std, global_noise_sigma=sigma,
            fill_holes=True, fill_holes_axis=2,
        )
        np.random.seed(0)
        _, full_disp, _ = _sample_blob_field(
            dim, struct_seg, 0.0, 0.0, 1.0, False, std, sigma, 0, (1, 2), True, 2,
        )
        field_by_level[name] = (full_disp[0, :, :, DEPTH_SLICE], full_disp[1, :, :, DEPTH_SLICE])

    header_left = _label(np.full((20, (x1 - x0) * upscale, 3), 0, dtype=np.uint8), "GT", color=(255, 255, 255))
    header_right = _label(np.full((20, (x1 - x0) * upscale, 3), 0, dtype=np.uint8),
                           "Warped (red = field)", color=(255, 255, 255))
    header = np.concatenate([header_left, np.full((20, 2, 3), 255, dtype=np.uint8), header_right], axis=1)

    frames = []
    for t in range(cycle.shape[0]):
        rows = [header]
        for name, *_ in levels:
            left = np.stack([_to_frame(cycle[t], y0, y1, x0, x1, upscale)] * 3, axis=-1)
            left = _label(left, name)
            right = _to_frame(warped_by_level[name][t], y0, y1, x0, x1, upscale)
            disp_h, disp_w = field_by_level[name]
            right = _draw_arrows(right, disp_h, disp_w, y0, x0, upscale, step=arrow_step)
            sep = np.full((left.shape[0], 2, 3), 255, dtype=np.uint8)
            rows.append(np.concatenate([left, sep, right], axis=1))
        sep_row = np.full((2, rows[0].shape[1], 3), 255, dtype=np.uint8)
        grid = rows[0]
        for r in rows[1:]:
            grid = np.concatenate([grid, sep_row, r], axis=0)
        frames.append(grid)

    if pingpong:
        frames = frames + frames[-2:0:-1]

    dirname = os.path.dirname(out_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=FPS, loop=0)
    print(f"saved {out_path}, frame shape {frames[0].shape}, levels={levels}")


def main():
    generate()


if __name__ == "__main__":
    main()
