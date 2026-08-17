"""
Prepares ISLES-2024 raw data into the per-patient isles3/npy_data/<patient>.npz
layout that isles_loader.py's ISLESDataset uses by default (isles_seg_mode='real').

Each output .npz holds, per patient, on a shared resampled (1.5, 1.5, 5.0) mm grid:
    ctp          (H, W, D, T)  brain-extracted, bias-uncorrected CT perfusion series
    cbf, cbv     (H, W, D)     perfusion maps (used to derive a real ROI segmentation
                                target on the fly — see isles_loader.py's
                                _make_perfusion_mask)
    cow_mask     (H, W, D)     circle-of-Willis mask (if provided in derivatives)
    lvo_mask     (H, W, D)     large-vessel-occlusion expert annotation
    lesion_mask  (H, W, D)     ischemic lesion expert annotation

Data access: ISLES-2024 is a MICCAI challenge dataset distributed under its own
registration / data-use agreement through the official challenge organizers —
it is not a plain public download. Register for the challenge, accept its DUA,
and download the training data before running this script.

Expected raw layout (as distributed by the challenge, BIDS-like):
    <RAW_DIR>/<patient>/ses-01/*ctp*.nii.gz
    <RAW_DIR>/<patient>/ses-01/perfusion-maps/*tmax*.nii.gz
    <RAW_DIR>/<patient>/ses-01/perfusion-maps/*cbf*.nii.gz
    <RAW_DIR>/<patient>/ses-01/perfusion-maps/*cbv*.nii.gz
    <DERIVATIVES_DIR>/<patient>/ses-*/  containing *cow-msk*, *lvo-msk*, and
        *lesion-msk*/*isch-msk* NIfTI files (whichever the challenge release names
        them; not every patient necessarily has all three)

Usage:
    python src/data_loaders/isles_prepare.py

Requires nilearn (pip install nilearn) in addition to this repo's core requirements
— only for this prep script, not for training/eval.
"""
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion

DATA_DIR = os.getenv("DATA_DIR", "./data/")
RAW_DIR = os.path.join(DATA_DIR, "isles3", "train", "raw_data")
DERIVATIVES_DIR = os.path.join(DATA_DIR, "isles3", "train", "derivatives")
SAVE_DIR = os.path.join(DATA_DIR, "isles3", "npy_data")

TARGET_SPACING = np.array([1.5, 1.5, 5.0])
TARGET_SHAPE = (112, 112, 16)


def filter_and_normalize(data, lower_percentile=2, upper_percentile=99):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    data_clipped = np.clip(data, lower_bound, upper_bound)
    return (data_clipped - data_clipped.min()) / (data_clipped.max() - data_clipped.min())


def find_session_dir(patient_deriv_dir, session_prefix="ses-01"):
    """Find session dir, handling ses-01 vs ses-0001 style naming differences."""
    if not os.path.exists(patient_deriv_dir):
        return None
    for d in os.listdir(patient_deriv_dir):
        if d.startswith(session_prefix):
            return os.path.join(patient_deriv_dir, d)
    session_num = session_prefix.split("-")[-1].lstrip("0") or "0"
    for d in os.listdir(patient_deriv_dir):
        if session_num in d:
            return os.path.join(patient_deriv_dir, d)
    return None


def get_bounding_box(mask, margin=0):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return [(0, s) for s in mask.shape]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0) + 1
    bbox = []
    for i in range(3):
        lo = max(0, mins[i] - margin)
        hi = min(mask.shape[i], maxs[i] + margin)
        bbox.append((lo, hi))
    return bbox


def crop_or_pad(img, target_shape):
    """Crop or pad a 4D array (H, W, D, T) spatially to target_shape (H, W, D)."""
    for dim in range(3):
        current, target = img.shape[dim], target_shape[dim]
        if current > target:
            start = (current - target) // 2
            slices = [slice(None)] * 4
            slices[dim] = slice(start, start + target)
            img = img[tuple(slices)]
        elif current < target:
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_widths = [(0, 0)] * 4
            pad_widths[dim] = (pad_before, pad_after)
            img = np.pad(img, pad_widths, mode="constant", constant_values=0)
    return img


def crop_or_pad_3d(img, target_shape):
    """Crop or pad a 3D array (H, W, D) to target_shape."""
    for dim in range(3):
        current, target = img.shape[dim], target_shape[dim]
        if current > target:
            start = (current - target) // 2
            slices = [slice(None)] * 3
            slices[dim] = slice(start, start + target)
            img = img[tuple(slices)]
        elif current < target:
            pad_before = (target - current) // 2
            pad_after = target - current - pad_before
            pad_widths = [(0, 0)] * 3
            pad_widths[dim] = (pad_before, pad_after)
            img = np.pad(img, pad_widths, mode="constant", constant_values=0)
    return img


def preprocess():
    from nilearn.image import resample_img, resample_to_img

    assert os.path.isdir(RAW_DIR), (
        f"{RAW_DIR} is not a directory — see this file's docstring for the expected "
        f"raw ISLES-2024 layout and data-access process."
    )
    os.makedirs(SAVE_DIR, exist_ok=True)

    patient_counter = 0
    cropped_shapes = []

    for patient in sorted(os.listdir(RAW_DIR)):
        print(f"\n=== [{patient_counter}] {patient} ===")
        patient_dir = os.path.join(RAW_DIR, patient, "ses-01")

        deriv_ses_dir = find_session_dir(os.path.join(DERIVATIVES_DIR, patient), "ses-01")
        if deriv_ses_dir is None:
            print("  SKIPPING: no derivatives session dir found")
            continue

        # --- 1. Load CTP and build a target affine at TARGET_SPACING ---
        ctp_str = [f for f in os.listdir(patient_dir) if "ctp" in f][0]
        ctp_nib = nib.load(os.path.join(patient_dir, ctp_str))
        print(f"  CTP orig: {ctp_nib.shape} @ {np.array(ctp_nib.header.get_zooms()[:3]).round(2)}")

        target_affine = ctp_nib.affine.copy()
        for i in range(3):
            axis = target_affine[:3, i]
            target_affine[:3, i] = axis / np.linalg.norm(axis) * TARGET_SPACING[i]

        # --- 2. Resample CTP to target spacing ---
        ctp_resampled_nib = resample_img(ctp_nib, target_affine=target_affine, interpolation="linear")
        ctp = ctp_resampled_nib.get_fdata().astype(np.float32)
        print(f"  CTP resampled: {ctp.shape}")

        # --- 3. Load & resample perfusion maps (same native space as CTP) ---
        perfusion_dir = os.path.join(patient_dir, "perfusion-maps")
        tmax_str = [f for f in os.listdir(perfusion_dir) if "tmax" in f][0]
        cbf_str = [f for f in os.listdir(perfusion_dir) if "cbf" in f][0]
        cbv_str = [f for f in os.listdir(perfusion_dir) if "cbv" in f][0]
        tmax = resample_img(nib.load(os.path.join(perfusion_dir, tmax_str)),
                            target_affine=target_affine, interpolation="linear").get_fdata().astype(np.float32)
        cbf = resample_img(nib.load(os.path.join(perfusion_dir, cbf_str)),
                           target_affine=target_affine, interpolation="linear").get_fdata().astype(np.float32)
        cbv = resample_img(nib.load(os.path.join(perfusion_dir, cbv_str)),
                           target_affine=target_affine, interpolation="linear").get_fdata().astype(np.float32)
        print(f"  Tmax: {tmax.shape}, CBF: {cbf.shape}, CBV: {cbv.shape}")

        # --- 4. Brain mask from tmax, eroded to strip the skull boundary ---
        brain_mask = tmax > tmax.min()
        brain_mask = binary_erosion(brain_mask, iterations=3)
        ctp = ctp * brain_mask[..., None]
        cbf = cbf * brain_mask
        cbv = cbv * brain_mask
        print(f"  Brain mask after erosion: {brain_mask.sum()} voxels")

        # --- 5. Load & resample CoW / LVO / lesion masks onto the same grid as ctp ---
        ref_3d_nib = nib.Nifti1Image(ctp[..., 0], ctp_resampled_nib.affine)

        def _load_deriv_mask(patterns, fallback=None):
            for pat in patterns:
                files = [f for f in os.listdir(deriv_ses_dir) if pat in f]
                if files:
                    m = nib.load(os.path.join(deriv_ses_dir, files[0]))
                    return resample_to_img(m, ref_3d_nib, interpolation="nearest").get_fdata().astype(np.float32)
            return fallback

        cow = _load_deriv_mask(["cow-msk"], fallback=np.zeros(ctp.shape[:3], dtype=np.float32))
        lvo_mask = _load_deriv_mask(["lvo-msk"], fallback=np.zeros(ctp.shape[:3], dtype=np.float32))
        lesion_mask = _load_deriv_mask(["lesion-msk", "isch-msk"], fallback=lvo_mask.copy())
        print(f"  CoW mask sum={cow.sum():.0f}, LVO mask sum={lvo_mask.sum():.0f}, "
              f"lesion mask sum={lesion_mask.sum():.0f}")

        # --- 6. Brain bounding-box crop (no margin) ---
        bbox = get_bounding_box(brain_mask, margin=0)
        s = tuple(slice(b[0], b[1]) for b in bbox)
        ctp = ctp[s[0], s[1], s[2], :]
        cbf, cbv = cbf[s[0], s[1], s[2]], cbv[s[0], s[1], s[2]]
        cow, lvo_mask, lesion_mask = cow[s[0], s[1], s[2]], lvo_mask[s[0], s[1], s[2]], lesion_mask[s[0], s[1], s[2]]
        cropped_shapes.append(ctp.shape[:3])
        print(f"  After bbox crop: CTP={ctp.shape}")

        if ctp.shape[2] < 4:
            print(f"  SKIPPING {patient}: depth={ctp.shape[2]} < 4 after crop")
            continue

        # --- 7. Normalize CTP, crop/pad everything to TARGET_SHAPE ---
        ctp = filter_and_normalize(ctp)
        ctp = crop_or_pad(ctp, TARGET_SHAPE)
        cow = crop_or_pad_3d(cow, TARGET_SHAPE)
        lvo_mask = crop_or_pad_3d(lvo_mask, TARGET_SHAPE)
        lesion_mask = crop_or_pad_3d(lesion_mask, TARGET_SHAPE)
        cbf = crop_or_pad_3d(cbf, TARGET_SHAPE)
        cbv = crop_or_pad_3d(cbv, TARGET_SHAPE)
        print(f"  After crop/pad: CTP={ctp.shape}")

        # --- 8. Save ---
        np.savez_compressed(
            os.path.join(SAVE_DIR, f"{patient}.npz"),
            ctp=ctp, cbf=cbf, cbv=cbv, cow_mask=cow, lvo_mask=lvo_mask, lesion_mask=lesion_mask,
        )
        patient_counter += 1

    if cropped_shapes:
        cropped_shapes = np.array(cropped_shapes)
        print(f"\n=== Cropped Shape Summary ===")
        print(f"Min:    {cropped_shapes.min(axis=0)}")
        print(f"Max:    {cropped_shapes.max(axis=0)}")
        print(f"Median: {np.median(cropped_shapes, axis=0).astype(int)}")
    print(f"\nFinished: {patient_counter} patients saved to {SAVE_DIR}")


if __name__ == "__main__":
    preprocess()
