"""
Calibrate LAUGEN's generation parameters against real ACDC ED->ES pairs.

Loads ~20 ACDC patients' ED (frame 0) / ES (last frame) pairs, builds one 2D
slice per patient (picked the same way make_hero_gif.py does -- the depth
slice with the most total labeled voxels, kept shared across all 3
structures so the same `pairs` list is reused for every fit_params() call),
and runs laugen.calibration.fit_params() once per ACDC structure (RV=1,
Myocardium=2, LV=3).

Does NOT save results anywhere and does NOT modify LAUGEN's defaults with
the numbers it finds -- that's a separate decision for later, after eyeballing
this script's output.

Data: point ACDC_DATA_DIR at a directory containing trn_dat.npy / trn_seg.npy
(defaults to the SADM/Synthwave repo's own data/ dir -- see laugen memory note
"ACDC data path", same convention as make_hero_gif.py).

Usage:
    python calibrate_acdc.py
    ACDC_DATA_DIR=/path/to/data python calibrate_acdc.py
"""
import os

import numpy as np

from laugen.calibration import fit_params, PARAM_GRID

ACDC_DATA_DIR = os.environ.get(
    "ACDC_DATA_DIR",
    os.path.expanduser(
        "~/PycharmProjects/Synthwave/SADM-Longitudinal-Medical-Image-Generation/data"
    ),
)

N_PATIENTS = 20
N_SEEDS = 3
STRUCTURES = [("RV", 1), ("Myocardium", 2), ("LV", 3)]
MIN_LABELED_VOXELS = 50  # skip a patient's slice if too few labeled voxels


def build_pairs():
    data = np.load(os.path.join(ACDC_DATA_DIR, "trn_dat.npy"), mmap_mode="r")
    seg = np.load(os.path.join(ACDC_DATA_DIR, "trn_seg.npy"), mmap_mode="r")

    pairs = []
    for pidx in range(N_PATIENTS):
        A = np.asarray(data[pidx, 0]).astype(np.float32)   # (H, W, D), ED
        B = np.asarray(data[pidx, -1]).astype(np.float32)  # (H, W, D), ES
        A = (A - A.min()) / (A.max() - A.min() + 1e-8)
        B = (B - B.min()) / (B.max() - B.min() + 1e-8)
        seg_full = np.round(np.asarray(seg[pidx, ..., 0])).astype(np.int32)  # (H, W, D)

        per_slice = (seg_full > 0).sum(axis=(0, 1))
        depth = int(np.argmax(per_slice))
        if per_slice[depth] < MIN_LABELED_VOXELS:
            continue

        pairs.append((A[:, :, depth], B[:, :, depth], seg_full[:, :, depth]))

    return pairs


def print_top5(name, result):
    print(f"\nTop 5 param settings for {name}:")
    for rank, (params, score) in enumerate(result["all_results"][:5], start=1):
        print(f"  {rank}. score={score:.5f}  {params}")


def main():
    pairs = build_pairs()

    results = {}
    for name, label in STRUCTURES:
        print(f"\n=== Calibrating {name} (label={label}) over {len(pairs)} pairs ===")
        result = fit_params(pairs, structure_label=label, n_seeds=N_SEEDS, verbose=True)
        results[name] = result
        print_top5(name, result)

    n_configs = 1
    for v in PARAM_GRID.values():
        n_configs *= len(v)

    best_str = ", ".join(f"{n}: {r['best_params']}" for n, r in results.items())
    print(f"\ncalibrated {len(pairs)} pairs x {n_configs} configs x {N_SEEDS} seeds, "
          f"structures={[s[0] for s in STRUCTURES]}, best={{{best_str}}}")


if __name__ == "__main__":
    main()
