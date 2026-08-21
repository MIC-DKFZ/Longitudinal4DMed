"""
LAUGEN parameter calibration utility.

Grid-searches deform_structure()'s generation parameters against real paired
frames from the same patient, and reports the setting whose *distribution* of
synthetic endpoints best matches the distribution of real endpoints -- not a
per-pair fit. LAUGEN is stochastic (random blob directions/counts/positions),
so for a fixed parameter set theta and a fixed input A, deform_structure(A, ...)
is a sample from a distribution, not a deterministic warp toward any specific B.
Calibration therefore compares *distributions* of a masked change-magnitude
summary statistic (real deltas vs. many synthetic-seed deltas), never a single
pair's warp error.

Does not modify deform_structure or any other part of the package -- this is a
pure consumer of the public deform_structure() API.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import wasserstein_distance, ks_2samp

from .deform import deform_structure

# Fixed grid -- see the handoff doc, Section 4. Not adaptive, not Bayesian.
PARAM_GRID = {
    "intensity": [1.0, 3.0, 5.0, 8.0, 12.0, 16.0],
    "growth_max": [1.5, 2.5, 4.0, 6.0],
    "blur_range": [(0.5, 1.0), (1.0, 2.0), (2.0, 4.0)],
    "n_blobs": [1, 2, 3],
}

# Fixed during calibration, not searched: every draw should actually deform,
# so variability in the endpoint distribution comes from the grid params
# above, not from apply_prob sometimes no-op'ing a blob.
_APPLY_PROB = 1.0
_TIME_POINTS = np.linspace(0, 1, 10)


def masked_signal(a: np.ndarray, b: np.ndarray, seg: np.ndarray) -> float:
    """Scalar summary of the change between two frames, restricted to the segmentation."""
    mask = binary_dilation(seg, iterations=2).astype(bool)
    return float(np.sqrt(((a[mask] - b[mask]) ** 2).mean()))


def _iter_grid(param_grid: dict):
    keys = list(param_grid.keys())
    for values in itertools.product(*(param_grid[k] for k in keys)):
        yield dict(zip(keys, values))


def fit_params(
    pairs: list,
    structure_label: int,
    param_grid: dict = None,
    n_seeds: int = 3,
    metric: str = "wasserstein",
    verbose: bool = True,
) -> dict:
    """
    Grid-search LAUGEN parameters against real paired frames.

    Parameters
    ----------
    pairs : list of (A, B, seg_A) tuples
        A, B: paired frames, same patient, different timepoints. shape (H, W[, D])
        seg_A: integer segmentation for frame A, values include `structure_label`.
    structure_label : int
        Which anatomical label to calibrate for (e.g., 3 for LV in ACDC).
        Per-structure calibration -- call this function separately per structure.
    param_grid : dict, optional
        Overrides the default grid (see PARAM_GRID).
    n_seeds : int
        Random seeds per (pair, param setting). Higher = less variance, more compute.
    metric : {"wasserstein", "ks"}
        Distributional distance for scoring.
    verbose : bool
        Print progress every 10% of configs.

    Returns
    -------
    dict with keys:
        "best_params": dict of the optimal param setting
        "best_score": float, distance value at best params
        "all_results": list of (params_dict, score) tuples, sorted ascending by score
    """
    if param_grid is None:
        param_grid = PARAM_GRID
    if metric not in ("wasserstein", "ks"):
        raise ValueError(f"metric must be 'wasserstein' or 'ks', got {metric!r}")

    # Real deltas and per-pair binary structure masks don't depend on theta --
    # compute once, outside the grid loop.
    real_deltas = []
    seg_binaries = []
    for A, B, seg_A in pairs:
        seg_bin = (seg_A == structure_label).astype(np.float32)
        seg_binaries.append(seg_bin)
        real_deltas.append(masked_signal(A, B, seg_bin))
    real_deltas = np.asarray(real_deltas)

    configs = list(_iter_grid(param_grid))
    n_configs = len(configs)
    progress_step = max(1, n_configs // 10)

    all_results = []
    for i, params in enumerate(configs):
        synth_deltas = []
        for (A, B, seg_A), seg_bin in zip(pairs, seg_binaries):
            if seg_bin.sum() == 0:
                continue
            for seed in range(n_seeds):
                np.random.seed(seed)
                seq = deform_structure(
                    A, seg_bin, _TIME_POINTS,
                    intensity=params["intensity"],
                    growth_max=params["growth_max"],
                    blur_range=params["blur_range"],
                    n_blobs=params["n_blobs"],
                    apply_prob=_APPLY_PROB,
                    target_shape=A.shape,
                )
                synth_deltas.append(masked_signal(A, seq[-1], seg_bin))
        synth_deltas = np.asarray(synth_deltas)

        if metric == "wasserstein":
            score = wasserstein_distance(real_deltas, synth_deltas)
        else:
            score = float(ks_2samp(real_deltas, synth_deltas).statistic)

        all_results.append((params, float(score)))

        if verbose and (i + 1) % progress_step == 0:
            print(f"[fit_params] {i + 1}/{n_configs} configs done "
                  f"(structure_label={structure_label})")

    all_results.sort(key=lambda item: item[1])
    best_params, best_score = all_results[0]

    return {
        "best_params": best_params,
        "best_score": best_score,
        "all_results": all_results,
    }
