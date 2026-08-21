import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, binary_fill_holes
from scipy.interpolate import RegularGridInterpolator


def gaussian_kernel(shape, center, std_dev):
    """
    Truly 3D (or 2D) isotropic Gaussian centred at `center`.
    Previous implementation extruded a 2D kernel along depth — now fully volumetric.
    """
    if len(shape) == 2:
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        return np.exp(-((y - center[0]) ** 2 + (x - center[1]) ** 2) / (2 * std_dev ** 2))
    elif len(shape) == 3:
        y, x, z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij'
        )
        return np.exp(
            -((y - center[0]) ** 2 + (x - center[1]) ** 2 + (z - center[2]) ** 2)
            / (2 * std_dev ** 2)
        )
    else:
        raise ValueError("shape must be 2D or 3D")


def _warp_time(time_points: np.ndarray) -> np.ndarray:
    """
    Randomly apply a nonlinear monotone warp to the time axis.
    Modes: linear (identity), power-law, tanh (sigmoid-like).
    Keeps the sign and endpoint range of the original time_points.
    """
    mode = np.random.choice(['linear', 'power', 'tanh'])
    if mode == 'linear' or len(time_points) < 2:
        return time_points
    t_abs_max = np.abs(time_points).max()
    if t_abs_max < 1e-8:
        return time_points
    t_norm = time_points / t_abs_max  # [-1, 1]
    if mode == 'power':
        alpha = np.random.uniform(0.4, 2.5)
        return np.sign(t_norm) * np.abs(t_norm) ** alpha * t_abs_max
    else:  # tanh
        k = np.random.uniform(0.8, 2.5)
        warped = np.tanh(t_norm * k) / np.tanh(k)
        return warped * t_abs_max


def deform_structure(
    img,
    seg,
    time_points,
    intensity=20.0,
    growth_max=5.0,
    apply_prob=0.5,
    target_shape=(96, 96, 64),
    planar_only=False,
    global_noise_std=0.0,
    global_noise_sigma=8.0,
    n_blobs=2,
    time_nonlinear=True,
    blur_range=(2, 7),
    fill_holes=True,
    fill_holes_axis=None,
):
    """
    Apply anatomy-guided, time-varying deformation to `img`.

    Args:
        img:               (H, W, D) source image (float32, normalised [0,1])
        seg:               (H, W, D) binary segmentation mask
        time_points:       1-D array in [0, 1]; flip externally for shrinkage sequences
        intensity:         base deformation amplitude (pixels)
        growth_max:        upper multiplier drawn from U(2, growth_max)
        apply_prob:        probability that a blob is Gaussian-localised (vs global)
        target_shape:      output spatial shape; each frame is resampled to this size
        planar_only:       force depth component of direction vectors to zero
        global_noise_std:  amplitude of smooth global displacement field (0 = disabled)
        global_noise_sigma: smoothing sigma for the global noise field
        n_blobs:           max number of blobs; actual count drawn from U(1, n_blobs)
        time_nonlinear:    if True, randomly warp the time axis (power-law or tanh)
        blur_range:        (lo, hi) for the boundary-gradient Gaussian filter sigma,
                            drawn from U(lo, hi) each call. Controls how far the
                            anatomy-guided displacement fades out around the
                            structure boundary — lower = sharper, more localised
                            displacement (less "halo"/ghosting on the surrounding
                            tissue); higher = softer, wider-reaching displacement.
                            Pass (v, v) for a fixed, non-random value.
        fill_holes:        if True (default), fill enclosed interior holes in `seg`
                            before computing the boundary gradient field. A
                            ring-/donut-shaped structure (e.g. myocardium around
                            the LV cavity) otherwise has a spurious *inner*
                            boundary too, and blobs anchored there displace tissue
                            in ways that look like ghosting/double edges near the
                            cavity — filling treats the structure as a single
                            solid silhouette instead.
        fill_holes_axis:   if not None, fill holes per 2-D slice along this axis
                            instead of (or in addition to) the plain n-D fill.
                            Needed when the boundary-gradient Gaussian's sigma
                            (`blur_range`) is much smaller than the volume's extent
                            along one axis (e.g. a thin through-plane/depth axis
                            with few slices) — a cavity that is topologically open
                            in full n-D (connects out through slices near the
                            volume's edge along that axis) is still, locally, a
                            hole in every slice the blur actually sees. Axis index
                            into `seg`'s dims, e.g. 2 for a (H, W, D) volume with D
                            as the through-plane axis.

    Returns:
        np.ndarray of shape (T, *target_shape), float32
    """
    dim = img.shape
    n_dim = len(dim)

    time_points_eff = _warp_time(np.asarray(time_points, dtype=float)) if time_nonlinear else np.asarray(time_points, dtype=float)

    tmp = [np.arange(i) for i in dim]
    coords_base = np.array(np.meshgrid(*tmp, indexing='ij')).astype(float)
    for d in range(n_dim):
        coords_base[d] -= (np.array(dim).astype(float) - 1)[d] / 2.0

    organ_mask = (seg >= 1)
    if fill_holes:
        if fill_holes_axis is not None:
            organ_mask = organ_mask.copy()
            for idx in range(organ_mask.shape[fill_holes_axis]):
                sl = [slice(None)] * n_dim
                sl[fill_holes_axis] = idx
                sl = tuple(sl)
                organ_mask[sl] = binary_fill_holes(organ_mask[sl])
        else:
            organ_mask = binary_fill_holes(organ_mask)

    spacing_ratio = 1.0
    blur = np.random.uniform(*blur_range)  # randomise boundary sharpness
    organ_blurred = gaussian_filter(
        organ_mask.astype(float),
        sigma=(blur * spacing_ratio, blur, blur) if n_dim == 3 else (blur, blur),
        order=0,
        mode='nearest',
    )
    grads = np.gradient(organ_blurred)
    vec_metric = sum(g ** 2 for g in grads) ** 0.5
    non_zero_pts = np.array(np.nonzero(vec_metric > 0.1)).T  # (N, n_dim)

    # Randomly choose how many blobs to activate this sample
    n_active = np.random.randint(1, max(2, n_blobs) + 1)

    blob_params = []
    for _ in range(n_active):
        mag = intensity * np.random.uniform(2, growth_max)
        if np.random.rand() < 0.5:
            mag *= -1

        # Always draw a random unit-vector direction — fixes previous diagonal bias
        vec = np.random.randn(n_dim)
        if planar_only and n_dim == 3:
            vec[2] = 0.0
        norm = np.linalg.norm(vec)
        vhat = vec / norm if norm > 1e-8 else np.ones(n_dim) / np.sqrt(n_dim)

        # Randomised Gaussian spatial spread
        std_dev = np.random.uniform(10, 40)

        # 35% chance: purely Gaussian blob not anchored to seg boundary
        # simulates satellite lesion or edema independent of the existing structure
        if np.random.rand() < 0.35 or len(non_zero_pts) == 0:
            center = tuple(np.random.randint(0, s) for s in dim)
            gauss = gaussian_kernel(dim, center, std_dev)
            blob_params.append({'mag': mag, 'gauss': gauss, 'vhat': vhat, 'guided': False})
        elif np.random.rand() < apply_prob:
            pt = non_zero_pts[np.random.choice(len(non_zero_pts))]
            gauss = gaussian_kernel(dim, pt, std_dev)
            blob_params.append({'mag': mag, 'gauss': gauss, 'vhat': vhat, 'guided': True})
        else:
            # global (no Gaussian mask)
            blob_params.append({'mag': mag, 'gauss': 1.0, 'vhat': vhat, 'guided': True})

    # Global smooth displacement field — constant across time
    if global_noise_std > 0.0:
        global_disp = [
            gaussian_filter(
                np.random.randn(*dim).astype(np.float64) * global_noise_std,
                sigma=global_noise_sigma,
            )
            for _ in range(n_dim)
        ]
    else:
        global_disp = None

    res_list = []
    for tp_eff in time_points_eff:
        coords = coords_base.copy()
        for blob in blob_params:
            m, g, v = blob['mag'], blob['gauss'], blob['vhat']
            if blob['guided']:
                # anatomy-guided: displace along seg-gradient weighted by Gaussian
                for d, grad in enumerate(grads):
                    coords[d] += grad * m * g * v[d] * tp_eff
            else:
                # free Gaussian blob: displace in random direction, Gaussian-weighted
                for d in range(n_dim):
                    coords[d] += g * v[d] * m * tp_eff
        if global_disp is not None:
            for d in range(n_dim):
                coords[d] += global_disp[d]
        for d in range(n_dim):
            coords[d] += img.shape[d] / 2.0 - 0.5

        warped = map_coordinates(img, coords, order=1, mode='nearest')

        new_dims = [
            np.linspace(0, warped.shape[i] - 1, target_shape[i])
            for i in range(n_dim)
        ]
        coords_down = np.meshgrid(*new_dims, indexing='ij')
        new_coords = np.stack([g.ravel() for g in coords_down], axis=-1)
        original_grid = tuple(
            np.linspace(0, warped.shape[i] - 1, warped.shape[i])
            for i in range(n_dim)
        )
        interpolator = RegularGridInterpolator(
            original_grid, warped, method='linear', bounds_error=False, fill_value=0.0
        )
        res_list.append(interpolator(new_coords).reshape(target_shape).astype(np.float32))

    return np.stack(res_list)  # (T, *target_shape)


def apply_bias_field(
    seq: np.ndarray,
    seg: np.ndarray,
    coeff: float = 0.5,
    std_dev: float = 4.0,
) -> np.ndarray:
    """
    Apply a smooth multiplicative bias field to a temporal sequence.

    The field center is sampled from the segmentation mask (anatomy-guided),
    matching the approach in LongiDatasetSemiV2._bias_field_augmentation.
    The coefficient ramps linearly from 0 at the first frame to `coeff` at
    the last, simulating progressive scanner inhomogeneity.

    Args:
        seq:      (T, *spatial) float32 sequence
        seg:      (*spatial) binary mask — used to weight the center sample
        coeff:    peak bias amplitude (at the final frame)
        std_dev:  Gaussian spread of the bias field in pixels

    Returns:
        (T, *spatial) float32, globally normalised to [0, max].
    """
    T = seq.shape[0]
    spatial = seq.shape[1:]

    flat = seg.ravel().astype(float)
    if flat.sum() > 1e-8:
        probs = flat / flat.sum()
        center = np.unravel_index(np.random.choice(len(probs), p=probs), seg.shape)
    else:
        center = tuple(s // 2 for s in spatial)

    result = seq.copy()
    for i in range(T):
        t_coeff = coeff * i / max(T - 1, 1)
        field = t_coeff * gaussian_kernel(spatial, center, std_dev)
        result[i] = seq[i] * (1.0 + field)

    norm_max = result.max()
    if norm_max > 1e-8:
        result /= norm_max
    return result.astype(np.float32)


def apply_seg_intensity(
    seq: np.ndarray,
    seg: np.ndarray,
    alpha: float = 0.15,
    smooth_sigma: float = 6.0,
) -> np.ndarray:
    """
    Add a time-varying intensity change localised to the segmentation mask.

    The mask is Gaussian-smoothed before use so intensity edges are soft rather
    than binary.  The amplitude ramps linearly from 0 at the first frame to
    `alpha` at the last — simulating progressive signal change (enhancement,
    FLAIR increase) co-located with the lesion.  Sign is randomised so the
    lesion can brighten or darken.

    Args:
        seq:          (T, *spatial) float32 sequence
        seg:          (*spatial) binary mask
        alpha:        peak intensity change at the last frame (in normalised units)
        smooth_sigma: Gaussian smoothing applied to the mask before scaling

    Returns:
        (T, *spatial) float32, clipped to [0, 1]
    """
    soft_mask = gaussian_filter(seg.astype(np.float32), sigma=smooth_sigma)
    if soft_mask.max() > 1e-8:
        soft_mask /= soft_mask.max()

    sign = 1.0 if np.random.rand() < 0.5 else -1.0
    T = seq.shape[0]
    result = seq.copy()
    for i in range(T):
        t_norm = i / max(T - 1, 1)
        result[i] = seq[i] + sign * alpha * t_norm * soft_mask

    return np.clip(result, 0.0, 1.0).astype(np.float32)
