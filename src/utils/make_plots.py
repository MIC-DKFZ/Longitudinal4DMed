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


def save_overlay_gif(images, flows, path, fps: int = 4, alpha: float = 0.3, norm_percentile: int = 99):
    """Save a GIF with a red-blue flow overlay blended onto a grayscale background.

    images:          (T, H, W) grayscale tensor — context + target frames
    flows:           (T, H, W) signed difference tensor — same length as images
    alpha:           max opacity of the flow colour; scaled per-pixel by flow magnitude
    norm_percentile: flow values are normalised by this percentile of all absolute
                     values, so small-but-real cardiac motion stays visible
    """
    # Collect all flow magnitudes for percentile-based normalisation.
    # Using max would let a single outlier frame wash out the rest.
    all_abs = np.concatenate([
        (f.detach().cpu().float().numpy() if torch.is_tensor(f) else np.asarray(f, float)).ravel()
        for f in flows
    ])
    nonzero_abs = all_abs[np.abs(all_abs) > 1e-8]
    global_scale = (float(np.percentile(nonzero_abs, norm_percentile)) + 1e-8
                    if len(nonzero_abs) > 0 else 1.0)

    frames = []
    for img, flow in zip(images, flows):
        img_np = img.detach().cpu().float().numpy().squeeze() if torch.is_tensor(img) else np.asarray(img, float).squeeze()
        flow_np = flow.detach().cpu().float().numpy().squeeze() if torch.is_tensor(flow) else np.asarray(flow, float).squeeze()

        # Grayscale background → RGB [0, 1]
        lo, hi = img_np.min(), img_np.max()
        g = (img_np - lo) / (hi - lo + 1e-8)
        bg = np.stack([g, g, g], axis=-1)  # (H, W, 3)

        # Red = positive change, Blue = negative change; clip outliers to ±1
        f = np.clip(flow_np / global_scale, -1, 1)
        overlay = np.stack([np.clip(f, 0, 1), np.zeros_like(f), np.clip(-f, 0, 1)], axis=-1)

        # Magnitude-weighted blend: static background stays pure grey
        eff_alpha = alpha * np.abs(f)[..., None]
        composite = np.clip((1 - eff_alpha) * bg + eff_alpha * overlay, 0, 1)
        frames.append((composite * 255).astype(np.uint8))

    imageio.mimsave(str(path), frames, fps=fps, loop=0)


def _sample_to_gifs(sample, results_dir, name, fps=4):
    """Generate plain and overlay GIFs from a dataset sample dict.

    Handles both torch-tensor and numpy-array samples.
    Expected keys: 'context' (T, C, D, H, W), 'target_img' (1, C, D, H, W).
    """
    def _to_tensor(x):
        if torch.is_tensor(x):
            return x.float()
        return torch.from_numpy(np.array(x, dtype=np.float32))

    ctx = _to_tensor(sample['context'])
    tgt = _to_tensor(sample['target_img'])
    all_img = torch.cat([ctx, tgt], dim=0)               # (T, C, D, H, W)
    imgs = all_img[:, 0, all_img.shape[2] // 2, :, :]   # mid-depth slice → (T, H, W)

    flows = torch.diff(imgs, dim=0)
    flows = torch.cat([torch.zeros_like(flows[:1]), flows], dim=0)

    # Suppress diffs at transitions from/to missing (all-zero) frames
    is_missing = imgs.flatten(1).abs().sum(1) == 0
    prev_missing = torch.cat([is_missing.new_tensor([False]), is_missing[:-1]])
    flows[is_missing | prev_missing] = 0.0

    plain_path = os.path.join(results_dir, f"{name}_example.gif")
    save_gif(imgs, plain_path, fps=fps)
    print(f"Saved plain GIF:   {plain_path}")

    overlay_path = os.path.join(results_dir, f"{name}_overlay.gif")
    save_overlay_gif(imgs, flows, overlay_path, fps=fps)
    print(f"Saved overlay GIF: {overlay_path}")


def _build_dataset(name, data_dir, debug=True, val_split=0):
    if name == 'acdc':
        from data_loaders.acdc_loader import ACDCDataset
        return ACDCDataset(data_dir=os.path.join(data_dir, 'ACDC'), split='val',
                           num_to_keep_context=12, debug=debug, val_split=val_split)
    if name == 'isles':
        from data_loaders.isles_loader import ISLESDataset
        return ISLESDataset(data_dir=data_dir, train_test_val='val',
                            num_to_keep_context=5, debug=debug, val_split=val_split, dense=True)
    if name == 'lumiere':
        from data_loaders.lumiere_loader import LumiereDataset
        # NOTE: LumiereDataset's split kwarg is train_test_val, not split — matches the same
        # fix applied to data_utils.build_dataloader this session (was silently always 'trn').
        # TODO: review comment
        return LumiereDataset(data_dir=data_dir, train_test_val='val',
                              num_to_keep_context=5, debug=debug, val_split=val_split)
    if name == 'oasis':
        from data_loaders.oasis_loader import OASISDataset
        # NOTE: same split-kwarg fix as lumiere above.
        # TODO: review comment
        return OASISDataset(data_dir=data_dir, train_test_val='val',
                            num_to_keep_context=5, debug=debug, val_split=val_split)
    if name == 'aimi':
        from data_loaders.aimi_loader import AimiDataset
        return AimiDataset(data_dir=data_dir, train_test_val='val', debug=debug)
    raise ValueError(f"Unknown dataset '{name}'. Add it to _build_dataset().")


def _extract_3d(x):
    """Squeeze a target_img/target_seg tensor down to a plain (D, H, W) volume,
    regardless of how many leading (T, C, ...) dims a given dataset's convention adds.
    # TODO: review comment
    """
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > 3:
        x = x[0]
    return x


def semisynth_deform_gif(sample, results_dir, name, structure='all', num_frames=8, fps=4, seed=None):
    """Generate a synthetic-longitudinal-deformation GIF via SemiSynthLongi.deform_structure,
    seeded from one real dataset sample's target_img + target_seg.

    Dataset-agnostic: deform_structure only ever needs an (image, label) pair, so this works on
    any dataset with a real segmentation target (ACDC's multi-structure cardiac segmentation,
    OASIS's CSF/hippocampus masks, aimi's LV mask, ...), not just BraTS — no real BraTS data
    needed to exercise this augmentation.
    # TODO: review comment

    structure: 'all' merges every nonzero label (deform_structure's own convention, label>=1);
               or a specific label value (e.g. '3' for ACDC's LV) to deform just that structure.
    """
    from data_loaders.brats_semi_loader import SemiSynthLongi

    class _DeformDemo(SemiSynthLongi):
        def __init__(self):
            self.two_dim = False
            self.memory_efficient = True

    img = _extract_3d(sample['target_img'])
    label = _extract_3d(sample['target_seg'])
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if structure != 'all':
        label = (label == float(structure)).astype(np.float32)
    else:
        # Some datasets' seg masks aren't strictly binary (e.g. Lumiere's linear-interpolation
        # resize can leave fractional/>1 values) — deform_structure's internal Gaussian-weighting
        # step blurs the mask and thresholds its gradient at a fixed 0.1, which a smeared-out
        # fractional mask can fail to cross anywhere, crashing deep inside it. Binarize first so
        # the gradient is always as sharp as the mask's true boundary, regardless of a given
        # dataset's own resize pipeline.
        # TODO: review comment
        label = (label >= 0.5).astype(np.float32)

    # deform_structure needs at least one voxel with a real label gradient to seed its
    # deformation field — an all-zero mask (e.g. a real clinical sample with no visible
    # structure at that timepoint) would otherwise crash deep inside it with a cryptic
    # numpy error. Fail with a clear, actionable message instead.
    # TODO: review comment
    if not (label > 0).any():
        raise ValueError(
            f"target_seg for this sample is all-zero for structure={structure!r} — "
            f"deform_structure needs a real segmentation to deform. Try a different "
            f"--sample_idx (or omit --longest) for this dataset."
        )

    if seed is not None:
        np.random.seed(seed)
    demo = _DeformDemo()
    time_points = np.linspace(-3, 3, num_frames)
    seq = demo.deform_structure(modality=0, context_img=img, label_img=label,
                                intensity=10, time_point=time_points, growth_max=5, apply_prob=1.0)

    mid = seq.shape[-1] // 2
    frames = []
    for t in range(seq.shape[0]):
        sl = seq[t, :, :, mid]
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        frames.append((sl * 255).astype(np.uint8))
    out_path = os.path.join(results_dir, f"{name}_semisynth_deform_{structure}.gif")
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"Saved semisynth-deform GIF: {out_path}")


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser(description="Generate example GIFs for a dataset.")
    parser.add_argument('--dataset', type=str, default='isles',
                        choices=['acdc', 'isles', 'lumiere', 'oasis', 'aimi'],
                        help='Which dataset to visualise.')
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--val_split', type=int, default=0)
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Which dataset sample to use (0 = first).')
    parser.add_argument('--longest', action='store_true',
                        help='Pick the sample with the most real (non-padded) context frames '
                             'instead of --sample_idx, for a more visually interesting sequence.')
    parser.add_argument('--deform_demo', action='store_true',
                        help='Also generate a SemiSynthLongi.deform_structure synthetic-'
                             'deformation GIF from this sample (needs a real target_seg — '
                             'works on any dataset, not just BraTS).')
    parser.add_argument('--deform_structure', type=str, default='all',
                        help="Label value to deform (e.g. '3' for ACDC's LV), or 'all' to merge "
                             "every nonzero label into one structure.")
    parser.add_argument('--deform_frames', type=int, default=8)
    parser.add_argument('--deform_seed', type=int, default=None)
    args = parser.parse_args()

    data_dir = os.getenv("DATA_DIR", "./data")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    dataset = _build_dataset(args.dataset, data_dir, debug=True, val_split=args.val_split)

    if args.longest:
        # dataset.data is a list of per-patient records for every loader in this repo;
        # "longest" = most real (pre-padding) timepoints, e.g. LumiereDataset.data[i] is a
        # dict of visits, OASISDataset.data[i]['dates'] is a list of visit dates, etc.
        # TODO: review comment
        def _seq_len(rec):
            if isinstance(rec, dict) and 'dates' in rec:
                return len(rec['dates'])
            if hasattr(rec, '__len__'):
                return len(rec)
            return 1
        idx = max(range(len(dataset.data)), key=lambda i: _seq_len(dataset.data[i]))
        print(f"--longest: using sample index {idx} ({_seq_len(dataset.data[idx])} real timepoints)")
    else:
        idx = args.sample_idx
    sample = dataset[idx]
    _sample_to_gifs(sample, results_dir, name=args.dataset, fps=args.fps)

    if args.deform_demo:
        semisynth_deform_gif(sample, results_dir, name=args.dataset,
                             structure=args.deform_structure, num_frames=args.deform_frames,
                             fps=args.fps, seed=args.deform_seed)
