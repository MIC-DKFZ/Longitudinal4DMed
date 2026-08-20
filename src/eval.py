#!/usr/bin/env python
"""
eval.py — standalone, dataset-agnostic test-set evaluation.

Builds a per-sample metrics table (whole-image and, wherever a batch exposes any
target_seg / target_seg_<name> key, segmentation-masked as well), aggregates it
per checkpoint, and writes CSV / LaTeX tables plus prediction-grid visualizations.

Usage:
    python src/eval.py --ckpts checkpoints/acdc --n_last 3
    python src/eval.py --ckpts checkpoints/acdc/20260508_170731_tfm_best.pt \
                                checkpoints/acdc/20260509_000022_tfm_best.pt
    python src/eval.py --ckpts checkpoints/acdc --split test --n_samples 20 --no_viz

Each checkpoint must have been produced by train.py (a dict with model_state_dict
and args). Model + dataset are rebuilt straight from that saved args dict, so no
config file is needed at eval time.
"""
import argparse
import json
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loaders.data_utils import build_dataloader
from train import build_model
from utils.validation_utils import (
    metric_functions as _base_metric_functions,
    get_last_context_image_baseline,
    _extract_batch,
    _forward_and_reshape,
    _compute_metric_with_mask,
    _ensure_min_spatial_for_lpips,
    load_torcheval,
)
from utils.plotting import make_prediction_grid_and_save

if load_torcheval:
    from utils.validation_utils import update_fid_metric
    from torcheval.metrics import FrechetInceptionDistance


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _laplacian_edge_nrmse(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """NRMSE on 3-D Laplacian-filtered volumes: quantifies edge/structure
    sharpness preservation, independent of the plain intensity NRMSE.
    Also appears to help in training for structure. 
    TODO: add this optionally as a loss
    """
    k = pred.new_zeros(1, 1, 3, 3, 3)
    k[0, 0, 1, 1, 1] = -6
    for idx in [(0, 1, 1), (2, 1, 1), (1, 0, 1), (1, 2, 1), (1, 1, 0), (1, 1, 2)]:
        k[0, 0, idx[0], idx[1], idx[2]] = 1
    C = pred.shape[1]
    k = k.expand(C, 1, 3, 3, 3)
    lap_pred = F.conv3d(pred.float(), k, padding=1, groups=C)
    lap_gt = F.conv3d(gt.float(), k, padding=1, groups=C)
    return torch.sqrt(F.mse_loss(lap_pred, lap_gt))


def _build_metric_functions(device):
    fns = dict(_base_metric_functions)
    fns['edge_nrmse'] = _laplacian_edge_nrmse
    for name in fns:
        try:
            fns[name] = fns[name].to(device)
        except Exception:
            pass
    return fns


_BASE_METRICS = ['mse', 'nrmse', 'ssim', 'psnr', 'edge_nrmse', 'perc', 'corr', 'auroc']

_CORR_EPS = 0.02       # |Δgt| threshold below which voxels are excluded from the correlation
_CORR_SUBSAMPLE = 5000  # cap on voxels fed to spearmanr, for speed
_AUROC_K = 5.0          # top-K% of Δgt increases/decreases define the "changed" AUROC labels


def _correlation_distance(pred, gt, lci, eps=_CORR_EPS, n_subsample=_CORR_SUBSAMPLE):
    """Offset-only Spearman distance: (1 - Spearman(Δpred, Δgt)) / 2, computed over
    voxels where |Δgt| > eps (Δ = value - lci baseline). 0 = perfect, 0.5 = no
    correlation (also what a do-nothing pred=lci baseline scores), 1 = anti-correlated.
    Note that we could also use this in principle without the lci delta. 
    Also, this loss is quite compute intense, so it is adviced to only use it sparingly or during inference. 

    # TODO: ad a warning that if this metric is used it may be compute intense. 
    # NOTE: Compute intense, but neat as additional metric. 
    """
    from scipy.stats import spearmanr
    d_gt = (gt - lci).flatten().cpu().numpy()
    d_pred = (pred - lci).flatten().cpu().numpy()
    sel = np.abs(d_gt) > eps
    if sel.sum() < 10:
        return 0.5
    a, b = d_pred[sel], d_gt[sel]
    if len(a) > n_subsample:
        idx = np.random.RandomState(42).choice(len(a), size=n_subsample, replace=False)
        a, b = a[idx], b[idx]
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.5
    rho, _ = spearmanr(a, b)
    return (1.0 - (0.0 if np.isnan(rho) else float(rho))) / 2.0


def _auroc_distance(pred, gt, lci, k=_AUROC_K):
    """Direction-aware double-AUROC distance on Δ = value - lci: does the predicted
    change rank-order the top-K% largest ground-truth increases/decreases correctly?
    0 = perfect, 0.5 = no signal (also the do-nothing pred=lci baseline score),
    1 = predicts the wrong direction. Standard-library reimplementation
    (sklearn.metrics.roc_auc_score) of the same recipe used internally.
    """
    from sklearn.metrics import roc_auc_score
    d_gt = (gt - lci).flatten().cpu().numpy()
    d_pred = (pred - lci).flatten().cpu().numpy()
    pos_vals = d_gt[d_gt > 0]
    neg_vals = -d_gt[d_gt < 0]
    tau_pos = float(np.percentile(pos_vals, 100.0 - k)) if len(pos_vals) else np.inf
    tau_neg = float(np.percentile(neg_vals, 100.0 - k)) if len(neg_vals) else np.inf
    label_pos = (d_gt >= tau_pos).astype(np.int32)
    label_neg = (d_gt <= -tau_neg).astype(np.int32)
    aurocs = []
    n = len(label_pos)
    if 0 < label_pos.sum() < n:
        aurocs.append(roc_auc_score(label_pos, d_pred))
    if 0 < label_neg.sum() < n:
        aurocs.append(roc_auc_score(label_neg, -d_pred))
    return 1.0 - float(np.mean(aurocs)) if aurocs else 0.5


def _is_metric_col(col: str) -> bool:
    """True for a per-sample result column that's a metric to aggregate — either
    a bare whole-image metric ('ssim') or a seg-masked one ('seg_ssim', or
    'seg_<name>_ssim' for a dataset-defined named structure). Deliberately
    name-agnostic: eval.py has no built-in notion of what structures exist for
    any given dataset, it just reports whatever a batch happens to expose."""
    if col in _BASE_METRICS:
        return True
    if col.startswith('seg_'):
        rest = col[len('seg_'):]
        # Check longest metric names first -> so no collisions. 
        return rest in _BASE_METRICS or any(
            rest.endswith(f'_{m}') for m in sorted(_BASE_METRICS, key=len, reverse=True)
        )
    return False


def _compute_sample_metrics(pred, gt, metric_functions, mask=None, lci=None):
    """pred, gt: (1, C, *spatial) tensors. mask: same shape bool tensor, or None
    for whole-image. lci: same-shape baseline tensor, required for corr/auroc
    (skipped with NaN if not provided). Returns a flat {metric_name: float} dict."""
    out = {}
    for name, fn in metric_functions.items():
        try:
            val = _compute_metric_with_mask(
                fn, pred, gt, mask if mask is not None else torch.ones_like(pred, dtype=torch.bool),
                apply_mask=mask is not None, metric_function_name=name,
            )
            out[name] = float(val.item() if torch.is_tensor(val) else val)
        except Exception:
            out[name] = float('nan')
        if name == 'mse':
            out['nrmse'] = math.sqrt(out['mse']) if out['mse'] == out['mse'] else float('nan')

    if lci is not None:
        p, g, l = (pred, gt, lci) if mask is None else (pred[mask], gt[mask], lci[mask])
        try:
            out['corr'] = _correlation_distance(p, g, l)
        except Exception:
            out['corr'] = float('nan')
        try:
            out['auroc'] = _auroc_distance(p, g, l)
        except Exception:
            out['auroc'] = float('nan')
    else:
        out['corr'] = float('nan')
        out['auroc'] = float('nan')
    return out


def _build_seg_mask(seg_raw, pred_shape, device):
    """seg_raw: whatever a batch returns under a target_seg* key (or None).
    Returns a boolean mask of shape pred_shape, or None if unavailable/empty."""
    if seg_raw is None:
        return None
    sr = seg_raw.to(device).float()
    if sr.ndim == len(pred_shape) - 1:
        sr = sr.unsqueeze(1)
    if sr.ndim != len(pred_shape) and sr.numel() == int(np.prod(pred_shape)):
        sr = sr.reshape(pred_shape)
    if tuple(sr.shape) != tuple(pred_shape):
        return None
    mask = sr > 0.5
    return mask if bool(mask.any()) else None


# ---------------------------------------------------------------------------
# Checkpoint discovery / loading
# ---------------------------------------------------------------------------

def _resolve_ckpts(ckpts, n_last, dataset_filter=None):
    """If a single directory is given: glob *.pt sorted by mtime, optionally
    filter by dataset name substring in the filename, take the last n_last
    (0 = all). Otherwise return the explicit file list unchanged."""
    if len(ckpts) == 1 and Path(ckpts[0]).is_dir():
        folder = Path(ckpts[0])
        all_pts = sorted(folder.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not all_pts:
            raise FileNotFoundError(f"No .pt files found in {folder}")
        if dataset_filter:
            all_pts = [p for p in all_pts if dataset_filter.lower() in p.stem.lower()]
            if not all_pts:
                raise FileNotFoundError(
                    f"No checkpoints with '{dataset_filter}' in filename found in {folder}"
                )
        pts = all_pts[-n_last:] if n_last > 0 else all_pts
        print(f"[ckpt folder] using {len(pts)} checkpoint(s) from {folder}:")
        for p in pts:
            print(f"  {p.name}")
        return [str(p) for p in pts]
    return ckpts


def _load_ckpt_args(ckpt_paths):
    """Cheap metadata pass: read the saved args dict for every checkpoint
    without holding onto the (much larger) state dicts yet."""
    all_args = []
    for path in ckpt_paths:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
        all_args.append(ckpt['args'])
    return all_args


def _make_labels(ckpt_paths, all_args):
    """Derive a human-readable label per checkpoint from whichever scalar
    hparams actually vary across the provided set, falling back to model_type;
    de-duplicated so repeats get a numeric suffix."""
    all_keys = sorted({
        k for a in all_args for k, v in a.items()
        if isinstance(v, (int, float, str, bool))
    })
    varying = [
        k for k in all_keys
        if len({str(a.get(k, '__missing__')) for a in all_args}) > 1
        and k not in ('save_dir', 'log_dir', 'in_shape', 'model_type')
    ]
    raw = []
    for a in all_args:
        base = str(a.get('model_type', 'model'))
        if varying:
            base += '_' + '_'.join(f"{k}={a.get(k, '?')}" for k in varying)
        raw.append(base)
    seen, labels = {}, []
    for lbl in raw:
        seen[lbl] = seen.get(lbl, 0) + 1
        labels.append(lbl if seen[lbl] == 1 else f"{lbl}_{seen[lbl]}")
    return labels, varying


def _load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args_ns = argparse.Namespace(**ckpt['args'])
    model = build_model(args_ns, device)
    incompatible = model.load_state_dict(ckpt['model_state_dict'], strict=False)
    if incompatible.missing_keys:
        print(f"    missing keys : {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(f"    ignored keys : {incompatible.unexpected_keys}")
    model.eval().to(device)
    return model, ckpt['args']


# ---------------------------------------------------------------------------
# Aggregate table / LaTeX export
# ---------------------------------------------------------------------------

_TABLE_COLS = [
    ('nrmse_median', 'NRMSE', '.4f'),
    ('ssim_median', 'SSIM', '.4f'),
    ('psnr_median', 'PSNR', '.2f'),
    ('edge_nrmse_median', 'Edge-NRMSE', '.4f'),
    ('perc_median', 'Perc', '.4f'),
    ('corr_median', 'Corr', '.3f'),
    ('auroc_median', 'AUROC', '.3f'),
]
_HIGHER_IS_BETTER = {'SSIM', 'PSNR'}
# Corr/AUROC here are distances (0=perfect, 0.5=do-nothing, 1=worst) — lower is better,
_LOWER_IS_BETTER = {'NRMSE', 'Edge-NRMSE', 'Perc', 'Corr', 'AUROC'}


def _seg_table_cols(prefix):
    return [(f"{prefix}{c}", lbl, f) for c, lbl, f in _TABLE_COLS]


def _discover_seg_structures(agg_df):
    """Find dataset-defined named seg structures from agg_df's own columns, e.g.
    'seg_csf_ssim_median' -> 'csf'. eval.py never hardcodes a structure name."""
    names = set()
    for col in agg_df.columns:
        if not (col.startswith('seg_') and col.endswith('_median')):
            continue
        rest = col[len('seg_'):-len('_median')]
        if rest in _BASE_METRICS:
            continue
        for m in sorted(_BASE_METRICS, key=len, reverse=True):
            if rest.endswith(f'_{m}'):
                names.add(rest[:-(len(m) + 1)])
                break
    return sorted(names)


def _save_latex(display, cols, out_dir, file_stem, caption):
    def _rank(series, higher):
        valid = series.dropna()
        if valid.empty:
            return {}
        return valid.rank(ascending=not higher, method='min').to_dict()

    ranks = {}
    for col in display.columns:
        if col in _HIGHER_IS_BETTER:
            ranks[col] = _rank(display[col], higher=True)
        elif col in _LOWER_IS_BETTER:
            ranks[col] = _rank(display[col], higher=False)

    def _fmt_cell(col, row_label, val):
        _, _, f = next((x for x in cols if x[1] == col), ('', col, '.4f'))
        if not isinstance(val, float) or np.isnan(val):
            return '---'
        s = f"{val:{f}}"
        r = ranks.get(col, {}).get(row_label)
        if r == 1:
            return r"\textbf{" + s + "}"
        if r == 2:
            return r"\underline{" + s + "}"
        return s

    col_spec = 'l' + 'r' * len(display.columns)
    header = ' & '.join(['Model'] + list(display.columns)) + r' \\'
    lines = [r'\begin{table}[t]', r'\centering', r'\begin{tabular}{' + col_spec + '}',
             r'\toprule', header, r'\midrule']
    for row_label, row in display.iterrows():
        cells = [str(row_label)] + [_fmt_cell(c, row_label, row[c]) for c in display.columns]
        lines.append(' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{' + caption + r'. \textbf{Bold}: best, \underline{underline}: second best.}',
              r'\label{tab:' + file_stem + '}', r'\end{table}']
    (out_dir / f"{file_stem}.tex").write_text('\n'.join(lines) + '\n')
    print(f"LaTeX table saved to {out_dir / f'{file_stem}.tex'}")


def _print_one_table(agg_df, out_dir, cols, title, file_stem):
    cols = [(c, lbl, f) for c, lbl, f in cols if c in agg_df.columns]
    if not cols:
        return
    display = agg_df[[c for c, _, _ in cols]].copy()
    display.columns = [lbl for _, lbl, _ in cols]
    display.index.name = 'Model'
    pd.set_option('display.float_format', '{:.4f}'.format)
    pd.set_option('display.width', None)
    print(f"\n=== {title} ===")
    print(display.to_string())
    display.to_csv(out_dir / f"{file_stem}.csv")
    _save_latex(display, cols, out_dir, file_stem, title)


def _print_aggregate(agg_df, out_dir):
    _print_one_table(agg_df, out_dir, _TABLE_COLS, 'Aggregate Metrics — Whole Image (median)',
                     'metrics_table_whole')
    _print_one_table(agg_df, out_dir, _seg_table_cols('seg_'),
                     'Aggregate Metrics — Segmentation-Masked (median)', 'metrics_table_seg')
    for name in _discover_seg_structures(agg_df):
        _print_one_table(agg_df, out_dir, _seg_table_cols(f'seg_{name}_'),
                         f"Aggregate Metrics — '{name}'-Masked (median)", f"metrics_table_seg_{name}")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def _eval_dataset_for(args_dict, split_str):
    """Build the (deterministic, batch_size=1) eval dataset from one
    checkpoint's saved args dict."""
    args_ns = argparse.Namespace(**args_dict)
    loader = build_dataloader(args_ns, train_test_val=split_str)
    return loader.dataset


def run_eval(ckpt_paths, args, device):
    all_args = _load_ckpt_args(ckpt_paths)
    datasets = {str(a.get('dataset')) for a in all_args}
    assert len(datasets) == 1, f"All checkpoints must share the same --dataset, got: {datasets}"
    dataset_name = all_args[0].get('dataset', 'unknown')

    labels, varying_keys = _make_labels(ckpt_paths, all_args)

    out_dir = Path(args.out_dir) if args.out_dir else Path("results") / "eval" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    split_str = 'tst' if args.split == 'test' else 'val'
    print("Loading dataset ...")
    eval_dataset = _eval_dataset_for(all_args[0], split_str)
    loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    print("Loading models ...")
    models = []
    for path, label in zip(ckpt_paths, labels):
        print(f"  {Path(path).name}  [{label}]")
        model, _ = _load_model(path, device)
        models.append(model)

    metric_functions = _build_metric_functions(device)
    fid_metrics = None
    if load_torcheval:
        fid_metrics = {lbl: FrechetInceptionDistance().to(device) for lbl in labels}
        fid_metrics['lci'] = FrechetInceptionDistance().to(device)

    n_total = min(len(loader), args.n_samples) if args.n_samples else len(loader)
    results = []
    for sample_idx, batch in enumerate(tqdm(loader, total=n_total, desc="Evaluating")):
        if args.n_samples and sample_idx >= args.n_samples:
            break

        batch_x, batch_y, _, _, time_points = _extract_batch(batch, device)
        lci = get_last_context_image_baseline(batch_x)
        pred_shape = lci.shape
        gt = batch_y.view(pred_shape)

        seg_masks = {}
        generic_mask = _build_seg_mask(batch.get('target_seg'), pred_shape, device)
        if generic_mask is not None:
            seg_masks[''] = generic_mask
        for key in batch.keys():
            if key == 'target_seg' or not key.startswith('target_seg_'):
                continue
            named_mask = _build_seg_mask(batch.get(key), pred_shape, device)
            if named_mask is not None:
                seg_masks[key[len('target_seg_'):]] = named_mask

        # LCI baseline row
        lci_metrics = _compute_sample_metrics(lci, gt, metric_functions, lci=lci)
        for name, mask in seg_masks.items():
            prefix = f"seg_{name}_" if name else "seg_"
            lci_metrics.update({
                f"{prefix}{k}": v for k, v in _compute_sample_metrics(lci, gt, metric_functions, mask, lci=lci).items()
            })
        results.append({'sample_idx': sample_idx, 'model_label': 'lci', **lci_metrics})
        if fid_metrics is not None:
            update_fid_metric(gt, lci, fid_metrics['lci'])

        gt_np = gt[0, 0].cpu().numpy()
        lci_np = lci[0, 0].cpu().numpy()

        for model, label in zip(models, labels):
            with torch.no_grad():
                pred, _ = _forward_and_reshape(model, batch_x, batch_y, time_points, in_shape=pred_shape)
            m = _compute_sample_metrics(pred, gt, metric_functions, lci=lci)
            for name, mask in seg_masks.items():
                prefix = f"seg_{name}_" if name else "seg_"
                m.update({
                    f"{prefix}{k}": v for k, v in _compute_sample_metrics(pred, gt, metric_functions, mask, lci=lci).items()
                })
            results.append({'sample_idx': sample_idx, 'model_label': label, **m})
            if fid_metrics is not None:
                update_fid_metric(gt, pred, fid_metrics[label])

            if not args.no_viz:
                pred_np = pred[0, 0].cpu().numpy()
                make_prediction_grid_and_save(
                    gt_np, lci_np, pred_np,
                    dataset_name=dataset_name, method_name=label,
                    run_id=sample_idx, base_dir=str(out_dir / 'visualizations'),
                )

    df = pd.DataFrame(results)
    df.to_csv(out_dir / 'per_sample_metrics.csv', index=False)

    print("\nModel legend (config differences):")
    if varying_keys:
        for lbl, a in zip(labels, all_args):
            diffs = '  |  '.join(f"{k}={a.get(k, '?')}" for k in varying_keys)
            print(f"  {lbl}: {diffs}")
    else:
        print("  (all checkpoints share identical scalar hparams)")

    agg_rows = []
    for label in df['model_label'].unique():
        sub = df[df['model_label'] == label]
        row = {'model_label': label}
        for col in sub.columns:
            if not _is_metric_col(col):
                continue
            vals = sub[col].dropna()
            if len(vals) == 0:
                continue
            row[f"{col}_median"] = float(vals.median())
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows).set_index('model_label')
    agg_df.to_csv(out_dir / 'aggregate_metrics.csv')
    _print_aggregate(agg_df, out_dir)

    if fid_metrics is not None:
        print("\n=== FID (whole eval set) ===")
        fid_rows = {}
        for lbl, m in fid_metrics.items():
            try:
                fid_rows[lbl] = float(m.compute())
            except Exception:
                fid_rows[lbl] = float('nan')
            print(f"  {lbl}: {fid_rows[lbl]:.4f}")
        pd.Series(fid_rows, name='fid').to_csv(out_dir / 'fid.csv')

    print(f"\nDone. Results written to {out_dir}/")
    return df, agg_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more checkpoints on a dataset split.")
    parser.add_argument('--ckpts', nargs='+', required=True,
                        help=".pt checkpoint files, or a single directory to search")
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--out_dir', default=None,
                        help="Output directory (default: results/eval/<dataset>)")
    parser.add_argument('--n_last', type=int, default=0,
                        help="When --ckpts is a directory: use only the last N checkpoints by mtime. 0 = all.")
    parser.add_argument('--n_samples', type=int, default=None,
                        help="Limit number of evaluated samples (default: all)")
    parser.add_argument('--dataset_filter', default='',
                        help="Only use checkpoints whose filename contains this string (e.g. 'acdc')")
    parser.add_argument('--no_viz', action='store_true',
                        help="Skip writing prediction-grid visualizations (faster)")
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == 'cpu' or torch.cuda.is_available()) else 'cpu')
    ckpt_paths = _resolve_ckpts(args.ckpts, args.n_last, dataset_filter=args.dataset_filter)
    run_eval(ckpt_paths, args, device)


if __name__ == "__main__":
    main()
