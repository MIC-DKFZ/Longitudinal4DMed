import numpy
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch.nn as nn
from pathlib import Path


def plot_and_save_numpy_array(np_array, file_path=None):
    """
    Save a numpy array to a file. Helper function
    """
    fig, ax = plt.subplots()
    # Example content
    # Remove everything
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.imshow(np_array, cmap='gray')  # or any image
    # Save tightly
    if file_path is not None:
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()


def to_nump(input_tensor: torch.tensor, infer_shape=True) -> np.array:
    array = input_tensor.detach().cpu().numpy()
    return array


def make_prediction_grid_and_save(x_gt, x_con, x_pred, dataset_name, method_name, run_id=0,
                                  metric_functions={'MSE': nn.MSELoss()},
                                  time_string=None,
                                  base_dir="runs",
                                  show: bool = False):
    """
    x_gt, x_pred, residual, mask: 2D numpy arrays
    metrics: dict like {'PSNR':…, 'SSIM':…, 'DSC':…, 'HD':…}
    """

    run_path_abs = Path(base_dir.replace('/Logs', '')) / 'visual' / dataset_name / method_name
    run_path = run_path_abs / str(run_id)
    # run_path.mkdir(parents=True, exist_ok=True)
    context = x_con  # if we need that for additional processing:
    if isinstance(x_pred, torch.Tensor):
        x_pred = to_nump(x_pred)
    if isinstance(context, torch.Tensor):
        context = to_nump(context)
    if isinstance(x_gt, torch.Tensor):
        x_gt = to_nump(x_gt)
    residual_pred = np.abs(x_pred - x_gt)
    residual_lib = np.abs(context - x_gt)

    if x_gt.ndim == 3:
        # x_gt, context, x_pred are [S, H, W]
        diff = np.abs(context - x_gt).mean(axis=(1, 2))
        s = int(np.argmax(diff))
        x_gt = x_gt[s]
        x_pred = x_pred[s]
        residual_lib = residual_lib[s]
        residual_pred = residual_pred[s]
        context = context[s]

    # save raw images
    #plt.imsave(run_path / "gt.png", x_gt, cmap="gray")
    #plt.imsave(run_path / "context.png", context, cmap="gray")
    #plt.imsave(run_path / "pred.png", x_pred, cmap="gray")
    #plt.imsave(run_path / "residualLIB.png", residual_lib, cmap="gray")
    #plt.imsave(run_path / "residualPRD.png", residual_pred, cmap="gray")
    # if mask is not None:
    #    plt.imsave(run_path / "mask.png", mask, cmap="gray")

    # summary grid: 2 rows × 3 cols (residual above, image below)
    # format metrics strings
    # model_metrics, context_metrics = metrics
    placeholder = np.zeros_like(x_gt)
    ctx_psnr = peak_signal_noise_ratio(x_gt, context, data_range=x_gt.max() - x_gt.min())
    ctx_ssim = structural_similarity(x_gt, context, data_range=x_gt.max() - x_gt.min())
    prd_psnr = peak_signal_noise_ratio(x_gt, x_pred, data_range=x_gt.max() - x_gt.min())
    prd_ssim = structural_similarity(x_gt, x_pred, data_range=x_gt.max() - x_gt.min())

    # 2 rows × 3 cols: images over residuals
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    im = None
    cols = ["GT", "Context", "Prediction"]
    rel_scale = np.abs(residual_lib).max()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)
    for c, col in enumerate(cols):
        # bottom row: residuals
        ax = axs[1, c]
        if col == "GT":
            img = placeholder
        elif col == "Context":
            img = residual_lib / rel_scale
        else:  # Prediction
            img = residual_pred / rel_scale
        # Clip to [0, 2] and normalize for consistent scaling

        vmin, vmax = 0, 2
        im = ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax)
        ax.axis("off")
        if c == 0:
            ax.set_ylabel("Residual", fontsize=12)

        # top row: images
        ax = axs[0, c]
        img = (x_gt if col == "GT" else
               context if col == "Context" else
               x_pred)
        ax.imshow(img, cmap="gray", norm=norm)
        if col == "Context":
            ssim, psnr = ctx_ssim, ctx_psnr
        elif col == "Prediction":
            ssim, psnr = prd_ssim, prd_psnr
        else:
            ssim = psnr = None

        # build title
        if ssim is not None:
            title = f"{col}  SSIM: {ssim:.3f} | PSNR: {psnr:.1f}"
        else:
            title = col

        ax.set_title(title, fontsize=12)
        ax.axis("off")
        if c == 0:
            ax.set_ylabel("Image", fontsize=12)
    # (keep your original fig, axs = plt.subplots(2,3, figsize=(12,8)))
    fig.subplots_adjust(bottom=0.16)
    fig.tight_layout(rect=[0.04, 0.15, 0.96, 0.95])
    cax = fig.add_axes([0.12, 0.06, 0.76, 0.03])
    fig.colorbar(im, cax=cax, orientation='horizontal')

    if time_string is not None:
        summary_path = run_path_abs / time_string
    else:
        summary_path = run_path
    summary_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(summary_path / f"grid_{run_id:03d}.png", dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    plt.close(fig)


def visualize_predictions(context, target, pred, lci=None, title="Predictions", show=True):
    """Notebook-friendly 2x3 prediction grid (images + residuals).

    Parameters
    ----------
    context : tensor (B, T, C, D, H, W) or (T, C, D, H, W)
    target  : tensor (..., D, H, W)
    pred    : tensor (..., D, H, W)
    lci     : tensor (B, C, D, H, W) or (C, D, H, W).
              The last observed context frame. Compute with
              get_last_context_image_baseline() from validation_utils.

    Returns the matplotlib Figure.
    """
    def _central_slice(t):
        arr = t.squeeze().detach().cpu().float().numpy()
        while arr.ndim > 2:
            arr = arr[arr.shape[0] // 2]
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-8)

    if target.ndim >= 5:
        target = target[0]
    if pred.ndim >= 5:
        pred = pred[0]
    if lci is not None and lci.ndim >= 5:
        lci = lci[0]
    gt_s, lci_s, pred_s = _central_slice(target), _central_slice(lci), _central_slice(pred)

    res_lci  = np.abs(lci_s  - gt_s)
    res_pred = np.abs(pred_s - gt_s)
    rel_max  = max(res_lci.max(), res_pred.max(), 1e-8)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, img, ttl in zip(
        axes[0],
        [gt_s, lci_s, pred_s],
        ["Ground truth", "Last context", "Prediction"],
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")

    axes[1, 0].axis("off")
    im = None
    for ax, res, ttl in zip(
        axes[1, 1:],
        [res_lci / rel_max, res_pred / rel_max],
        ["|LCI - GT|", "|Pred - GT|"],
    ):
        im = ax.imshow(res, cmap="magma", vmin=0, vmax=1)
        ax.set_title(ttl, fontsize=11)
        ax.axis("off")

    cax = fig.add_axes([0.35, 0.03, 0.30, 0.02])
    fig.colorbar(im, cax=cax, orientation="horizontal", label="Normalised abs. error")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])

    if show:
        plt.show()
    return fig
