import torch
import os
import os
import torch.nn as nn
import torch
import torchmetrics
from torchmetrics.functional import structural_similarity_index_measure as tm_ssim

DATA_DIR = os.getenv('DATA_DIR', './data')
RESULT_DIR = os.getenv('RESULT_DIR', './results')
LOG_DIR = os.getenv('LOG_DIR', './results/Logs')


# todo: make that nicer later
metric_functions = {
    'mse': nn.MSELoss(reduction='none'),
    'ssim': tm_ssim,
    'psnr': torchmetrics.image.PeakSignalNoiseRatio(data_range=1.0),
}
#  since both may be a little finicky, we just have a fallback option
try:
    from generative.losses import PerceptualLoss
    perc_loss_fn = PerceptualLoss(spatial_dims=3,
                                  network_type="squeeze",
                                  is_fake_3d=True,
                                  fake_3d_ratio=0.2)
    metric_functions['perc'] = perc_loss_fn

except:
    print('could not load perceptual loss')

try:
    from torcheval.metrics import FrechetInceptionDistance
    load_torcheval = True
    fid_metric = FrechetInceptionDistance().to('cuda')
    fid_metric_last_image = FrechetInceptionDistance().to('cuda')
except:
    print('could not load torcheval')
    load_torcheval = False


def get_last_context_image_baseline(batch_x_val: torch.Tensor) -> torch.Tensor:
    # batch_x_val: [B, T, C, W, H, D]
    B, T = batch_x_val.shape[:2]
    flat = batch_x_val.view(B, T, -1)  # [B, T, *]
    nonzero = flat.ne(0).any(dim=-1)  # [B, T] bool

    # indices 0..T-1 for each batch
    idx_grid = torch.arange(T, device=batch_x_val.device).view(1, T).expand(B, T)
    # set index to 0 where all-zero, keep true index where non-zero
    idx = torch.where(nonzero, idx_grid, torch.zeros_like(idx_grid))
    last_idx = idx.max(dim=1).values  # [B]

    # gather last non-zero frame per batch
    return batch_x_val[torch.arange(B, device=batch_x_val.device), last_idx]  # [B, C, W, H, D]



def _ensure_min_spatial_for_lpips(x: torch.Tensor, min_spatial: int = 32) -> torch.Tensor:
    """
    Ensure that the spatial dimensions of x are at least `min_spatial`,
    using interpolation. This is meant to avoid LPIPS / perceptual 2D
    backbones failing on tiny spatial sizes.

    Accepts:
        x: 4D (B, C, H, W) or 5D (B, C, D, H, W) tensor.
        min_spatial: minimum size for each spatial dimension.

    Returns:
        Tensor with same rank as x, possibly upsampled in spatial dims.
    """
    import torch.nn.functional as F

    if x is None:
        return x

    if not torch.is_tensor(x):
        raise TypeError(f"_ensure_min_spatial_for_lpips expected Tensor, got {type(x)}")

    ndim = x.dim()
    if ndim < 4:
        # nothing to do (not an image / volume)
        return x

    # 2D image: (B, C, H, W)
    if ndim == 4:
        _, _, h, w = x.shape
        if h >= min_spatial and w >= min_spatial:
            return x

        new_h = max(h, min_spatial)
        new_w = max(w, min_spatial)
        return F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    # 3D volume: (B, C, D, H, W)
    if ndim == 5:
        _, _, d, h, w = x.shape
        if d >= min_spatial and h >= min_spatial and w >= min_spatial:
            return x

        new_d = max(d, min_spatial)
        new_h = max(h, min_spatial)
        new_w = max(w, min_spatial)
        return F.interpolate(x, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)

    # higher ranks: keep as is (LPIPS shouldn't see these directly)
    return x


def temporal_change_mask(context_input, target_input, threshold = 0.015):
    shape = target_input.shape
    # we assume a shape of B,T,C,W,H,D
    time_points = context_input.shape[1]
    mask = torch.zeros(shape, dtype=torch.bool, device=target_input.device)
    for batch_idx in range(shape[0]):
        for time in range(time_points):
            # sometimes, due to pre-processing, the context windows are zero, but those are sadly independent of batch
            # check for that, otherwise it horrendously fails
            if context_input[batch_idx, time].sum().item() != 0:
                # naively check the change, can also be a different transformation
                change_at_time = torch.abs(context_input[batch_idx, time] - target_input[batch_idx]) > threshold
                mask[batch_idx] = torch.logical_or(mask[batch_idx], change_at_time.squeeze())

    return mask


def _extract_batch(batch_val, device):
    batch_y = batch_val['target_img'].to(device)
    batch_x = batch_val['context'].to(device)
    batch_y_seg = batch_val['target_seg'].to(device)
    batch_x_seg = batch_val['context_seg'].to(device)
    target_time = batch_val['target_time'].to(device)
    context_time = batch_val['context_time'].to(device)
    time_points = None
    if target_time is not None and context_time is not None:
        time_points = torch.concat([context_time, target_time], dim=1)

    return batch_x, batch_y, batch_x_seg, batch_y_seg, time_points


def _apply_temporal_masking_to_input(batch_x_val, mask_ratio, mask_order):
    '''
    F
    :param batch_x_val:
    :param mask_ratio:
    :param mask_order:
    :return:
    '''
    if mask_ratio is None:
        return batch_x_val
    try:
        if mask_order == 'front':
            batch_x_val[:, 0:int(mask_ratio)] = 0.0
        elif mask_order == 'back':
            batch_x_val[:, -int(mask_ratio):] = 0.0
    except Exception:
        print('masking failed, maybe not enough time points?')
    return batch_x_val


def _forward_and_reshape(train_model, batch_x_val, batch_y_val, time_points, **kwargs):
    '''
    :param train_model:
    :param batch_x_val:
    :param batch_y_val:
    :param time_points:
    :param in_shape:
    :return: the prediction of the model, handles different outputs
    '''

    in_shape = kwargs['in_shape']
    # in_shape = batch_y_val.shape
    model_output = train_model.validation_step(batch_x_val, batch_y_val, time_points=time_points)

    if isinstance(model_output, (list, tuple)):
        batch_y_val = model_output[1]
        model_output = model_output[0].unsqueeze(1)

    pred_shape = (batch_y_val.size(0), *in_shape[1:])
    model_output = model_output.view(pred_shape)
    batch_y_val = batch_y_val.view(pred_shape)

    return model_output, batch_y_val


def _compute_change_mask(batch_x_val, batch_y_val, batch_x_seg, batch_y_seg, temporal_masking=False, use_seg=False, device='cuda', **kwargs):
    if not temporal_masking:
        return torch.ones(batch_y_val.shape, dtype=torch.bool, device=device)

    if use_seg:
        if batch_x_seg is not None and batch_y_seg is not None:
            seg_mask = torch.logical_or(batch_x_seg, batch_y_seg)
        elif batch_x_seg is not None:
            seg_mask = batch_x_seg
        elif batch_y_seg is not None:
            seg_mask = batch_y_seg
        else:
            seg_mask = None
        if seg_mask is not None:
            return seg_mask.unsqueeze(1).to(device)

    return temporal_change_mask(batch_x_val, batch_y_val)


def _compute_metric_with_mask(metric_function, model_output, batch_y_val, change_mask, apply_mask = False, metric_function_name = ''):
    ## refactor this, as we assume the change mask to be zero?

    if apply_mask:
        model_output = model_output * change_mask
        batch_y_val = batch_y_val*change_mask
        # metric_tensor = metric_function(model_output*change_mask, batch_y_val*change_mask)

    if metric_function_name == 'perc':
        # for lpips, ensure minimum spatial size, as some dims may be too low.
        model_output = _ensure_min_spatial_for_lpips(model_output)
        batch_y_val = _ensure_min_spatial_for_lpips(batch_y_val)
        metric_tensor = metric_function(model_output, batch_y_val)
        return metric_tensor.mean()

    metric_tensor = metric_function(model_output, batch_y_val)
    return metric_tensor.mean()



def update_fid_metric(val_y, val_x, fid_metric):

    # B, C, W, H, D = val_x.shape
    # global W
    max_val = val_x.abs().max()
    norm_output = val_x / (max_val + 1e-8)
    norm_output = 0.5 * (norm_output + 1.0)  # shift from [-1,1] to [0,1]

    # Do the same for batch_y_val if needed
    max_val_y = val_y.abs().max()
    norm_gt = val_y / (max_val_y + 1e-8)
    # norm_gt = 0.5 * (norm_gt + 1.0)

    # Now reshape for FID
    if norm_output.dim() == 5:
        B, C, D, H, W = norm_output.shape
    elif norm_output.dim() ==4:
        B,D,H,W = norm_output.shape
    elif norm_output.dim() ==6:
        B,T,C,D,H,W = norm_output.shape
    else:
        print('error in prediction shape')
    norm_output = norm_output.reshape(B, C, D, H, W)
    norm_gt = norm_gt.reshape(B,C,D,H,W)
    gen_2d = norm_output.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
    real_2d = norm_gt.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)

    # If single-channel, repeat to 3 channels
    if C == 1:
        gen_2d = gen_2d.repeat(1, 3, 1, 1)
        real_2d = real_2d.repeat(1, 3, 1, 1)

    # Update the FID metric
    fid_metric.update(real_2d, True)
    fid_metric.update(gen_2d, False)
    return


def _update_metrics_dicts(metric_functions, model_output, batch_y_val, change_mask,
                          result_metrics_dict, plotting=False, metric_list=None, str_prefix=''):
    for metric_name, metric_function in metric_functions.items():
        # todo: add the option for a change mask and rename it
        # todo: rename the change mask, as it uses either segmentation mask, or a difference mask?
        metric_value = _compute_metric_with_mask(metric_function, model_output, batch_y_val, change_mask, metric_function_name = metric_name)

        if metric_value is None:
            continue

        if metric_name == 'mse':
            result_metrics_dict[str_prefix + 'nrmse'] += torch.sqrt(metric_value).item()

        val_item = metric_value.item()
        # we use the str_prefix so that we can differentiate between different metric names, e.g. for the min metrics
        result_metrics_dict[str_prefix + metric_name] += val_item

        if plotting and metric_list is not None:
            metric_list[metric_name].append(val_item)





def val_step(valid_loader, train_model, device, metric_functions = metric_functions,
             use_last_image_as_baseline: bool = True, temporal_masking: bool = False, min_val=10, logger=None, use_seg=False,
             save_path=None,
             log_images=False, time_string=None, return_metrics=True, mask_ratio=None, mask_order='front',
             **kwargs):
    # val_loss = torch.inf
    do_plotting = True
    # do not do this every time?
    for name, metric in metric_functions.items():
        try:
            metric_functions[name] = metric.to(device)
        except:
            pass

    with torch.no_grad():
        total_metrics = {name: 0.0 for name, _ in metric_functions.items()}
        total_metrics['nrmse'] = 0.0
        total_last_metrics = {f'last_{name}': 0.0 for name, _ in total_metrics.items()}

        # if we want to plot the distribution over the batches
        if do_plotting:
            metric_list = {name: [] for name, _ in metric_functions.items()}
        else:
            metric_list = None

        for batch_idx_val, batch_val in enumerate(valid_loader):
            batch_x_val, batch_y_val, batch_x_seg, batch_y_seg, time_points = _extract_batch(batch_val, device)
            batch_x_val = _apply_temporal_masking_to_input(batch_x_val, mask_ratio, mask_order)
            last_context_image = get_last_context_image_baseline(batch_x_val)
            B, T, C, W, H, D = batch_x_val.shape

            # do some reshaping which may not happen in the model
            model_output, batch_y_val = _forward_and_reshape(
                train_model,
                batch_x_val,
                batch_y_val,
                time_points,
                **kwargs
            )

            if load_torcheval:
                update_fid_metric(batch_y_val, model_output, fid_metric)
                update_fid_metric(batch_y_val, last_context_image, fid_metric_last_image)

            # todo: two different change masks: once regular, and on
            no_change_mask = _compute_change_mask(
                batch_x_val,
                batch_y_val,
                batch_x_seg,
                batch_y_seg,
                temporal_masking=False,
                use_seg=False,
                device=batch_y_val.device,
            )

            _update_metrics_dicts(
                metric_functions,
                model_output,
                batch_y_val,
                no_change_mask,
                total_metrics,
                # plotting=do_plotting, todo:change to different name
                metric_list=metric_list,
            )

            if use_last_image_as_baseline:
                _update_metrics_dicts(
                    metric_functions,
                    last_context_image,
                    batch_y_val,
                    no_change_mask,
                    total_last_metrics,
                    str_prefix='last_',
                    # plotting=do_plotting, # todo: see above
                    metric_list=metric_list,
                )
                if do_plotting:
                    from utils.plotting import make_prediction_grid_and_save
                    for batch in range(B):
                        for channel in range(C):
                            make_prediction_grid_and_save(batch_y_val[batch, channel],
                                                          last_context_image[batch, channel],
                                                          model_output[batch, channel],
                                                          'test',
                                                          'test_method')
        avg_metrics = {key: value / len(valid_loader) for key, value in total_metrics.items()}

        if load_torcheval:
            fid_value = fid_metric.compute()
            avg_metrics['fid'] = fid_value.item() if torch.is_tensor(fid_value) else fid_value
            fid_metric.reset()
            if use_last_image_as_baseline:
                fid_value_min = fid_metric_last_image.compute()
                avg_metrics['last_fid'] = fid_value_min.item() if torch.is_tensor(fid_value) else fid_value_min
                fid_metric_last_image.reset()

        if use_last_image_as_baseline:
            avg_last_metrics = {key: float(value) / len(valid_loader) for key, value in total_last_metrics.items()}
            avg_metrics.update(avg_last_metrics)

        print("Validation Metrics:")
        for metric_name, value in avg_metrics.items():
            print(f"{metric_name}: {value}")

        if 'mse' in avg_metrics:
            val_loss = avg_metrics['mse']
        elif 'nrmse' in avg_metrics:
            val_loss = avg_metrics['nrmse']
        else:
            val_loss = min_val

    return (avg_metrics, val_loss) if return_metrics else val_loss
