import torch
import torch.nn.functional as F


def process_fill_empty(batch_x, batch_y=None, time_points=None, max_images=16, **kwargs):
    """
    Preprocess batch for flow models.

    batch_x:     (B, T, C, H, W, D)
    time_points: (B, T') or (T',) or None
                 If T' != T, we either drop or resample to match T.

    Returns:
        processed_images: (B, max_images, C, H, W, D)
        processed_times:  (B, max_images)
    """
    device = batch_x.device
    B, T, C, H, W, D = batch_x.shape

    # handle time_points
    if time_points is None:
        # uniform times in [0,1] for T frames
        time_points = torch.linspace(0, 1, T, device=device).expand(B, T)
    else:
        if isinstance(time_points, list):
            time_points = torch.stack([tp.to(device) for tp in time_points], dim=0)
        else:
            time_points = time_points.to(device)
            if time_points.dim() == 1:
                # (T') -> (B, T')
                time_points = time_points.unsqueeze(0).expand(B, -1)

        # now time_points: (B, T')
        T_tp = time_points.size(1)
        if T_tp == T:
            pass
        elif T_tp == T + 1:
            # typical case: context+target; drop last (target) for context preprocessing
            time_points = time_points[:, :T]
        else:
            # general fallback: resample to length T
            idx = torch.linspace(0, T_tp - 1, steps=T, device=device).round().long()
            time_points = time_points[:, idx]

    # non-zero mask over channels and spatial dims
    train_mask = batch_x.sum(dim=(2, 3, 4, 5)) != 0   # (B, T)

    processed_images = []
    processed_times = []

    for b in range(B):
        mask_b = train_mask[b]                # (T,)
        imgs_b = batch_x[b, mask_b]          # (#valid, C, H, W, D)
        times_b = time_points[b, mask_b]     # (#valid,)

        # if no valid frames, fall back to first frame
        if imgs_b.numel() == 0:
            imgs_b = batch_x[b, :1]          # (1, C, H, W, D)
            times_b = time_points[b, :1]     # (1,)

        n = imgs_b.size(0)

        if n > max_images:
            # subsample evenly to max_images
            idx = torch.linspace(0, n - 1, steps=max_images, device=device).long()
            imgs_b = imgs_b[idx]
            times_b = times_b[idx]
        elif n < max_images:
            pad = max_images - n
            first_img = imgs_b[0:1].expand(pad, -1, -1, -1, -1)
            first_time = times_b[0].expand(pad)
            imgs_b = torch.cat([first_img, imgs_b], dim=0)
            times_b = torch.cat([first_time, times_b], dim=0)

        # now imgs_b: (max_images, C, H, W, D)
        #     times_b: (max_images,)
        processed_images.append(imgs_b)
        processed_times.append(times_b)

    processed_images = torch.stack(processed_images, dim=0)  # (B, max_images, C, H, W, D)
    processed_times = torch.stack(processed_times, dim=0)    # (B, max_images)

    return processed_images, processed_times



def process_batch_non_zero(batch_x, batch_y=None, time_points=None, max_images=8):
    B, N, C, D, H, W = batch_x.shape
    filtered_list = []
    if time_points is not None:
        time_points = time_points.to(batch_x.device)  # todo: move the timepoints to the device beforehand
    tp_list = []
    for b in range(B):
        # get the non zero indices
        mask = batch_x[b].sum(dim=(1, 2, 3, 4)) != 0  # (N,)
        valid_idx = torch.where(mask)[0]  # (T,)
        if valid_idx.numel() == 0:
            # if no valid indices, repeat the first image so every sample keeps max_images frames
            idxs = torch.zeros(max_images, dtype=torch.long, device=batch_x.device)
        else:
            if valid_idx.numel() > max_images:
                # get the last images
                idxs = valid_idx[-max_images:]
            else:
                # need to pad with the last image
                pad_count = max_images - valid_idx.numel()
                pad_idx = valid_idx[-1].repeat(pad_count)
                idxs = torch.cat([valid_idx, pad_idx], dim=0)
        filtered_list.append(batch_x[b, idxs])
        if time_points is not None:
            tp_list.append(time_points[b, idxs])
    # stack the filtered images
    filtered_images = torch.stack(filtered_list, dim=0)  # (B, T, C, D, H, W)
    if time_points is not None:
        return filtered_images, torch.stack(tp_list, dim=0)  # (B, T, C, D, H, W), (B, T)
    else:
        return filtered_images, time_points


def process_batch_non_zero_masked(batch_x, time_points=None, max_images=8):
    """process_batch_non_zero plus a (B, max_images) bool mask, False on the repeat-padded frames."""
    images, times = process_batch_non_zero(batch_x, time_points=time_points, max_images=max_images)
    n_valid = (batch_x.sum(dim=(2, 3, 4, 5)) != 0).sum(dim=1).clamp(min=1, max=max_images)  # (B,)
    mask = torch.arange(max_images, device=batch_x.device)[None] < n_valid[:, None]
    return images, times, mask


def _normalize_seg_for_loss(seg, loss_shape):
    """Reshape a target_seg tensor so that the spatial shape fits.
    """
    seg = seg.float()
    B = loss_shape[0]
    spatial = tuple(loss_shape[2:])
    expected = B
    for s in spatial:
        expected *= s
    if seg.numel() != expected:
        return None
    return seg.reshape(B, 1, *spatial)


def compute_roi_term(loss, target_seg, roi_dilation: int = 0, valid=None):
    """Mean of per-voxel `loss` restricted to voxels where target_seg > 0.5. Returns 0 (not
    NaN) if target_seg is None, all-zero, for safety.

    Meant to be added ON TOP OF (not replacing) the base reduced loss, scaled by a
    lambda_roi_seg hyperparameter at the call site.

    NOTE: turns out this can be really useful for stabilizing training for cronos for different settings.
    loss: per-voxel loss tensor, NOT yet reduced, shape (B, C', *spatial).
    roi_dilation: grows the mask by this many voxels (max-pool dilation) before
    use -- helps when the ROI is tiny (e.g. small tumors), where too few voxels
    give a sparse/noisy gradient. 0 (default) = no dilation, original behavior.
    valid: optional bool tensor broadcastable to `loss`; False voxels (e.g. padded
    frames) are left out of both the sum and the voxel count.
    """
    if target_seg is None:
        return loss.new_zeros(())
    seg = _normalize_seg_for_loss(target_seg, loss.shape)
    if seg is None:
        return loss.new_zeros(())
    roi_mask = (seg.to(loss.device) > 0.5).float()
    if roi_dilation > 0:
        # roi_mask has leading singleton dims to broadcast against loss's full
        # rank (e.g. (B,1,1,D,H,W) vs a (B,T,C,D,H,W) loss) -- max_pool3d only
        # takes 4D/5D, so flatten everything but the trailing 3 spatial dims.
        k = 2 * roi_dilation + 1
        orig_shape = roi_mask.shape
        flat = roi_mask.reshape(-1, 1, *orig_shape[-3:])
        flat = F.max_pool3d(flat, kernel_size=k, stride=1, padding=roi_dilation)
        roi_mask = flat.reshape(orig_shape)
    roi_mask = (roi_mask > 0).expand_as(loss)
    if valid is not None:
        roi_mask = roi_mask & valid.to(loss.device).expand_as(loss)
    return (loss * roi_mask).sum() / roi_mask.float().sum().clamp(min=1)
