import torch


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
            # if no valid indices, use the first image
            idxs = torch.tensor([0], device=batch_x.device)
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