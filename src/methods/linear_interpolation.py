import torch
from torch import nn


class LinearExtrapolation(nn.Module):
    """Non-learnable baseline: fits a linear trend through the last two valid
    context frames and extrapolates to the target time.

    Degenerates to the last-context-image (LCI) baseline when only one valid
    context frame is available.

    Interface matches CRONOS / TemporalFlowMatching so it can be dropped into
    the training loop unchanged — training_step returns a zero loss.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.device = kwargs.get('device', 'cpu')
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def training_step(self, batch, batch_idx):
        return self._dummy.sum() * 0.0

    def validation_step(self, batch_x, batch_y=None, time_points=None):
        """
        batch_x:     (B, T, C, D, H, W) context frames (zeros = missing)
        time_points: (B, T+1) — first T entries are context times, last is target time
                     Missing frames have time = -1.
        Returns:     (B, C, D, H, W) prediction
        """
        device = batch_x.device
        B, T, C, D, H, W = batch_x.shape

        if time_points is None:
            ctx_times = torch.linspace(0.0, (T - 1) / T, T, device=device).unsqueeze(0).expand(B, T)
            tgt_times = torch.ones(B, device=device)
        else:
            ctx_times = time_points[:, :-1]  # (B, T)
            tgt_times = time_points[:, -1]   # (B,)

        preds = []
        for b in range(B):
            valid = (ctx_times[b] >= 0) & (batch_x[b].flatten(1).abs().sum(1) > 0)
            if not valid.any():
                preds.append(torch.zeros(C, D, H, W, device=device))
                continue

            t_ctx = ctx_times[b][valid]   # (N,)
            x_ctx = batch_x[b][valid]     # (N, C, D, H, W)

            x_last = x_ctx[-1]
            if valid.sum() == 1:
                # only one frame available: zero-order extrapolation (= LCI baseline)
                preds.append(x_last)
                continue

            # linear extrapolation from the last two valid frames
            t_tgt = tgt_times[b]
            t1, t0 = t_ctx[-1], t_ctx[-2]
            x0 = x_ctx[-2]
            dt = t1 - t0
            slope = (x_last - x0) / (dt + 1e-8)
            pred = x_last + slope * (t_tgt - t1)
            preds.append(pred.clamp(0.0, 1.0))

        return torch.stack(preds, dim=0)  # (B, C, D, H, W)

    def forward(self, batch_x, batch_y=None, time_points=None, **kwargs):
        pred = self.validation_step(batch_x, batch_y, time_points)
        return pred, pred, pred
