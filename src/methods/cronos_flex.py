"""CRONOSFlex: CRONOS with the time axis kept separate from the channel axis.

Data stays (B, T, C, D, H, W), C = modalities. Each context frame flows to the
target on its own path, x_i(tau) = (1 - tau) x_i + tau y, with the per-frame
velocity from PerFrameUNet (see fm_utils/per_frame_unet.py). The prediction
aggregates the T end states ('mean' or 'last'), never the C modalities, so the
output is (B, C, D, H, W).
"""
import torch
from torch import nn
import torch.nn.functional as F
from torchdiffeq import odeint
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher

from .fm_utils.fm_process_utils import compute_roi_term, process_batch_non_zero_masked
from .fm_utils.per_frame_unet import PerFrameUNet


class CRONOSFlex(nn.Module):
    AGGREGATES = ('auto', 'mean', 'last')

    def __init__(self, in_shape=None, **kwargs):
        super().__init__()
        self.hparams = kwargs
        self.in_shape = in_shape
        self.num_channels = in_shape[1]
        self.num_context = kwargs.get('num_context', in_shape[0])
        self.n_T = int(kwargs.get('number_evals', 50))
        self.training_noise = kwargs.get('training_noise', 0.01)
        self.random_inference = kwargs.get('random_inference', False)
        self.mask_noise_by_foreground = kwargs.get('mask_noise_by_foreground', True)
        self.lambda_roi_seg = kwargs.get('lambda_roi_seg', 0.0)
        self.roi_dilation = kwargs.get('roi_dilation', 0)
        # 'auto' matches CRONOS: last frame after SDE inference, mean after ODE inference
        self.aggregate = kwargs.get('aggregate', 'auto')
        if self.aggregate not in self.AGGREGATES:
            raise ValueError(f'aggregate must be one of {self.AGGREGATES}, got {self.aggregate!r}')

        self.u_net = PerFrameUNet(
            in_channels=self.num_channels, out_channels=self.num_channels,
            spatial_shape=tuple(in_shape[2:]),
            feature_size=kwargs.get('feature_size', 32),
            channel_mult=kwargs.get('fm_model_unet_expands', [1, 1, 2, 4]),
            frame_encoder=kwargs.get('frame_encoder', 'conv'),
            cross_frame=kwargs.get('cross_frame', 'attn_deepest'),
            mask_time=kwargs.get('mask_time', 0.0),
            frame_encoder_ckpt=kwargs.get('frame_encoder_ckpt'),
            freeze_frame_encoder=kwargs.get('freeze_frame_encoder', False),
        )
        # plain (paired) CFM: the minibatch-OT variant resamples and reorders the batch,
        # which would mismatch each context with its own times, target and seg
        self.fm = ConditionalFlowMatcher(sigma=self.training_noise)

    def pretrained_parameters(self):
        return self.u_net.pretrained_parameters()

    def _prep_context(self, batch_x, time_points):
        """(B, N, C, ...) context + (B, N+1) times -> (B, T, C, ...), (B, T+1) times, (B, T) mask."""
        target_time = time_points[:, -1:]
        context, times, frame_mask = process_batch_non_zero_masked(
            batch_x, time_points=time_points, max_images=self.num_context)
        return context, torch.cat([times, target_time.to(times.device)], dim=1), frame_mask

    @staticmethod
    def _frame_times(tau, time_points):
        """t_i(tau) = (1 - tau) t_i + tau t_target, (B, T)."""
        tau = tau.reshape(-1, 1)
        return (1 - tau) * time_points[:, :-1] + tau * time_points[:, -1:]

    def _aggregate(self, states, frame_mask, stochastic):
        mode = self.aggregate if self.aggregate != 'auto' else ('last' if stochastic else 'mean')
        if mode == 'last':
            last = frame_mask.long().sum(dim=1) - 1  # padding sits at the end
            return states[torch.arange(states.shape[0], device=states.device), last]
        w = frame_mask.float().view(*frame_mask.shape, 1, 1, 1, 1)
        return (states * w).sum(dim=1) / w.sum(dim=1)

    def training_step(self, batch, batch_idx):
        device = self.hparams.get('device', 'cpu')
        batch_x = batch['context'].to(device)
        batch_y = batch['target_img'].to(device)
        time_vec = torch.cat([batch['context_time'], batch['target_time']], dim=1).to(device)
        context, time_points, frame_mask = self._prep_context(batch_x, time_vec)
        B, T = context.shape[:2]

        target = batch_y.reshape(B, 1, *context.shape[2:]).expand_as(context)
        tau, xt, ut = self.fm.sample_location_and_conditional_flow(context, target)
        vt = self.u_net(self._frame_times(tau, time_points), [xt, None], frame_mask=frame_mask)

        # padded frames are repeats, keep them out of the loss
        per_voxel = F.mse_loss(vt, ut, reduction='none')
        w = frame_mask.float().view(B, T, 1, 1, 1, 1).expand_as(per_voxel)
        loss = (per_voxel * w).sum() / w.sum()
        if self.lambda_roi_seg > 0:
            flat = lambda x: x.reshape(B, -1, *per_voxel.shape[3:])  # (B, T*C, ...), seg broadcasts over T*C
            loss = loss + self.lambda_roi_seg * compute_roi_term(
                flat(per_voxel), batch.get('target_seg'), roi_dilation=self.roi_dilation, valid=flat(w > 0))
        return loss

    def validation_step(self, batch, batch_y=None, time_points=None):
        """Returns the predicted target, (B, C, D, H, W)."""
        context, time_points, frame_mask = self._prep_context(batch, time_points.to(batch.device))

        def velocity(tau, x):
            tau = tau.reshape(1).expand(x.shape[0])
            return self.u_net(self._frame_times(tau, time_points), [x, None], frame_mask=frame_mask)

        stochastic = bool(self.training_noise and self.random_inference)
        with torch.no_grad():
            if stochastic:
                # Euler-Maruyama as in CRONOS, noise zeroed on background if enabled
                mask = (context > 0.05).float() if self.mask_noise_by_foreground else torch.ones_like(context)
                t_span = torch.linspace(0, 1, self.n_T, device=context.device)
                dt = t_span[1] - t_span[0]
                x = context
                for tau in t_span:
                    x = x + velocity(tau, x) * dt + self.training_noise * torch.randn_like(x) * dt.sqrt() * mask
            else:
                t_span = torch.linspace(0, 1, self.n_T, device=context.device)
                x = odeint(velocity, context, t_span, atol=1e-5, rtol=1e-5)[-1]
        return self._aggregate(x, frame_mask, stochastic)
