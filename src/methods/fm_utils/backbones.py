"""Shared pluggable-backbone interface for the flow-matching methods
(TemporalFlowMatching, CRONOS).

Every backbone built by `build_backbone` exposes the same calling
convention:

    forward(t, x_list) -> velocity

where `x_list = [z, conditioning]` (z is (B, T*C, D, H, W), conditioning may
be None) and `t` is a per-sample or per-context-frame continuous flow-time
tensor. Adding a new architecture means adding one branch here, not touching
the two methods themselves. All conditioning-aware backbones here share one
time-conditioning primitive, `TemporalFiLMAdapter` (see time_conditioning.py) —
NanoDiT's adaLN-Zero is the transformer-block analogue of the same FiLM idea.
"""
import inspect

import torch
from torch import nn

from .image_cond_unet import ConditionedUNet
from .unet_wrapper import UNetModelWrapper
from .dit import NanoDiT
from .mednext import MedNeXt
from .time_conditioning import TemporalFiLMAdapter


def _filter_kwargs(cls, kwargs):
    valid_params = inspect.signature(cls.__init__).parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid_params}


class _BareTensorAdapter(nn.Module):
    """Wraps a `forward(t, x)` backbone (no conditioning support, e.g.
    UNetModelWrapper) behind the shared `forward(t, x_list)` contract."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, t, x_list):
        x, _conditioning = x_list
        return self.backbone(t, x)


class FiLMConvLSTMCell(nn.Module):
    """3D-conv LSTM cell, FiLM-conditioned on the current step's flow time."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv3d(in_channels + hidden_channels, 4 * hidden_channels,
                               kernel_size, padding=padding)
        self.film = TemporalFiLMAdapter(time_embed_dim=1, hidden_dim=hidden_channels,
                                         feature_dim=4 * hidden_channels)

    def forward(self, x, state, t_step):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        gates = self.film(gates, t_step.view(-1, 1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch, spatial_shape, device, dtype):
        zeros = torch.zeros(batch, self.hidden_channels, *spatial_shape, device=device, dtype=dtype)
        return zeros, zeros.clone()


class ConvLSTMBackbone(nn.Module):
    """Recurrent alternative to the UNet/DiT backbones: runs the T context
    frames through a 2-layer FiLM-conditioned ConvLSTM instead of a single
    spatial network. Simple, cheap, a useful sanity-check baseline.
    """

    def __init__(self, in_channels, feature_size, num_frames, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.num_frames = num_frames
        self.cell1 = FiLMConvLSTMCell(in_channels, feature_size)
        self.cell2 = FiLMConvLSTMCell(feature_size, in_channels)

    def forward(self, t, x_list):
        z, _conditioning = x_list  # z: (B, T*C, D, H, W)
        B = z.shape[0]
        D, H, W = z.shape[-3:]
        z_seq = z.view(B, self.num_frames, self.in_channels, D, H, W)

        # Accept any per-sample time convention: a single scalar (B, 1),
        # already-per-frame (B, num_frames), or e.g. MeanFlow's
        # [flow_time, anchor_time] (B, 2) — anything that isn't already
        # per-frame gets reduced to one scalar per sample, then broadcast.
        t = t.reshape(B, -1)
        if t.shape[1] != self.num_frames:
            t = t.mean(dim=1, keepdim=True).expand(B, self.num_frames)

        h1, c1 = self.cell1.init_state(B, (D, H, W), z.device, z.dtype)
        h2, c2 = self.cell2.init_state(B, (D, H, W), z.device, z.dtype)
        outputs = []
        for step in range(self.num_frames):
            t_step = t[:, step]
            h1, c1 = self.cell1(z_seq[:, step], (h1, c1), t_step)
            h2, c2 = self.cell2(h1, (h2, c2), t_step)
            outputs.append(h2)
        out = torch.stack(outputs, dim=1)  # (B, T, C, D, H, W)
        return out.reshape(B, self.num_frames * self.in_channels, D, H, W)


def build_backbone(unet_type, *, in_shape, num_context, feature_size,
                    fm_model_unet_expands=(1, 1, 2, 4), **kwargs):
    """Build a backbone behind the shared `forward(t, x_list)` contract.

    `in_shape` is (T, C, H, W, D). `num_context` is the flattened T*C input
    channel count fed to the backbone's conv/patch stem (== T when C == 1,
    the common case across every loader in this repo).

    `unet_type` is one of the canonical keys: 'cond_unet' (ConditionedUNet,
    FiLM-conditioned), 'unet' (bare UNetModelWrapper, no conditioning), 'dit'
    (NanoDiT), 'convlstm', 'mednext' (MedNeXt, FiLM-conditioned), 's4nd'
    (deferred). CRONOS and TemporalFlowMatching
    each translate their own legacy `unet_type` strings (both historically
    used 'fmu' for two *different* backbones) to these keys at their call
    sites — see each method's __init__.
    """
    if unet_type == 'cond_unet':
        filtered = _filter_kwargs(ConditionedUNet, kwargs)
        return ConditionedUNet(dim=(num_context,) + tuple(in_shape[2:]), num_channels=feature_size,
                                num_res_blocks=1, channel_mult=fm_model_unet_expands, **filtered)
    if unet_type == 'dit':
        # NanoDiT patches then attends over the patch grid, so it wants a
        # coarser stem than the UNet's stride-2 downsampling — match CRONOS's
        # existing convention of an extra /4 on top of the /4 default.
        patch_size = [max(1, s // 16) for s in in_shape[2:]]
        return NanoDiT(input_size=list(in_shape[2:]), patch_size=patch_size,
                        in_channels=num_context, hidden_size=feature_size)
    if unet_type == 'convlstm':
        num_frames = kwargs.get('num_frames', num_context)
        frame_channels = max(1, num_context // num_frames)
        return ConvLSTMBackbone(in_channels=frame_channels, feature_size=feature_size, num_frames=num_frames)
    if unet_type == 'mednext':
        # in_channels/feature_size/channel_mult are set explicitly below;
        # drop them from kwargs so a same-named arg (e.g. argparse's
        # --in-channels) doesn't collide as a duplicate keyword.
        filtered = _filter_kwargs(MedNeXt, kwargs)
        for key in ('in_channels', 'feature_size', 'channel_mult'):
            filtered.pop(key, None)
        return MedNeXt(in_channels=num_context, feature_size=feature_size,
                        channel_mult=fm_model_unet_expands, **filtered)
    if unet_type == 's4nd':
        raise NotImplementedError(
            "S4ND backbone is a deferred TODO (see TODO.md) — it needs the vendored "
            "S4 research codebase and isn't ported yet."
        )
    if unet_type == 'unet':
        backbone = UNetModelWrapper(dim=(num_context,) + tuple(in_shape[2:]), num_channels=feature_size,
                                     num_res_blocks=1, channel_mult=fm_model_unet_expands,
                                     use_checkpoint=True, attention_resolutions="9999")
        return _BareTensorAdapter(backbone)
    raise ValueError(f'unknown unet_type {unet_type!r}')
