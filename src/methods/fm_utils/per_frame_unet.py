"""Per-frame velocity UNet for CRONOSFlex: the time axis stays separate end to end.

Input z is (B, T, C, D, H, W) with C = modalities. Every frame runs through the
same encoder/decoder, FiLM-modulated by its own time t_i(tau), and frames only
interact through time-biased attention over T. Output is one C-channel velocity
per frame, (B, T, C, D, H, W); T and C are never folded together.

The encoder sits behind `FrameEncoder` so a pretrained encoder can be swapped in
via `build_frame_encoder` without touching the decoder or CRONOSFlex.
"""
import math
import os

import torch
from torch import nn
import torch.nn.functional as F

from .image_cond_unet import timestep_embedding


def _groups(ch, max_groups=8):
    g = min(max_groups, ch)
    while ch % g:
        g -= 1
    return g


def auto_strides(spatial_shape, n_levels, min_size=4):
    """Per-level, per-axis downsampling strides; thin or odd axes stop being halved."""
    size = list(spatial_shape)
    strides = [(1, 1, 1)]
    for _ in range(n_levels - 1):
        s = tuple(2 if (n % 2 == 0 and n // 2 >= min_size) else 1 for n in size)
        size = [n // k for n, k in zip(size, s)]
        strides.append(s)
    return strides


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(in_ch), in_ch), nn.SiLU(), nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(_groups(out_ch), out_ch), nn.SiLU(), nn.Conv3d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.net(x)


class FrameEncoder(nn.Module):
    """Interface: (N, C_in, D, H, W) -> list of skips, finest first.

    Subclasses set `output_channels` and `strides` (one entry per skip, strides[0]
    is (1, 1, 1)) so the decoder can size itself. A pretrained encoder also
    overrides `pretrained_parameters` so the method can freeze / lr-scale it.
    """
    output_channels: list
    strides: list

    def pretrained_parameters(self):
        return iter(())


class ConvFrameEncoder(FrameEncoder):
    """Freshly initialised conv encoder, one ResBlock per level."""

    def __init__(self, in_channels, spatial_shape, feature_size=32, channel_mult=(1, 1, 2, 4)):
        super().__init__()
        self.output_channels = [feature_size * m for m in channel_mult]
        self.strides = auto_strides(spatial_shape, len(channel_mult))
        self.stem = nn.Conv3d(in_channels, self.output_channels[0], 3, padding=1)
        self.levels = nn.ModuleList()
        prev = self.output_channels[0]
        for ch, s in zip(self.output_channels, self.strides):
            down = nn.Identity() if s == (1, 1, 1) else nn.Conv3d(prev, prev, 3, stride=s, padding=1)
            self.levels.append(nn.Sequential(down, ResBlock(prev, ch)))
            prev = ch

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for level in self.levels:
            h = level(h)
            skips.append(h)
        return skips


def build_frame_encoder(name, *, in_channels, spatial_shape, feature_size, channel_mult, ckpt=None, freeze=False):
    if name == 'conv':
        return ConvFrameEncoder(in_channels, spatial_shape, feature_size, channel_mult)
    if name == 'resenc':
        # never fall back to random init silently: 'scratch' is the explicit opt-in
        if ckpt is None:
            raise ValueError("frame_encoder 'resenc' needs frame_encoder_ckpt (a path, or 'scratch')")
        from .resenc_encoder import ResEncFrameEncoder
        if ckpt != 'scratch':
            ckpt = os.path.expandvars(ckpt)  # configs can say $RESENC_CKPT instead of a machine path
            if not os.path.isfile(ckpt):
                raise FileNotFoundError(f'frame_encoder_ckpt not found: {ckpt!r}')
        return ResEncFrameEncoder(in_channels, spatial_shape, None if ckpt == 'scratch' else ckpt, freeze)
    raise ValueError(f'unknown frame_encoder {name!r}')


class FiLM(nn.Module):
    """Per-frame scale/shift from a time embedding; zero-init so it starts as identity."""

    def __init__(self, emb_dim, ch):
        super().__init__()
        self.lin = nn.Linear(emb_dim, 2 * ch)
        nn.init.zeros_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x, emb):
        scale, shift = self.lin(F.silu(emb))[..., None, None, None].chunk(2, dim=1)
        return x * (1 + scale) + shift


class TimeBiasedFrameAttention(nn.Module):
    """Attention across the T frames of a sample on spatially pooled features, with
    additive bias -alpha_h * |t_i - t_j| and invalid (padded) frames masked out as keys.
    Zero-init output projection, so the residual starts at exactly zero."""

    def __init__(self, ch, heads=4):
        super().__init__()
        while ch % heads:
            heads -= 1
        self.heads, self.dh = heads, ch // heads
        self.qkv = nn.Linear(ch, 3 * ch)
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        self.out = nn.Linear(ch, ch)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.alpha_raw = nn.Parameter(torch.full((heads,), math.log(math.e - 1)))  # softplus -> 1.0

    def forward(self, h, t, B, T, frame_mask=None):
        C = h.shape[1]
        pooled = h.mean(dim=(2, 3, 4)).view(B, T, C)
        q, k, v = self.qkv(pooled).view(B, T, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)  # (B, H, T, dh)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.dh)
        dist = (t[:, :, None] - t[:, None, :]).abs()  # (B, T, T)
        logits = logits - F.softplus(self.alpha_raw)[None, :, None, None] * dist[:, None]
        if frame_mask is not None:
            logits = logits.masked_fill(~frame_mask[:, None, None, :], float('-inf'))
        o = (logits.softmax(-1) @ v).transpose(1, 2).reshape(B, T, C)
        return h + self.out(o).reshape(B * T, C, 1, 1, 1)


class PerFrameUNet(nn.Module):
    """forward(t, [z, cond], frame_mask=None) -> velocity, shapes:
        t          : (B, T) per-frame times t_i(tau)
        z          : (B, T, C, D, H, W)
        frame_mask : (B, T) bool, True = real frame (None = all real)
        returns    : (B, T, out_channels, D, H, W)
    """

    CROSS_FRAME = ('none', 'attn_deepest', 'attn_all')

    def __init__(self, in_channels, out_channels, spatial_shape, feature_size=32,
                 channel_mult=(1, 1, 2, 4), frame_encoder='conv', cross_frame='attn_deepest',
                 mask_time=0.0, frame_encoder_ckpt=None, freeze_frame_encoder=False):
        super().__init__()
        if cross_frame not in self.CROSS_FRAME:
            raise ValueError(f'cross_frame must be one of {self.CROSS_FRAME}, got {cross_frame!r}')
        self.mask_time = mask_time
        self.encoder = build_frame_encoder(frame_encoder, in_channels=in_channels, spatial_shape=spatial_shape,
                                           feature_size=feature_size, channel_mult=channel_mult,
                                           ckpt=frame_encoder_ckpt, freeze=freeze_frame_encoder)
        chs, strides = list(self.encoder.output_channels), list(self.encoder.strides)
        self.model_channels = chs[0]
        emb_dim = 4 * chs[0]
        self.time_embed = nn.Sequential(nn.Linear(chs[0], emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.skip_film = nn.ModuleList(FiLM(emb_dim, c) for c in chs)

        # decoder, deepest first: upsample level l+1 to level l, fuse with skip l
        self.ups, self.dec_blocks, self.dec_film = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for l in reversed(range(len(chs) - 1)):
            s = strides[l + 1]
            up = nn.Identity() if s == (1, 1, 1) else nn.ConvTranspose3d(chs[l + 1], chs[l + 1], s, stride=s)
            self.ups.append(up)
            self.dec_blocks.append(ResBlock(chs[l + 1] + chs[l], chs[l]))
            self.dec_film.append(FiLM(emb_dim, chs[l]))

        levels = {'none': [], 'attn_deepest': [len(chs) - 1], 'attn_all': list(range(len(chs)))}[cross_frame]
        self.attn = nn.ModuleDict({str(l): TimeBiasedFrameAttention(chs[l]) for l in levels})
        self.out = nn.Sequential(nn.GroupNorm(_groups(chs[0]), chs[0]), nn.SiLU(),
                                 nn.Conv3d(chs[0], out_channels, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def pretrained_parameters(self):
        return self.encoder.pretrained_parameters()

    def forward(self, t, x, frame_mask=None):
        z, cond = x
        if cond is not None:
            raise ValueError('PerFrameUNet has no conditioning input, expected cond=None')
        B, T, C = z.shape[:3]
        spatial = z.shape[3:]
        if t.shape != (B, T):
            raise ValueError(f'per-frame times of shape {(B, T)} required, got {tuple(t.shape)}')

        t = t.float()
        if self.mask_time >= 0.5:
            t = torch.zeros_like(t)
        emb = self.time_embed(timestep_embedding(t.reshape(B * T), self.model_channels))  # (B*T, E)

        skips = self.encoder(z.reshape(B * T, C, *spatial))
        skips = [film(s, emb) for film, s in zip(self.skip_film, skips)]
        for l, attn in self.attn.items():
            skips[int(l)] = attn(skips[int(l)], t, B, T, frame_mask)

        h = skips[-1]
        for i, (up, block, film) in enumerate(zip(self.ups, self.dec_blocks, self.dec_film)):
            h = film(block(torch.cat([up(h), skips[-(i + 2)]], dim=1)), emb)
        out = self.out(h)
        return out.reshape(B, T, -1, *spatial)
