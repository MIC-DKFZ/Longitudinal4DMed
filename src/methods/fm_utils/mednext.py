# TODO: review
"""3D MedNeXt backbone (Roy et al., MICCAI 2023), drop-in alternative to
the UNet/DiT backbones in fm_utils. FiLM time-conditioned residual
ConvNeXt-style blocks; down/up-sampling are themselves residual blocks
with a strided/transposed depthwise conv (paper Fig. 2)."""
import torch
from torch import nn

from .time_conditioning import TemporalFiLMAdapter


def _group_norm_groups(channels, max_groups=8):
    """Largest divisor of `channels` that is <= max_groups (GroupNorm
    requires channels % num_groups == 0; min(max_groups, channels) alone
    isn't safe when channels isn't a multiple of it, e.g. channels=10)."""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


# TODO: review
class MedNeXtBlock(nn.Module):
    """Residual ConvNeXt block: dwconv -> GroupNorm -> FiLM(t) -> expand ->
    GELU -> compress. `resample` in (None, 'down', 'up') also strides the
    dwconv and the residual 1x1x1 proj, so up/downsampling is one block."""

    def __init__(self, in_channels, out_channels, time_embed_dim, expand_ratio=4,
                 kernel_size=7, resample=None, num_groups=8):
        super().__init__()
        assert resample in (None, 'down', 'up')
        self.resample = resample
        hidden = out_channels * expand_ratio

        if resample == 'down':
            self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=2,
                                     padding=kernel_size // 2, groups=in_channels)
            self.residual_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2)
        elif resample == 'up':
            self.dwconv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2,
                                              groups=in_channels)
            self.residual_proj = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        else:
            self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                     padding=kernel_size // 2, groups=in_channels)
            self.residual_proj = nn.Conv3d(in_channels, out_channels, kernel_size=1) \
                if in_channels != out_channels else nn.Identity()

        self.norm = nn.GroupNorm(_group_norm_groups(in_channels, num_groups), in_channels)
        self.film = TemporalFiLMAdapter(time_embed_dim=time_embed_dim, hidden_dim=in_channels,
                                         feature_dim=in_channels)
        self.pw_expand = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.pw_compress = nn.Conv3d(hidden, out_channels, kernel_size=1)

    def forward(self, x, t_emb):
        h = self.dwconv(x)
        h = self.norm(h)
        h = self.film(h, t_emb)
        h = self.pw_expand(h)
        h = self.act(h)
        h = self.pw_compress(h)
        return h + self.residual_proj(x)


# TODO: review
class MedNeXt(nn.Module):
    """Encoder-decoder MedNeXt matching the shared `forward(t, x_list)`
    backbone contract. `conditioning` (x_list[1]) may be None or a tensor
    depending on caller, so it's projected to stem width and added
    post-stem (like SpatialAdapter) rather than concatenated at input."""

    def __init__(self, in_channels, feature_size, channel_mult=(1, 2, 4, 8), num_res_blocks=2,
                 out_channels=None, expand_ratio=4, kernel_size=7, cond_channels=1,
                 time_embed_dim=1, **kwargs):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        channel_mult = list(channel_mult)
        widths = [int(feature_size * m) for m in channel_mult]

        self.stem = nn.Conv3d(in_channels, widths[0], kernel_size=1)
        self.cond_proj = nn.Conv3d(cond_channels, widths[0], kernel_size=1)

        self.encoder_stages = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        ch = widths[0]
        for i, w in enumerate(widths):
            stage = nn.ModuleList([
                MedNeXtBlock(ch if j == 0 else w, w, time_embed_dim, expand_ratio, kernel_size)
                for j in range(num_res_blocks)
            ])
            self.encoder_stages.append(stage)
            ch = w
            if i < len(widths) - 1:
                self.down_blocks.append(MedNeXtBlock(ch, widths[i + 1], time_embed_dim, expand_ratio,
                                                       kernel_size, resample='down'))
                ch = widths[i + 1]
            else:
                self.down_blocks.append(None)

        self.bottleneck = nn.ModuleList([
            MedNeXtBlock(ch, ch, time_embed_dim, expand_ratio, kernel_size) for _ in range(num_res_blocks)
        ])

        self.up_blocks = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for i in reversed(range(len(widths) - 1)):
            self.up_blocks.append(MedNeXtBlock(ch, widths[i], time_embed_dim, expand_ratio,
                                                 kernel_size, resample='up'))
            ch = widths[i]
            stage = nn.ModuleList([
                MedNeXtBlock(ch * 2 if j == 0 else ch, ch, time_embed_dim, expand_ratio, kernel_size)
                for j in range(num_res_blocks)
            ])
            self.decoder_stages.append(stage)

        self.out_norm = nn.GroupNorm(_group_norm_groups(ch), ch)
        self.out_conv = nn.Conv3d(ch, out_channels, kernel_size=1)

    def forward(self, t, x_list):
        z, conditioning = x_list
        t_emb = t.reshape(z.shape[0], -1).mean(dim=1, keepdim=True).float()

        h = self.stem(z)
        if conditioning is not None:
            if conditioning.shape[2:] != h.shape[2:]:
                conditioning = nn.functional.interpolate(conditioning, size=h.shape[2:], mode='trilinear',
                                                           align_corners=False)
            h = h + self.cond_proj(conditioning)

        skips = []
        for stage, down in zip(self.encoder_stages, self.down_blocks):
            for block in stage:
                h = block(h, t_emb)
            skips.append(h)
            if down is not None:
                h = down(h, t_emb)

        for block in self.bottleneck:
            h = block(h, t_emb)

        skips.pop()  # bottleneck already consumed the deepest stage's output
        for up, stage in zip(self.up_blocks, self.decoder_stages):
            h = up(h, t_emb)
            h = torch.cat([h, skips.pop()], dim=1)
            for block in stage:
                h = block(h, t_emb)

        h = self.out_norm(h)
        return self.out_conv(h)
