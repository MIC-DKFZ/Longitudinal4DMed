"""Pretrained nnssl ResEnc-L as a PerFrameUNet frame encoder (`frame_encoder: resenc`).

Build + strict checkpoint loading follow SADM's MedVP/pretrained_backbones.py.
Needs dynamic_network_architectures.
"""
import math
import pydoc

import torch
from torch import nn

from .per_frame_unet import FrameEncoder

# ResEnc-L as the nnssl MAE checkpoint was trained; the checkpoint's own plan overrides it
RESENC_L_DEFAULTS = dict(
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    kernel_sizes=[[3, 3, 3]] * 6,
    n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
)
_ENCODER_KEYS = tuple(RESENC_L_DEFAULTS) + ('block', 'bottleneck_channels', 'stem_channels')


def _load_nnssl_ckpt(path):
    """Returns (network_weights, adaptation_plan, arch_kwargs with imports resolved)."""
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    plan = ckpt.get('nnssl_adaptation_plan', {}) or {}
    arch = dict(plan.get('architecture_kwargs')
                or (plan.get('architecture_plans') or {}).get('arch_kwargs') or {})
    requires_import = (plan.get('architecture_plans') or {}).get('arch_kwargs_requires_import', ())
    for k, v in list(arch.items()):
        if isinstance(v, str) and (k in requires_import or v.startswith('torch.')):
            located = pydoc.locate(v)
            if located is None:
                raise ImportError(f'cannot import {k}={v!r} from checkpoint architecture plan')
            arch[k] = located
    return ckpt['network_weights'], plan, arch


def _load_strict(module, weights, prefix):
    """Load weights[prefix + k] into module[k], raising if any module tensor stays random.

    A stage with stride 1 has skip = (conv, norm) instead of (avgpool, conv, norm), so
    checkpoint key 'skip.1.x' is looked up as 'skip.0.x' when only the latter exists.
    """
    target = module.state_dict()
    matched, bad = {}, []
    for k, v in weights.items():
        if not k.startswith(prefix):
            continue
        nk = k[len(prefix):]
        if nk not in target and '.skip.1.' in nk:
            nk = nk.replace('.skip.1.', '.skip.0.')
        if nk in target:
            if target[nk].shape == v.shape:
                matched[nk] = v
            else:
                bad.append(f'{nk}: ckpt {tuple(v.shape)} vs model {tuple(target[nk].shape)}')
    missing = [k for k in target if k not in matched]
    if bad or missing:
        raise RuntimeError(f'ResEnc checkpoint does not fit: shape mismatches {bad[:5]}, missing {missing[:5]}')
    module.load_state_dict(matched)
    print(f'[ResEnc] loaded {len(matched)}/{len(target)} tensors')


def _strides(spatial_shape, n_stages, min_size=4):
    """Pretraining strides (isotropic 2x) when the volume allows it, else nnU-Net-style per-axis."""
    down = 2 ** (n_stages - 1)
    if all(s % down == 0 for s in spatial_shape) and math.prod(s // down for s in spatial_shape) > 1:
        return [[1, 1, 1]] + [[2, 2, 2]] * (n_stages - 1)
    strides, cur = [[1, 1, 1]], list(spatial_shape)
    for _ in range(n_stages - 1):
        s = [2 if (c % 2 == 0 and c // 2 >= min_size) else 1 for c in cur]
        cur = [c // k for c, k in zip(cur, s)]
        strides.append(s)
    return strides


class ResEncFrameEncoder(FrameEncoder):
    """nnssl ResEnc-L encoder; ckpt=None is random init.

    The pretrained stem takes 1 channel, so for C modalities its weights are repeated
    over C and divided by C. The stem's InstanceNorm makes the encoder blind to each
    frame's intensity scale/offset, which the velocity y - x_i depends on, so a small
    conv of the raw frame is added to the finest skip.
    """

    def __init__(self, in_channels, spatial_shape, ckpt=None, freeze=False):
        super().__init__()
        from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder

        self.pretrained = bool(ckpt)
        weights, plan, arch = _load_nnssl_ckpt(ckpt) if ckpt else ({}, {}, {})
        cfg = {**RESENC_L_DEFAULTS, **{k: v for k, v in arch.items() if k in _ENCODER_KEYS}}
        n_stages = cfg.pop('n_stages')
        self.enc = ResidualEncoder(input_channels=1, n_stages=n_stages, conv_op=nn.Conv3d,
                                   strides=_strides(spatial_shape, n_stages), return_skips=True, **cfg)
        if ckpt:
            prefix = plan.get('key_to_stem', 'encoder.stem').rsplit('stem', 1)[0]
            _load_strict(self.enc, weights, prefix)
        if in_channels != 1:
            self._inflate_stem(in_channels)
        if freeze:
            self.enc.requires_grad_(False)

        self.output_channels = list(self.enc.output_channels)
        self.strides = [tuple(s) for s in self.enc.strides]
        self.raw = nn.Conv3d(in_channels, self.output_channels[0], 3, padding=1)

    def _inflate_stem(self, in_channels):
        old = next(m for m in self.enc.stem.modules() if isinstance(m, nn.Conv3d))
        new = nn.Conv3d(in_channels, old.out_channels, old.kernel_size, old.stride, old.padding,
                        bias=old.bias is not None)
        with torch.no_grad():
            new.weight.copy_(old.weight.repeat(1, in_channels, 1, 1, 1) / in_channels)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        # DNA keeps the same conv under both .conv and .all_modules
        for parent in list(self.enc.stem.modules()):
            for name, child in list(parent.named_children()):
                if child is old:
                    setattr(parent, name, new)

    def pretrained_parameters(self):
        return self.enc.parameters() if self.pretrained else iter(())

    def forward(self, x):
        mu = x.mean(dim=(1, 2, 3, 4), keepdim=True)
        sd = x.std(dim=(1, 2, 3, 4), keepdim=True).clamp(min=1e-5)
        skips = self.enc((x - mu) / sd)
        skips[0] = skips[0] + self.raw(x)
        return skips
