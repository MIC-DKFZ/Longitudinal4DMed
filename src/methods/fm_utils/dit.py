"""
Adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

A small 3D Diffusion-Transformer (DiT) backbone, usable as a drop-in
alternative to the UNet-style backbones in fm_utils. Time-conditioning here
is adaLN-Zero (FiLM + a learned gate per residual branch) rather than the
plain FiLM used by TemporalFiLMAdapter elsewhere in fm_utils, but it's the
same family: a small MLP maps the time embedding to per-channel modulation
parameters.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, bias: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
        self.num_patches = math.prod(self.grid_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # NTDHW -> NLC
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads: int = 8, qkv_bias: bool = True, proj_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = bias if isinstance(bias, tuple) else (bias, bias)
        drop_probs = drop if isinstance(drop, tuple) else (drop, drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TimestepEmbedder(nn.Module):
    """Embeds scalar (or per-frame) timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, use_bias=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=use_bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=use_bias),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, freq_scale=1):
        half = dim // 2
        freqs = freq_scale * torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, freq_scale=1):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, freq_scale=freq_scale)
        return self.mlp(t_freq)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        patch_dim = math.prod(patch_size)
        self.linear = nn.Linear(hidden_size, patch_dim * out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class NanoDiT(nn.Module):
    """A small 3D DiT: patchify -> N transformer blocks with adaLN-Zero time
    conditioning -> unpatchify. `forward` takes `x` as `[tensor, conditioning]`
    to match the shared backbone contract in fm_utils.backbones; `conditioning`
    is accepted for interface parity but currently unused (no cross-attention
    or concat path implemented — that's TODO if a backbone needs it).
    """

    def __init__(self, input_size, patch_size, in_channels, out_channels=None, hidden_size=384,
                 depth=12, num_heads=16, mlp_ratio=4.0, num_classes=1, timestep_freq_scale=1.0):
        super().__init__()
        hidden_size = hidden_size * 8  # room for the 3D patch token count
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.timestep_freq_scale = timestep_freq_scale

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        if getattr(self.x_embedder.proj, "bias", None) is not None:
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        if getattr(self.final_layer.linear, "bias", None) is not None:
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        B, N, _ = x.shape
        c = self.out_channels
        gH, gW, gD = self.x_embedder.grid_size
        pH, pW, pD = self.x_embedder.patch_size
        assert N == gH * gW * gD
        x = x.reshape(B, gH, gW, gD, pH, pW, pD, c)
        x = torch.einsum("bhwdpqrc->bchpwdqr", x)
        return x.reshape(B, c, gH * pH, gW * pW, gD * pD)

    def forward(self, t, x, y=None):
        x, _conditioning = x
        x = self.x_embedder(x) + self.pos_embed  # (N, L, D)
        t = self.t_embedder(t, freq_scale=self.timestep_freq_scale)  # (N, D) or (N, T, D)
        c = t
        if c.dim() == 3:
            c = c.mean(dim=1)
        for block in self.blocks:
            x = block(x, c=c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)


# --- sine/cosine positional embedding, adapted for a 3D patch grid from
# https://github.com/facebookresearch/mae (which only handles a square 2D grid) ---

def _even_split(total, n):
    """Split `total` into `n` shares, each even, distributing the remainder
    in steps of 2 (any leftover odd unit goes on the last share)."""
    base = (total // n) // 2 * 2
    shares = [base] * n
    remainder = total - base * n
    i = 0
    while remainder >= 2:
        shares[i % n] += 2
        remainder -= 2
        i += 1
    if remainder == 1:
        shares[-1] += 1
    return shares


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    """grid_size: (gH, gW, gD) patch-grid shape. Returns (gH*gW*gD, embed_dim)."""
    gH, gW, gD = grid_size
    dH, dW, dD = _even_split(embed_dim, 3)
    pos_h = get_1d_sincos_pos_embed_from_grid(dH, np.arange(gH, dtype=np.float32)) if dH else None
    pos_w = get_1d_sincos_pos_embed_from_grid(dW, np.arange(gW, dtype=np.float32)) if dW else None
    pos_d = get_1d_sincos_pos_embed_from_grid(dD, np.arange(gD, dtype=np.float32)) if dD else None

    grid = np.zeros((gH, gW, gD, embed_dim), dtype=np.float32)
    off = 0
    if pos_h is not None:
        grid[..., off:off + dH] += pos_h[:, None, None, :]
        off += dH
    if pos_w is not None:
        grid[..., off:off + dW] += pos_w[None, :, None, :]
        off += dW
    if pos_d is not None:
        grid[..., off:off + dD] += pos_d[None, None, :, :]
    return grid.reshape(gH * gW * gD, embed_dim)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)
