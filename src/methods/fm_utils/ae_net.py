"""
Shared 3-D convolutional autoencoder (encoder + decoder) used by
LatentFMModel and any other model that wants a frozen or trainable latent
representation.

Save/load path convention:
    $RESULT_DIR/ae/{dataset}_lch{latent_ch}_nd{n_down}_ae.pt

Use ae_ckpt_path() to derive the path and save_ae() / load_ae() to
persist weights independently of the full training checkpoint.

Ported from SADM's utils/ae_net.py (unchanged — self-contained, no
TFM-specific dependencies).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _safe_groups(ch: int) -> int:
    """Largest divisor of ch that is ≤ 8, minimum 1."""
    for g in range(min(8, ch), 0, -1):
        if ch % g == 0:
            return g
    return 1


def _ch_schedule(base_ch: int, n_stages: int) -> list:
    """Channel widths per stage: base_ch → base_ch*2 → … (capped at 256)."""
    chs, ch = [], base_ch
    for _ in range(n_stages):
        chs.append(ch)
        ch = min(ch * 2, 256)
    return chs


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(_safe_groups(out_ch), out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# public modules
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Strided-conv encoder: each stage halves all spatial dims.

    Channel schedule: base_ch → base_ch*2 → … (capped at 256) → latent_ch (1×1 proj).
    """

    def __init__(self, img_ch: int, latent_ch: int, n_down: int = 3, base_ch: int = 32):
        super().__init__()
        chs = _ch_schedule(base_ch, n_down)
        self.blocks = nn.ModuleList()
        in_ch = img_ch
        for out_ch in chs:
            self.blocks.append(_ConvBlock(in_ch, out_ch, stride=2))
            in_ch = out_ch
        self.proj = nn.Conv3d(in_ch, latent_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.proj(x)


class Decoder(nn.Module):
    """Transposed-conv decoder mirroring the encoder (reverse channel schedule)."""

    def __init__(self, latent_ch: int, img_ch: int, n_up: int = 3, base_ch: int = 32):
        super().__init__()
        chs = list(reversed(_ch_schedule(base_ch, n_up)))
        self.in_proj = nn.Conv3d(latent_ch, chs[0], 1)
        self.blocks = nn.ModuleList()
        in_ch = chs[0]
        for i in range(n_up):
            out_ch = chs[i + 1] if i + 1 < len(chs) else base_ch
            self.blocks.append(nn.Sequential(
                nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
                nn.GroupNorm(_safe_groups(out_ch), out_ch),
                nn.SiLU(),
            ))
            in_ch = out_ch
        self.out = nn.Conv3d(base_ch, img_ch, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor, output_size=None) -> torch.Tensor:
        z = self.in_proj(z)
        for block in self.blocks:
            z = block(z)
        out = self.out(z)
        if output_size is not None and out.shape[2:] != torch.Size(output_size):
            out = F.interpolate(out, size=output_size, mode='trilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# persistence helpers
# ---------------------------------------------------------------------------

def ae_ckpt_path(dataset: str, latent_ch: int, n_down: int) -> Path:
    """Canonical save path — lives under $RESULT_DIR/ae/, not in checkpoints/."""
    result_dir = Path(os.getenv("RESULT_DIR", "./results"))
    return result_dir / "ae" / f"{dataset}_lch{latent_ch}_nd{n_down}_ae.pt"


def save_ae(encoder: Encoder, decoder: Decoder,
            dataset: str, latent_ch: int, n_down: int, base_ch: int, img_ch: int,
            path: "str | Path | None" = None, note: "str | None" = None) -> Path:
    p = Path(path) if path else ae_ckpt_path(dataset, latent_ch, n_down)
    p.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "latent_ch": latent_ch,
        "n_down": n_down,
        "base_ch": base_ch,
        "img_ch": img_ch,
    }
    if note:
        ckpt["note"] = note
    torch.save(ckpt, p)
    print(f"[AE] saved → {p}" + (f" ({note})" if note else ""))
    return p


def load_ae(path: "str | Path",
            device: "str | torch.device" = "cpu"):
    """Load encoder + decoder from a checkpoint. Returns (encoder, decoder, meta)."""
    ckpt = torch.load(path, map_location=device)
    enc = Encoder(ckpt["img_ch"], ckpt["latent_ch"], ckpt["n_down"], ckpt["base_ch"])
    dec = Decoder(ckpt["latent_ch"], ckpt["img_ch"], ckpt["n_down"], ckpt["base_ch"])
    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    enc.to(device)
    dec.to(device)
    print(f"[AE] loaded ← {path}")
    return enc, dec, ckpt


def load_ae_frozen(path: "str | Path",
                    device: "str | torch.device" = "cpu"):
    """Load and freeze encoder + decoder (for use in other models)."""
    enc, dec, _ = load_ae(path, device=device)
    for p in enc.parameters():
        p.requires_grad = False
    for p in dec.parameters():
        p.requires_grad = False
    return enc, dec
