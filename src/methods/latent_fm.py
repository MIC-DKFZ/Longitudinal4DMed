"""
LatentFMModel — single-run latent flow matching for 3D longitudinal medical images.

Two-phase training in one run, driven by a plateau mechanism (see
train.py's `bad_epochs` tracking and the `hasattr(model, 'plateau_function')`
hook):

  Phase 1 (pre-plateau): train a lightweight 3D convolutional AE (encoder + decoder)
      with L1 (+ optional perceptual) reconstruction on all valid (non-zero) frames.

  Phase 2 (post-plateau): freeze AE; train a flow-matching head that maps
      z_source (latent of last context frame) → z_target (latent of target frame)
      via OT-CFM, conditioned on the full context latent sequence.
      At inference: Euler-integrate from z_source, then decode.

CLI kwargs (all optional, sane defaults):
  diffusion_latent_channels  — latent channel count  (default 8)
  latent_down                — number of 2× downscaling stages (default 3 → 8×)
  training_noise             — OT-CFM sigma           (default 0.01)
  number_evals                — Euler steps at inference
  num_context                 — context sequence length
  ae_perc_weight               — perceptual loss weight (default 0.1, needs
                                  torchvision VGG16 weights — set to 0 to
                                  avoid any network access, e.g. in tests/CI)
  diffusion_mode               - Phase 2 flow matcher: OT-CFM (default) vs.
                                  VP-diffusion, same toggle as CRONOS

CLI: --model_type latent_fm

Ported from SADM's MedVP/latent_fm.py. Adapted to reuse TFM's existing
`process_batch_non_zero` (fm_utils/fm_process_utils.py) and the new
`fm_utils/ae_net.py` AE module instead of SADM's own copies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)

from methods.fm_utils.fm_process_utils import process_batch_non_zero
from methods.fm_utils.ae_net import Encoder, Decoder, ae_ckpt_path, save_ae, load_ae, _safe_groups


# ---------------------------------------------------------------------------
# FM head (operates in latent space)
# ---------------------------------------------------------------------------

class _LatentFMHead(nn.Module):
    """
    Predicts velocity in latent space.

    Conditioning: concatenated context latents z_ctx.
    State:        current FM interpolant x_t + scalar time t_fm.
    """

    def __init__(self, ctx_ch: int, latent_ch: int, feature_size: int = 32):
        super().__init__()
        in_ch = ctx_ch + latent_ch
        # round mid_ch up to a multiple of 8 for GroupNorm compatibility
        mid_ch = ((max(feature_size, 32) + 7) // 8) * 8
        self.t_proj = nn.Linear(1, mid_ch)
        self.conv1 = nn.Conv3d(in_ch, mid_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_safe_groups(mid_ch), mid_ch)
        self.conv2 = nn.Conv3d(mid_ch, latent_ch, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, z_ctx: torch.Tensor, x_t: torch.Tensor,
                t_fm: torch.Tensor) -> torch.Tensor:
        # z_ctx : (B, ctx_ch,    D', H', W')
        # x_t   : (B, latent_ch, D', H', W')
        # t_fm  : (B, 1)
        t_bias = self.t_proj(t_fm).view(t_fm.shape[0], -1, 1, 1, 1)
        h = F.silu(self.norm1(self.conv1(torch.cat([z_ctx, x_t], dim=1)) + t_bias))
        return self.conv2(h)


# ---------------------------------------------------------------------------
# Perceptual loss (VGG16 slice-based, 3D → 2D)
# ---------------------------------------------------------------------------

class _PerceptualLoss(nn.Module):
    """2D perceptual loss applied to axial slices of 3D volumes.

    Samples three depth positions (25 / 50 / 75 %) and computes L1 on
    VGG16 relu1_2 and relu2_2 feature maps. All VGG weights are frozen.
    Requires downloading torchvision's pretrained VGG16 weights on first
    use — pass ae_perc_weight=0 to skip this entirely (e.g. in tests/CI
    without network access).
    """

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.DEFAULT)
        feats = list(vgg.features.children())
        self.slice1 = nn.Sequential(*feats[:5])   # up to relu1_2
        self.slice2 = nn.Sequential(*feats[5:10])  # up to relu2_2
        for p in self.parameters():
            p.requires_grad = False

    def _feat(self, x: torch.Tensor):
        # x: (N, 1, H, W) — single-channel slice
        x = x.repeat(1, 3, 1, 1)
        if x.shape[-1] < 32 or x.shape[-2] < 32:
            x = F.interpolate(x, size=(max(x.shape[-2], 32), max(x.shape[-1], 32)),
                               mode='bilinear', align_corners=False)
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        return h1, h2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target: (B, C, D, H, W)
        D = pred.shape[2]
        loss = pred.new_zeros(1).squeeze()
        for frac in (0.25, 0.5, 0.75):
            s = int(D * frac)
            ph1, ph2 = self._feat(pred[:, :, s])
            th1, th2 = self._feat(target[:, :, s])
            loss = loss + F.l1_loss(ph1, th1) + F.l1_loss(ph2, th2)
        return loss / 3.0


# ---------------------------------------------------------------------------
# LatentFMModel
# ---------------------------------------------------------------------------

class LatentFMModel(nn.Module):
    """
    Single-run latent flow matching.

    Phase 1 (pre-plateau): AE trains with L1 (+ optional perceptual) reconstruction
        on all valid (non-zero) frames in each batch (context + target).
    Phase 2 (post-plateau): AE frozen; FM head trains in latent space via OT-CFM,
        flowing from the last valid context frame to the target.
    """

    def __init__(self, in_shape=None, **kwargs):
        super().__init__()

        self.in_shape = in_shape
        self.hparams = kwargs
        self.num_context = kwargs.get('num_context', in_shape[0])
        self.img_ch = in_shape[1]
        self.n_T = kwargs.get('number_evals', 10)

        self.latent_ch = kwargs.get('diffusion_latent_channels', 8)
        self.n_down = kwargs.get('latent_down', 3)
        self.base_ch = kwargs.get('ae_base_ch', 32)

        self.process_batch = process_batch_non_zero
        self.criterion_ae = nn.L1Loss()   # L1 avoids mean-collapse; MSE flattens SSIM
        self.criterion_fm = nn.MSELoss()  # velocity matching stays MSE

        self.perc_weight = kwargs.get('ae_perc_weight', 0.1)
        self.perc_loss_fn = _PerceptualLoss() if self.perc_weight > 0 else None

        # AE
        self.encoder = Encoder(self.img_ch, self.latent_ch, self.n_down, self.base_ch)
        self.decoder = Decoder(self.latent_ch, self.img_ch, self.n_down, self.base_ch)

        # FM head — conditioned on full context latent sequence
        ctx_ch = self.num_context * self.latent_ch
        self.fm_head = _LatentFMHead(ctx_ch, self.latent_ch, feature_size=kwargs.get('feature_size', 32))
        # TODO: review
        # diffusion_mode: same OT-CFM/VP-diffusion toggle as CRONOS. No
        # scaled_flow_time-style mismatch to guard against here (see
        # cronos.py) since t_fm is used consistently throughout this class.
        self.diffusion_mode = kwargs.get('diffusion_mode', False)
        fm_cls = VariancePreservingConditionalFlowMatcher if self.diffusion_mode \
            else ExactOptimalTransportConditionalFlowMatcher
        self.fm = fm_cls(sigma=kwargs.get('training_noise', 0.01))

        # FM head starts frozen; AE trains first
        for p in self.fm_head.parameters():
            p.requires_grad = False

        # `plat` (Phase 2 active?) is backed by a registered buffer so it
        # survives state_dict save/load — a checkpoint that reached Phase 2
        # during training must still be in Phase 2 after being reloaded for
        # eval/inference, otherwise validation_step silently falls back to
        # the Phase 1 (AE-only) branch regardless of how the model was
        # actually trained.
        self.register_buffer('_phase2_active', torch.tensor(False))
        self.plat = False
        self.plateau_activated = False
        self._ae_best_mse = float('inf')  # tracked in training_step to guard plateau
        # threshold is against L1+perc combined loss; 0.05 works well for normalised [0,1] images
        self.ae_plateau_threshold = kwargs.get('ae_plateau_threshold', 0.05)
        # hard floor: Phase 2 cannot start before this many AE training batches regardless of loss
        self.min_ae_steps = kwargs.get('min_ae_steps', 3000)

        self.writer = None
        self.global_step = 0

        # Optionally pre-load AE from a prior run (skips Phase 1 entirely).
        _ae_path = kwargs.get("ae_ckpt_path")
        if _ae_path and self.load_ae(_ae_path):
            # Fast-forward to Phase 2: freeze AE, unfreeze FM head
            for p in self.encoder.parameters():
                p.requires_grad = False
            for p in self.decoder.parameters():
                p.requires_grad = False
            for p in self.fm_head.parameters():
                p.requires_grad = True
            self.plateau_activated = True
            self.plat = True
            print("[LatentFM] Phase 2 active from loaded AE.")

    @property
    def plat(self):
        return bool(self._phase2_active.item())

    @plat.setter
    def plat(self, value):
        self._phase2_active.fill_(bool(value))

    def set_writer(self, writer):
        self.writer = writer

    # ------------------------------------------------------------------
    # plateau hook  (called by train.py's main loop when bad_epochs >= 2)
    # ------------------------------------------------------------------

    def plateau_function(self):
        """Freeze AE, unfreeze FM head. Idempotent.

        Guards against early triggering: the trigger condition is a coarse
        bad-epochs heuristic, so we only activate Phase 2 once the AE
        training loss has actually dropped below ae_plateau_threshold and
        a minimum number of steps have run.
        """
        if self.plateau_activated:
            return

        if self.global_step < self.min_ae_steps:
            print(f"[LatentFM] plateau skipped — {self.global_step}/{self.min_ae_steps} AE steps done.")
            return

        if self._ae_best_mse > self.ae_plateau_threshold:
            print(f"[LatentFM] plateau skipped — AE best loss {self._ae_best_mse:.4f} "
                  f"> threshold {self.ae_plateau_threshold:.4f}.")
            return

        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.fm_head.parameters():
            p.requires_grad = True

        self.plat = True
        self.plateau_activated = True
        print("[LatentFM] plateau reached — AE frozen, FM head activated.")
        self.save_ae()

    # ------------------------------------------------------------------
    # AE persistence
    # ------------------------------------------------------------------

    def save_ae(self, path=None, note=None):
        from pathlib import Path as _Path
        p = _Path(path) if path else ae_ckpt_path(
            self.hparams.get("dataset", "unknown"), self.latent_ch, self.n_down)
        if p.is_file():
            print(f"[LatentFM] AE already exists, skipping save: {p}")
            return
        save_ae(self.encoder, self.decoder,
                dataset=self.hparams.get("dataset", "unknown"),
                latent_ch=self.latent_ch, n_down=self.n_down,
                base_ch=self.base_ch, img_ch=self.img_ch,
                path=p, note=note)

    def finalize_training(self):
        """Force-save the AE if the organic plateau trigger never fired.

        A healthy, continuously-improving AE can run out the whole training
        budget without plateau_function() ever activating, leaving no AE
        checkpoint at all. Called once at the end of training (train.py)
        regardless of how it went (normal completion or KeyboardInterrupt)
        as a safety net.
        """
        if self.plateau_activated:
            return
        print("[LatentFM] plateau never triggered — force-saving AE at the last epoch.")
        self.save_ae(note="saved at last epoch — plateau not triggered")

    def load_ae(self, path=None):
        from pathlib import Path as _Path
        p = _Path(path) if path else ae_ckpt_path(
            self.hparams.get("dataset", "unknown"), self.latent_ch, self.n_down)
        if not p.is_file():
            print(f"[LatentFM] AE checkpoint not found: {p}")
            return False
        enc, dec, _ = load_ae(p, device="cpu")
        self.encoder.load_state_dict(enc.state_dict())
        self.decoder.load_state_dict(dec.state_dict())
        return True

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _prep_batch(self, batch: dict):
        device = self.hparams.get('device', 'cpu')
        batch_x = batch['context'].to(device)
        batch_y = batch['target_img'].to(device)
        ctx_time = batch['context_time'].to(device)
        tgt_time = batch['target_time'].to(device)

        time_vec = torch.cat([ctx_time, tgt_time], dim=1)
        B, T, C, D, H, W = batch_x.shape

        context, _ = self.process_batch(
            batch_x, batch_y=batch_y, time_points=time_vec,
            max_images=self.num_context,
        )

        target = batch_y.reshape(B, C, D, H, W)
        return context, target  # (B, T, C, D, H, W), (B, C, D, H, W)

    def _encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """Encode each context frame independently → (B, T*latent_ch, D', H', W')."""
        B, T, C, D, H, W = context.shape
        z = self.encoder(context.reshape(B * T, C, D, H, W))
        _, Lc, Ld, Lh, Lw = z.shape
        return z.reshape(B, T * Lc, Ld, Lh, Lw)

    def _last_valid(self, context: torch.Tensor) -> torch.Tensor:
        """Return the last non-zero frame for each item in the batch. (B, C, D, H, W)"""
        B, T, C, D, H, W = context.shape
        # (B, T) — energy per frame
        energy = context.reshape(B, T, -1).abs().sum(-1)
        # index of last non-zero frame; fall back to 0 if all zero (shouldn't happen)
        idx = T - 1 - energy.flip(1).bool().long().argmax(1)  # (B,)
        return context[torch.arange(B, device=context.device), idx]  # (B, C, D, H, W)

    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        context, target = self._prep_batch(batch)
        B, T, C, D, H, W = context.shape

        loss_ae = None
        loss_fm = None

        if not self.plat:
            # Phase 1: AE reconstruction of every valid (non-zero) frame in the batch.
            all_frames = torch.cat([context.reshape(B * T, C, D, H, W), target], dim=0)
            valid = all_frames.flatten(1).abs().sum(1) > 0
            frames = all_frames[valid]
            if frames.numel() == 0:
                return torch.zeros(1, device=target.device, requires_grad=True)
            z = self.encoder(frames)
            I_recon = self.decoder(z, output_size=frames.shape[2:])
            loss_ae = self.criterion_ae(I_recon, frames)
            if self.perc_loss_fn is not None:
                loss_ae = loss_ae + self.perc_weight * self.perc_loss_fn(I_recon, frames)
            loss = loss_ae
            if loss_ae.item() < self._ae_best_mse:
                self._ae_best_mse = loss_ae.item()

        else:
            # Phase 2: OT-CFM in latent space (AE frozen)
            with torch.no_grad():
                z_ctx = self._encode_context(context)    # (B, T*latent_ch, D', H', W')
                I_src = self._last_valid(context)         # (B, C, D, H, W)
                z_src = self.encoder(I_src)
                z_tgt = self.encoder(target)

            t_fm, x_t, u_t = self.fm.sample_location_and_conditional_flow(z_src, z_tgt)
            v_pred = self.fm_head(z_ctx, x_t, t_fm.view(B, 1))
            loss_fm = self.criterion_fm(v_pred, u_t)
            loss = loss_fm

        if self.writer is not None:
            s = self.global_step
            if loss_ae is not None:
                self.writer.add_scalar('LatentFM/loss_ae', loss_ae.item(), s)
            if loss_fm is not None:
                self.writer.add_scalar('LatentFM/loss_fm', loss_fm.item(), s)
        self.global_step += 1

        return loss

    # ------------------------------------------------------------------
    # validation step
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_y=None, time_points=None):
        """Returns (B, D, H, W) predicted target image."""
        with torch.no_grad():
            context, _ = self.process_batch(
                batch, time_points=time_points, max_images=self.num_context,
            )
            B, T, C, D, H, W = context.shape
            I_src = self._last_valid(context)  # (B, C, D, H, W)

            if not self.plat:
                # Phase 1: AE roundtrip of the target — gives a true reconstruction
                # SSIM/PSNR rather than source-vs-target which mixes AE error with
                # temporal change.
                target = batch_y.reshape(B, C, D, H, W) if batch_y is not None else I_src
                z_tgt = self.encoder(target)
                I_pred = self.decoder(z_tgt, output_size=(D, H, W))
            else:
                # Phase 2: Euler-integrate FM in latent space, then decode
                z_ctx = self._encode_context(context)
                z = self.encoder(I_src)
                dt = 1.0 / self.n_T
                for i in range(self.n_T):
                    t_i = torch.full((B, 1), i * dt, device=z.device)
                    z = z + dt * self.fm_head(z_ctx, z, t_i)
                I_pred = self.decoder(z, output_size=(D, H, W))

        return I_pred.mean(dim=1)   # (B, D, H, W)
