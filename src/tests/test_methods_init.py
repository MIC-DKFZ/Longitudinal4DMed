"""
Method-level smoke tests, adapted from the local SADM/Synthwave repo's
tests/test_fm_backbones.py::test_meanflow_init_with_backbone (which tests
`MedVP.mean_flow.MeanFlow` — a class TFM doesn't have). TFM's equivalents are
TemporalFlowMatching and CRONOS, each now built on the same build_backbone()
factory (see fm_utils/backbones.py), so this parametrizes over both methods
and every working backbone.

Unlike the SADM original (init-only, "avoids heavy MONAI deps"), this also
runs a full training_step + backward + validation_step — TFM's classes don't
hit the MONAI dependency at this layer, and the extra coverage is exactly
what caught three real bugs (a kwarg collision, a missing `adjoint_params`,
and a broken DiT positional-embedding init) when this was done manually
during development. See TODO.md's "pluggable-backbone" entry.
"""

import pytest
import torch

from tests.conftest import _B, _C, _D, _T

_IN_SHAPE = (_T, _C, _D, _D, _D)  # T, C, H, W, D


def _make_batch(unet_type):
    return {
        'target_img': torch.rand(_B, 1, 1, *_IN_SHAPE[2:]),
        'context': torch.rand(_B, *_IN_SHAPE),
        'target_seg': torch.zeros(_B, 1, 1, *_IN_SHAPE[2:]),
        'context_seg': torch.zeros(_B, *_IN_SHAPE),
        'target_time': torch.rand(_B, 1),
        'context_time': torch.rand(_B, _T),
    }


@pytest.mark.parametrize("unet_type", ["fmu", "dit", "convlstm"])
def test_temporal_flow_matching_backbone(unet_type):
    from methods.temporal_flow_matching_method import TemporalFlowMatching

    model = TemporalFlowMatching(in_shape=_IN_SHAPE, feature_size=4, unet_type=unet_type,
                                  fm_model_unet_expands=[1, 1], device='cpu', number_evals=3)
    batch = _make_batch(unet_type)

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), "training loss is NaN/Inf"
    loss.backward()

    val_batch = torch.rand(_B, *_IN_SHAPE)
    time_points = torch.linspace(0, 1, _T + 1).unsqueeze(0).expand(_B, -1)
    out = model.validation_step(val_batch, time_points=time_points)
    assert out.shape == (_B, 1, _D, _D, _D)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("unet_type", ["fmu", "dit", "convlstm"])
def test_cronos_backbone(unet_type):
    from methods.cronos import CRONOS

    model = CRONOS(in_shape=_IN_SHAPE, feature_size=4, unet_type=unet_type,
                    fm_model_unet_expands=[1, 1], device='cpu', number_evals=3,
                    unfreeze_epoch_ratio=0.0)
    batch = _make_batch(unet_type)

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), "training loss is NaN/Inf"
    loss.backward()

    val_batch = torch.rand(_B, *_IN_SHAPE)
    time_points = torch.linspace(0, 1, _T + 1).unsqueeze(0).expand(_B, -1)
    out = model.validation_step(val_batch, time_points=time_points)
    assert torch.isfinite(out).all()


def test_temporal_flow_matching_s4nd():
    pytest.importorskip("methods.fm_utils.s4nd", reason="S4ND backbone not ported yet, see TODO_S4ND.md")
    from methods.temporal_flow_matching_method import TemporalFlowMatching
    model = TemporalFlowMatching(in_shape=_IN_SHAPE, feature_size=4, unet_type='s4nd',
                                  fm_model_unet_expands=[1, 1], device='cpu', number_evals=3)
    loss = model.training_step(_make_batch('s4nd'), 0)
    assert torch.isfinite(loss)


def test_cronos_s4nd():
    pytest.importorskip("methods.fm_utils.s4nd", reason="S4ND backbone not ported yet, see TODO_S4ND.md")
    from methods.cronos import CRONOS
    model = CRONOS(in_shape=_IN_SHAPE, feature_size=4, unet_type='s4nd',
                    fm_model_unet_expands=[1, 1], device='cpu', number_evals=3,
                    unfreeze_epoch_ratio=0.0)
    loss = model.training_step(_make_batch('s4nd'), 0)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# LatentFMModel (TODO.md item 8) — AE phase (1) + forced FM phase (2)
# ---------------------------------------------------------------------------

def test_latent_fm_ae_and_fm_phase():
    from methods.latent_fm import LatentFMModel

    model = LatentFMModel(in_shape=_IN_SHAPE, feature_size=4, device='cpu', number_evals=3,
                           ae_perc_weight=0.0, diffusion_latent_channels=4, latent_down=1,
                           ae_base_ch=4, min_ae_steps=0)
    batch = _make_batch('unet')
    val_batch = torch.rand(_B, *_IN_SHAPE)
    time_points = torch.linspace(0, 1, _T + 1).unsqueeze(0).expand(_B, -1)

    # Phase 1: AE reconstruction
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), "AE-phase training loss is NaN/Inf"
    loss.backward()

    out = model.validation_step(val_batch, batch['target_img'], time_points=time_points)
    assert torch.isfinite(out).all()

    # Force Phase 2 (bypasses the real plateau trigger + its own AE freezing) and
    # exercise the FM path — plateau_function() unfreezes fm_head, mirror that here.
    model.plat = True
    for p in model.fm_head.parameters():
        p.requires_grad = True
    model.zero_grad()
    loss_fm = model.training_step(batch, 1)
    assert torch.isfinite(loss_fm), "FM-phase training loss is NaN/Inf"
    loss_fm.backward()

    out_fm = model.validation_step(val_batch, batch['target_img'], time_points=time_points)
    assert torch.isfinite(out_fm).all()


# ---------------------------------------------------------------------------
# DeformFlowModel (TODO.md item 9) — base SVF path + with_residual FM path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("backbone", ["unet", "dit", "convlstm"])
def test_deform_flow_backbone(backbone):
    from methods.deform_flow import DeformFlowModel

    model = DeformFlowModel(in_shape=_IN_SHAPE, backbone=backbone, feature_size=4,
                             device='cpu', number_evals=3, ss_steps=2,
                             fm_model_unet_expands=[1, 1])
    batch = _make_batch(backbone)

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), "training loss is NaN/Inf"
    loss.backward()

    val_batch = torch.rand(_B, *_IN_SHAPE)
    time_points = torch.linspace(0, 1, _T + 1).unsqueeze(0).expand(_B, -1)
    out = model.validation_step(val_batch, time_points=time_points)
    assert torch.isfinite(out).all()


def test_deform_flow_with_residual():
    from methods.deform_flow import DeformFlowModel

    model = DeformFlowModel(in_shape=_IN_SHAPE, backbone='unet', feature_size=4,
                             device='cpu', number_evals=3, ss_steps=2,
                             fm_model_unet_expands=[1, 1],
                             with_residual=True, joint_training=True)
    batch = _make_batch('unet')

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss), "training loss is NaN/Inf"
    loss.backward()

    val_batch = torch.rand(_B, *_IN_SHAPE)
    time_points = torch.linspace(0, 1, _T + 1).unsqueeze(0).expand(_B, -1)
    out = model.validation_step(val_batch, time_points=time_points)
    assert torch.isfinite(out).all()


def test_deform_flow_lambda_ncc_not_implemented():
    from methods.deform_flow import DeformFlowModel

    with pytest.raises(NotImplementedError):
        DeformFlowModel(in_shape=_IN_SHAPE, backbone='unet', feature_size=4,
                         device='cpu', fm_model_unet_expands=[1, 1], lambda_ncc=0.1)
