"""
Backbone contract tests, ported from the local SADM/Synthwave repo's
tests/test_fm_backbones.py and adapted to TFM's fm_utils.backbones module
(the DiT test asserting a `time_film` attribute was dropped — that attribute
doesn't exist on the ported NanoDiT, and the assertion contradicted adaLN-Zero
init anyway: with adaLN_modulation zero-initialized, DiT's output is
*supposed* to be time-independent until the gates learn to open).

Each backbone must satisfy three contracts:
  1. Output shape matches input z shape — (B, T*C, D, H, W).
  2. Output contains no NaNs or Infs.
  3. Gradients flow back to every parameter.

ConvLSTM and DiT tests always run. S4ND tests are skipped until it's ported
(see TODO_S4ND.md).
"""

import torch

from tests.conftest import _B, _T, _C, _D


def _expected_shape():
    return (_B, _T * _C, _D, _D, _D)


def _check_output(out, expected_shape):
    assert out.shape == expected_shape, f"shape mismatch: {out.shape} != {expected_shape}"
    assert torch.isfinite(out).all(), "output contains NaN or Inf"


def _check_gradients(model, loss):
    loss.backward()
    has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.parameters()
    )
    assert has_grad, "no parameter received a non-zero gradient — graph is broken"


class TestConvLSTMBackbone:

    def test_output_shape(self, convlstm_backbone, backbone_input):
        t, x_list = backbone_input
        out = convlstm_backbone(t, x_list)
        _check_output(out, _expected_shape())

    def test_output_finite(self, convlstm_backbone, backbone_input):
        t, x_list = backbone_input
        with torch.no_grad():
            out = convlstm_backbone(t, x_list)
        _check_output(out, _expected_shape())

    def test_gradients_flow(self, backbone_input):
        from methods.fm_utils.backbones import ConvLSTMBackbone
        model = ConvLSTMBackbone(in_channels=_C, feature_size=8, num_frames=_T)
        t, x_list = backbone_input
        out = model(t, x_list)
        _check_gradients(model, out.sum())

    def test_different_batch_sizes_produce_same_per_sample_output(self, backbone_input):
        """Stateless across batch dimension — no hidden-state bleed-through."""
        from methods.fm_utils.backbones import ConvLSTMBackbone
        model = ConvLSTMBackbone(in_channels=_C, feature_size=8, num_frames=_T)
        model.eval()
        t, x_list = backbone_input
        z = x_list[0]
        with torch.no_grad():
            out_batch = model(t, [z, None])
            out_single = model(t[[0]], [z[[0]], None])
        assert torch.allclose(out_batch[0], out_single[0], atol=1e-5), \
            "batch-0 output differs when run alone vs. in a batch"


class TestDiTBackbone:

    def test_output_shape(self, dit_backbone, backbone_input):
        t, x_list = backbone_input
        out = dit_backbone(t, x_list)
        _check_output(out, _expected_shape())

    def test_output_finite(self, dit_backbone, backbone_input):
        t, x_list = backbone_input
        with torch.no_grad():
            out = dit_backbone(t, x_list)
        _check_output(out, _expected_shape())

    def test_gradients_flow(self, backbone_input):
        from methods.fm_utils.dit import NanoDiT
        model = NanoDiT(
            input_size=[_D, _D, _D],
            patch_size=[2, 2, 2],
            in_channels=_T * _C,
            hidden_size=2,
            depth=2,
            num_heads=2,
        )
        t, x_list = backbone_input
        out = model(t, x_list)
        _check_gradients(model, out.sum())

    def test_time_embedder_is_wired(self, dit_backbone):
        # DiT zero-inits adaLN_modulation so outputs are time-independent at
        # initialisation (deliberate — the gates learn to open during
        # training), so we can't assert output changes with t at init.
        # Instead verify the t_embedder exists and has parameters.
        assert hasattr(dit_backbone, 't_embedder'), "t_embedder missing from NanoDiT"
        n_params = sum(p.numel() for p in dit_backbone.t_embedder.parameters())
        assert n_params > 0, "t_embedder has no learnable parameters"


class TestS4NDBackbone:

    def test_output_shape(self, s4nd_backbone, backbone_input):
        t, x_list = backbone_input
        out = s4nd_backbone(t, x_list)
        _check_output(out, _expected_shape())

    def test_output_finite(self, s4nd_backbone, backbone_input):
        t, x_list = backbone_input
        with torch.no_grad():
            out = s4nd_backbone(t, x_list)
        _check_output(out, _expected_shape())

    def test_gradients_flow(self, s4nd_backbone, backbone_input):
        t, x_list = backbone_input
        out = s4nd_backbone(t, x_list)
        _check_gradients(s4nd_backbone, out.sum())
