import os
import sys

import pytest
import torch

# Make `methods`, `utils`, etc. importable regardless of the directory pytest
# is invoked from (mirrors how train.py/eval.py expect `src/` on sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tiny spatial/temporal dims so tests run fast on CPU.
# T frames, C channels per frame, D^3 spatial volume.
_T, _C, _D = 4, 1, 8
_B = 2
_FEATURE = 8  # backbone hidden-channel width


@pytest.fixture(scope="session")
def backbone_input():
    """(t, x_list) matching the shared build_backbone `forward(t, x_list)` contract."""
    z = torch.rand(_B, _T * _C, _D, _D, _D)
    t = torch.rand(_B, 2).clamp(1e-3, 1 - 1e-3)
    return t, [z, None]


@pytest.fixture(scope="session")
def convlstm_backbone():
    from methods.fm_utils.backbones import ConvLSTMBackbone
    return ConvLSTMBackbone(in_channels=_C, feature_size=_FEATURE, num_frames=_T)


@pytest.fixture(scope="session")
def dit_backbone():
    from methods.fm_utils.dit import NanoDiT
    # patch_size must divide _D; use depth=2 and num_heads=2 for speed
    return NanoDiT(
        input_size=[_D, _D, _D],
        patch_size=[2, 2, 2],
        in_channels=_T * _C,
        hidden_size=2,  # NanoDiT multiplies by 8 internally -> 16
        depth=2,
        num_heads=2,
    )


@pytest.fixture(scope="session")
def s4nd_backbone():
    # S4ND isn't ported yet (see TODO_S4ND.md) — this always skips until it is.
    pytest.importorskip(
        "methods.fm_utils.s4nd",
        reason="S4ND backbone not ported yet, see TODO_S4ND.md",
    )
    from methods.fm_utils.backbones import build_backbone
    return build_backbone("s4nd", in_shape=(_T, _C, _D, _D, _D), num_context=_T,
                           feature_size=_FEATURE, num_frames=_T)
