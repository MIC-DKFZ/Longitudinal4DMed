"""CRONOSFlex tests: multi-modal (C > 1) shapes, frame masking, per-frame time, aggregation, ResEnc encoder."""
import os

import pytest
import torch

from tests.conftest import _B, _D, _T

_C = 3
_IN_SHAPE = (_T, _C, _D, _D, _D)


def _model(**kwargs):
    from methods.cronos_flex import CRONOSFlex
    return CRONOSFlex(in_shape=_IN_SHAPE, feature_size=4, fm_model_unet_expands=[1, 2],
                      device='cpu', number_evals=3, **kwargs)


def _batch():
    return {
        'target_img': torch.rand(_B, 1, _C, _D, _D, _D),
        'context': torch.rand(_B, *_IN_SHAPE),
        'target_seg': (torch.rand(_B, 1, 1, _D, _D, _D) > 0.5).float(),
        'context_time': torch.rand(_B, _T),
        'target_time': torch.ones(_B, 1),
    }


@pytest.mark.parametrize("cross_frame", ["none", "attn_deepest", "attn_all"])
@pytest.mark.parametrize("random_inference", [False, True])
def test_train_and_validate(cross_frame, random_inference):
    model = _model(cross_frame=cross_frame, random_inference=random_inference, lambda_roi_seg=1.0)
    loss = model.training_step(_batch(), 0)
    assert torch.isfinite(loss)
    loss.backward()
    time_points = torch.linspace(0, 1, _T + 1).unsqueeze(0).expand(_B, -1)
    out = model.validation_step(torch.rand(_B, *_IN_SHAPE), time_points=time_points)
    assert out.shape == (_B, _C, _D, _D, _D)
    assert torch.isfinite(out).all()


def test_backbone_keeps_time_and_modalities_separate():
    from methods.fm_utils.per_frame_unet import PerFrameUNet
    net = PerFrameUNet(in_channels=_C, out_channels=_C, spatial_shape=(_D,) * 3, feature_size=4,
                       channel_mult=[1, 2], cross_frame='none')
    torch.nn.init.normal_(net.out[-1].weight)  # zero-init output would hide any coupling
    z = torch.rand(_B, _T, _C, _D, _D, _D)
    t = torch.rand(_B, _T)
    out = net(t, [z, None])
    assert out.shape == z.shape
    # without cross-frame attention, perturbing frame 0 must not change frame 1
    z2 = z.clone()
    z2[:, 0] += 1.0
    assert torch.allclose(net(t, [z2, None])[:, 1], out[:, 1])


def test_padded_frames_ignored_by_attention():
    from methods.fm_utils.per_frame_unet import TimeBiasedFrameAttention
    attn = TimeBiasedFrameAttention(8)
    torch.nn.init.normal_(attn.out.weight)
    h = torch.rand(_B * _T, 8, 2, 2, 2)
    t = torch.rand(_B, _T)
    mask = torch.ones(_B, _T, dtype=torch.bool)
    mask[:, -1] = False
    out = attn(h, t, _B, _T, mask)
    h2 = h.clone().view(_B, _T, 8, 2, 2, 2)
    h2[:, -1] += 5.0
    out2 = attn(h2.view(_B * _T, 8, 2, 2, 2), t, _B, _T, mask).view(_B, _T, 8, 2, 2, 2)
    assert torch.allclose(out2[:, :-1], out.view(_B, _T, 8, 2, 2, 2)[:, :-1], atol=1e-6)


def test_aggregate_last_uses_last_real_frame():
    model = _model(aggregate='last')
    states = torch.arange(_T, dtype=torch.float).view(1, _T, 1, 1, 1, 1).expand(_B, _T, _C, 1, 1, 1)
    mask = torch.ones(_B, _T, dtype=torch.bool)
    mask[0, 2:] = False
    out = model._aggregate(states, mask, stochastic=False)
    assert out[0].eq(1).all() and out[1].eq(_T - 1).all()
    model.aggregate = 'mean'
    out = model._aggregate(states, mask, stochastic=False)
    assert out[0].eq(0.5).all()


def _randomized_net(cross_frame, C=_C, **kwargs):
    """PerFrameUNet with its zero-inits (FiLM, attention out, output conv) undone, so couplings show."""
    from methods.fm_utils.per_frame_unet import PerFrameUNet, FiLM, TimeBiasedFrameAttention
    net = PerFrameUNet(in_channels=C, out_channels=C, spatial_shape=(_D,) * 3, feature_size=4,
                       channel_mult=[1, 2], cross_frame=cross_frame, **kwargs)
    for m in net.modules():
        if isinstance(m, FiLM):
            torch.nn.init.normal_(m.lin.weight, std=0.1)
        if isinstance(m, TimeBiasedFrameAttention):
            torch.nn.init.normal_(m.out.weight, std=0.1)
    torch.nn.init.normal_(net.out[-1].weight, std=0.1)
    return net.eval()


@pytest.mark.parametrize("cross_frame", ["none", "attn_deepest", "attn_all"])
def test_samples_in_batch_independent(cross_frame):
    net = _randomized_net(cross_frame)
    z, t = torch.rand(_B, _T, _C, _D, _D, _D), torch.rand(_B, _T)
    with torch.no_grad():
        out = net(t, [z, None])
        z[1] += 1.0
        t[1] += 5.0
        assert torch.allclose(net(t, [z, None])[0], out[0], atol=1e-6)


def test_frame_time_reaches_only_its_frame_without_attention():
    net = _randomized_net('none')
    z, t = torch.rand(_B, _T, _C, _D, _D, _D), torch.rand(_B, _T)
    with torch.no_grad():
        out = net(t, [z, None])
        t[:, 0] += 0.5
        out2 = net(t, [z, None])
    assert not torch.allclose(out2[:, 0], out[:, 0], atol=1e-4)
    assert torch.allclose(out2[:, 1:], out[:, 1:])


def test_padded_frame_invisible_end_to_end():
    net = _randomized_net('attn_all')
    z, t = torch.rand(_B, _T, _C, _D, _D, _D), torch.rand(_B, _T)
    mask = torch.ones(_B, _T, dtype=torch.bool)
    mask[0, -1] = False
    with torch.no_grad():
        out = net(t, [z, None], frame_mask=mask)
        z[0, -1] = torch.rand_like(z[0, -1]) * 10
        t[0, -1] = -7.0
        out2 = net(t, [z, None], frame_mask=mask)
    assert torch.allclose(out2[0, :-1], out[0, :-1], atol=1e-5)


def test_frame_times_interpolate_context_to_target():
    model = _model()
    tp = torch.tensor([[0., 1., 2., 3., 5.], [0., 2., 3., 4., 6.]])
    assert torch.allclose(model._frame_times(torch.zeros(_B), tp), tp[:, :-1])
    assert torch.allclose(model._frame_times(torch.ones(_B), tp), tp[:, -1:].expand(_B, _T))


def test_empty_context_frame_masked():
    model = _model()
    ctx = torch.rand(_B, *_IN_SHAPE)
    ctx[0, -1] = 0
    _, _, mask = model._prep_context(ctx, torch.rand(_B, _T + 1))
    assert mask[0].tolist() == [True] * (_T - 1) + [False] and mask[1].all()


def test_roi_term_multimodal_ignores_padded_frames():
    from methods.fm_utils.fm_process_utils import compute_roi_term
    seg = (torch.rand(_B, 1, 1, _D, _D, _D) > 0.5).float()
    loss = torch.rand(_B, _T, _C, _D, _D, _D)
    valid = torch.ones(_B, _T, 1, 1, 1, 1, dtype=torch.bool)
    valid[0, -1] = False
    loss[0, -1] = 100.0  # a padded frame must not leak into the term
    flat = lambda x: x.expand(_B, _T, _C, _D, _D, _D).reshape(_B, _T * _C, _D, _D, _D)
    roi = compute_roi_term(flat(loss), seg, valid=flat(valid))
    keep = flat(valid) & (seg.view(_B, 1, _D, _D, _D) > 0.5)
    assert roi > 0 and torch.allclose(roi, flat(loss)[keep].mean())


def test_all_parameters_get_gradient():
    # zero-init output conv, then zero-init FiLM, delay gradient to the time embedding until step 3
    model = _model(cross_frame='attn_all', lambda_roi_seg=1.0)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(3):
        opt.zero_grad()
        model.training_step(_batch(), 0).backward()
        opt.step()
    dead = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum()]
    assert not dead


def test_overfit_beats_last_context_image():
    from methods.cronos_flex import CRONOSFlex
    torch.manual_seed(0)
    T, C, D = 3, 2, 8
    g = torch.linspace(-1, 1, D)
    r = torch.stack(torch.meshgrid(g, g, g, indexing='ij')).norm(dim=0)
    blob = lambda rad: torch.stack([torch.sigmoid((rad - r) * 12), 0.5 * torch.sigmoid((rad - r) * 12) + 0.2])
    ctx = torch.stack([blob(0.3 + 0.15 * i) for i in range(T)]).unsqueeze(0)  # growing sphere
    tgt = blob(0.3 + 0.15 * T)[None, None]
    ctime, ttime = torch.arange(T, dtype=torch.float)[None], torch.tensor([[float(T)]])
    batch = dict(context=ctx, target_img=tgt, target_seg=None, context_time=ctime, target_time=ttime)
    model = CRONOSFlex(in_shape=(T, C, D, D, D), feature_size=8, fm_model_unet_expands=[1, 2], device='cpu',
                       number_evals=10, training_noise=0.0)
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    for step in range(200):
        opt.zero_grad()
        model.training_step(batch, step).backward()
        opt.step()
    pred = model.eval().validation_step(ctx, time_points=torch.cat([ctime, ttime], 1))
    lci = ((ctx[:, -1] - tgt[:, 0]) ** 2).mean()
    assert ((pred - tgt[:, 0]) ** 2).mean() < lci / 5


# ---- ResEnc frame encoder ------------------------------------------------------

_RESENC_SHAPE = (32, 32, 32)  # too small for 5 isotropic halvings, so per-axis strides with stride-1 stages


def _resenc_flex(C, ckpt='scratch', **kwargs):
    pytest.importorskip('dynamic_network_architectures')
    from methods.cronos_flex import CRONOSFlex
    return CRONOSFlex(in_shape=(2, C, *_RESENC_SHAPE), frame_encoder='resenc', frame_encoder_ckpt=ckpt,
                      device='cpu', number_evals=2, **kwargs)


@pytest.mark.parametrize("C", [1, 3])
def test_resenc_train_and_validate(C):
    model = _resenc_flex(C, lambda_roi_seg=1.0)
    batch = {'context': torch.rand(1, 2, C, *_RESENC_SHAPE), 'target_img': torch.rand(1, 1, C, *_RESENC_SHAPE),
             'target_seg': (torch.rand(1, 1, 1, *_RESENC_SHAPE) > 0.5).float(),
             'context_time': torch.tensor([[0., 1.]]), 'target_time': torch.tensor([[2.]])}
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    loss.backward()
    out = model.eval().validation_step(batch['context'], time_points=torch.tensor([[0., 1., 2.]]))
    assert out.shape == (1, C, *_RESENC_SHAPE) and torch.isfinite(out).all()


def test_resenc_sees_frame_intensity():
    # InstanceNorm in the stem is blind to x -> a x + b per frame, the raw skip must restore it
    enc = _resenc_flex(1).u_net.encoder.eval()
    x = torch.rand(2, 1, *_RESENC_SHAPE)
    with torch.no_grad():
        a, b = enc(x)[0], enc(1.5 * x + 0.2)[0]
    assert (a - b).norm() / a.norm() > 1e-2


def test_resenc_requires_explicit_ckpt():
    with pytest.raises(ValueError, match='frame_encoder_ckpt'):
        _resenc_flex(1, ckpt=None)


@pytest.mark.skipif(not os.environ.get('RESENC_CKPT'), reason='set RESENC_CKPT to an nnssl ResEnc-L checkpoint')
def test_resenc_pretrained_loads_strictly():
    scratch = _resenc_flex(1).u_net.encoder.enc.state_dict()
    loaded = _resenc_flex(1, ckpt=os.environ['RESENC_CKPT']).u_net.encoder.enc.state_dict()
    assert scratch.keys() == loaded.keys()
    assert not torch.allclose(scratch['stages.3.blocks.0.conv1.conv.weight'], loaded['stages.3.blocks.0.conv1.conv.weight'])


def test_param_groups_single_without_pretrained_encoder():
    import argparse
    from train import build_param_groups
    args = argparse.Namespace(lr=1e-4)
    for model in (_model(), _resenc_flex(1)):  # conv and scratch resenc: nothing pretrained
        groups = build_param_groups(model, args)
        assert len(groups) == 1 and len(groups[0]['params']) == len(list(model.parameters()))


@pytest.mark.skipif(not os.environ.get('RESENC_CKPT'), reason='set RESENC_CKPT to an nnssl ResEnc-L checkpoint')
def test_param_groups_scale_pretrained_encoder_lr():
    import argparse
    from train import build_param_groups
    model = _resenc_flex(1, ckpt=os.environ['RESENC_CKPT'])
    groups = build_param_groups(model, argparse.Namespace(lr=1e-4, encoder_lr_scale=0.1))
    enc_ids = {id(p) for p in model.u_net.encoder.enc.parameters()}
    assert len(groups) == 2 and groups[1]['lr'] == pytest.approx(1e-5)
    assert {id(p) for p in groups[1]['params']} == enc_ids
    raw = {id(p) for p in model.u_net.encoder.raw.parameters()}
    assert raw <= {id(p) for p in groups[0]['params']}  # the fresh raw-intensity conv trains at full lr
    assert sum(len(g['params']) for g in groups) == len(list(model.parameters()))


def test_all_empty_context_sample_keeps_batch_shape():
    from methods.fm_utils.fm_process_utils import process_batch_non_zero_masked
    x = torch.rand(_B, _T, 1, _D, _D, _D)
    x[1] = 0
    images, times, mask = process_batch_non_zero_masked(x, time_points=torch.rand(_B, _T + 1), max_images=_T)
    assert images.shape == x.shape and times.shape == (_B, _T)
    assert mask[0].all() and mask[1].tolist() == [True] + [False] * (_T - 1)
