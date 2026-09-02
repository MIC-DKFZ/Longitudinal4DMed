import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint
from torchsde import sdeint
from torchdyn.models import NeuralODE
from torchcfm.conditional_flow_matching import (
    ExactOptimalTransportConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from .fm_utils.fm_process_utils import compute_roi_term
from .fm_utils.backbones import build_backbone
# just for saving weights easily
import os



class CRONOS(nn.Module):
    def __init__(self, network=None, in_shape=None, **kwargs):
        '''
        PAPER FREEZE
        :param network:
        :param in_shape:
        :param kwargs:
        '''
        super(CRONOS, self).__init__()
        self.n_T = int(kwargs.get('number_evals', 50))  # 50 # 250
        feature_size = kwargs.get('feature_size', 256)
        self.device = kwargs.get('device', 'cpu')
        self.num_context = kwargs.get('num_context', in_shape[0])  # 4
        self.use_guidance = kwargs.get('train_with_guidance', False)
        self.regularisation_loss = kwargs.get('regularisation_loss', True)
        self.use_pre_trained = kwargs.get('use_pre_trained', False)
        self.scale_time_embed = True  # kwargs.get('scale_time_real', False)
        self.train_multiple_time_steps = kwargs.get('train_multiple_time_steps', False)
        self.training_noise = kwargs.get('training_noise', 0.01)
        self.use_all_context = True  # kwargs.get('use_all_contex', True)
        self.lamba_size_reg = kwargs.get('regularisation_loss_weight_size', 0.1)
        self.lambda_smooth_reg = kwargs.get('regularisation_loss_weight_smooth', 0.1)
        self.contrastive = kwargs.get('contrastive_simsiam_lambda', 0.0) > 0
        self.simsiam_lambda = kwargs.get('contrastive_simsiam_lambda', 0.1)
        self.reconstruction_threshold = kwargs.get('reconstruction_threshold', 0.001)
        self.diffusion_mode = kwargs.get('diffusion_mode', False)
        # mask_time>=0.5 zeroes t before the backbone sees it (image_cond_unet.py).
        # Was 1.0 (time-blind) by default for all modes; now 0.0 (time-aware).
        # TODO: review -- new default, overfitting behavior not yet characterized.
        self.mask_time = kwargs.get('mask_time', 0.0)
        self.guidance_scale = kwargs.get('guidance_scale', 3.0)
        self.fm_model_unet_expands = kwargs.get('fm_model_unet_expands', [1, 1, 2, 4])
        self.random_inference = kwargs.get('random_inference', False)
        # zeroes SDE noise outside context>0.05 (background); from
        # nZhangx/TrajectoryFlowMatching FM_baseline.py#L431.
        # TODO: review -- 0.05 threshold is dataset-specific, untested elsewhere.
        self.mask_noise_by_foreground = kwargs.get('mask_noise_by_foreground', True)
        # remove that later todo:
        embed = 'non_zero'
        from methods.fm_utils.fm_process_utils import process_fill_empty, process_batch_non_zero #todo: add the others as well!
        if embed == 'grid':
            self.process_batch = interpolate_images # todo:
        elif embed == 'gauss':
            self.process_batch = gaussian_smoothing
        elif embed == 'temporal_gauss':
            self.process_batch = temporal_gaussian_smoothing
        elif embed == 'non_zero':
            self.process_batch = process_batch_non_zero
        else:
            self.process_batch = process_fill_empty

        self.hparams = kwargs
        # unet_type selects the backbone behind the shared build_backbone()
        # `forward(t, x_list)` contract. CRONOS's own historical default,
        # 'fmu', means ConditionedUNet ('cond_unet' in build_backbone's
        # canonical naming) — translate here, everything else passes through.
        _unet_type = kwargs.get('unet_type', 'fmu')
        _unet_type = 'cond_unet' if _unet_type == 'fmu' else _unet_type
        _reserved = {'feature_size', 'fm_model_unet_expands', 'unet_type', 'num_frames', 'in_shape', 'num_context'}
        self.u_net = build_backbone(
            _unet_type, in_shape=in_shape, num_context=self.num_context, feature_size=feature_size,
            fm_model_unet_expands=self.fm_model_unet_expands, num_frames=self.num_context,
            **{k: v for k, v in kwargs.items() if k not in _reserved},
        )
        # diffusion_mode (detected above) swaps the OT-CFM straight-line interpolant
        # for torchcfm's VariancePreservingConditionalFlowMatcher -- the trigonometric
        # ("stochastic interpolant") path x_t = cos(pi/2 t) x0 + sin(pi/2 t) x1, which
        # is the probability-flow-ODE equivalent of a variance-preserving diffusion
        # process (Albergo et al.). Same sample_location_and_conditional_flow(x0, x1)
        # interface, so nothing else in training_step/validation_step needs to change.
        # Experimental / not tuned for quality -- just a different training objective.
        if self.diffusion_mode:
            self.fm = VariancePreservingConditionalFlowMatcher(sigma=self.training_noise)
        else:
            self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=self.training_noise)
        self.criterion = nn.MSELoss()
        self.lambda_roi_seg = kwargs.get('lambda_roi_seg', 0.0)
        self.roi_dilation = kwargs.get('roi_dilation', 0)

    def forward(self, batch_x, batch_y=None, time_points=None, give_vf=False, **kwargs):
        context_tensor = batch_x['image']
        B,T,C,D,H,W = context_tensor.shape
        if time_points is not None:
            time_points = time_points.to(context_tensor.device)
        target_tensor = batch_y.expand(-1, T, -1, -1, -1, -1)  #.squeeze(2)
        t, xt, ut = self.fm.sample_location_and_conditional_flow(context_tensor,
                                                                 target_tensor)
        if self.scale_time_embed:
            t = t.view(-1, 1)
            if self.diffusion_mode:
                # ut above depends on this exact raw t; scaled_flow_time (real
                # acquisition-time blend) would feed the backbone a different
                # quantity than ut was computed from. TODO: review.
                t = t.expand(B, T)
            else:
                target_time = time_points[:, -1]
                t = (1 - t) * time_points[:, :-1] + t * target_time.view(-1, 1)
        else:
            t = t.view(-1, 1).repeat(1, context_tensor.shape[1])

        xt = xt.reshape(B, T * C, D, H, W)
        xt = [xt, batch_x['conditioning']]
        vt = self.u_net(t, xt)
        vt = vt.reshape(B, T, C, D, H, W)
        vt_masked = vt # if we want additinal operations
        ut_masked = ut

        if give_vf:
            return batch_x, ut_masked, vt_masked, xt, t
        return batch_x, ut_masked, vt_masked

    def training_step(self, batch, batch_idx):
        batch_y = batch['target_img']
        batch_x = batch['context']
        batch_y_seg = batch['target_seg']
        batch_x_seg = batch['context_seg']
        target_time = batch['target_time']
        context_time = batch['context_time']
        batch_x = batch_x.to(self.hparams['device'])
        batch_y = batch_y.to(self.hparams['device'])
        time_vec = torch.concat([context_time, target_time], dim=1)
        context, processed_times = self.process_batch(batch_x, batch_y=batch_y, time_points=time_vec, max_images=self.num_context)
        processed_times = torch.cat([processed_times, target_time.to(processed_times.device)], dim=1)
        conditioning_context = None
        image_and_condition = {'image': context, 'conditioning': conditioning_context}
        pred_y, predicted_velocity, true_velocity = self(image_and_condition, batch_y, time_points=processed_times)
        # do the contrastive stuff?
        true_velocity = true_velocity.to(self.hparams['device'])
        predicted_velocity = predicted_velocity.to(self.hparams['device'])
        if self.lambda_roi_seg > 0:
            per_voxel_loss = F.mse_loss(predicted_velocity, true_velocity, reduction='none')
            loss = per_voxel_loss.mean() + self.lambda_roi_seg * compute_roi_term(
                per_voxel_loss, batch_y_seg, roi_dilation=self.roi_dilation)
        else:
            loss = self.criterion(true_velocity, predicted_velocity)
        return loss

    def validation_step(self, batch, batch_idx=None, time_points=None):
        # does not actually perform the validation, just does the prediction
        # I think
        t_span = torch.linspace(0, 1, self.n_T).to(batch.device)

        if self.scale_time_embed:
            time_points = time_points.to(batch.device)
        actual_target_time = time_points[:, -1:]
        context, time_points = self.process_batch(batch, time_points=time_points, max_images=self.num_context)
        time_points = torch.cat([time_points, actual_target_time.to(time_points.device)], dim=1)
        conditioning = None

        def u_net_wrapper(t, x):
            if self.scale_time_embed:
                t = t.view(-1, 1)
                if self.diffusion_mode:
                    # must match forward()'s time convention -- see its comment.
                    t = t.expand(context.shape[0], context.shape[1])
                else:
                    target_time = time_points[:, -1]
                    t = (1 - t) * time_points[:, :-1] + t * target_time.view(-1, 1)
            else:
                t = t.view(-1, 1).repeat(1, context.shape[1])
            return self.u_net(t, [x, conditioning])  # [:,[-1]]

        if self.training_noise and self.random_inference > 0:
            # https://github.com/nZhangx/TrajectoryFlowMatching/blob/main/src/model/FM_baseline.py#L431
            # no_grad: unrolled loop otherwise retains a graph across every
            # step and OOMs at large n_T. Only final state used.
            with torch.no_grad():
                current_state = context.squeeze(2)
                if self.mask_noise_by_foreground:
                    mask = (context.squeeze(2) > 0.05).to(torch.float)  #.detach()
                else:
                    mask = torch.ones_like(context.squeeze(2))
                dt = t_span[1] - t_span[0]
                for t in t_span:
                    drift = u_net_wrapper(t, current_state)
                    diffusion = self.training_noise * torch.ones_like(drift)
                    noise = torch.randn_like(current_state) * torch.sqrt(dt) * mask
                    current_state = current_state + drift * dt + diffusion * noise
                val_res = current_state[:, -1]  #.mean(dim=1)

        else:
            with torch.no_grad():
                traj = odeint(u_net_wrapper, context.squeeze(2), t_span, atol=1e-5, rtol=1e-5,
                              adjoint_params=self.u_net.parameters())
                val_res = traj[-1].mean(dim=1) #[:, -1]  todo: mean or last into settings

        return val_res





if __name__ == "__main__":
    model = CRONOS(
        in_shape=(3, 1, 16, 16, 16),
        device='gpu',
        feature_size=8,
        fm_model_unet_expands=[1, 1, 1, 1],  # keep it shallow
    )
    x = torch.randn(1, 3, 1, 16, 16, 16)
    y = torch.randn(1, 3, 1, 16, 16, 16)
    pred_y, ut, vt = model({'image': x}, y, time_points=torch.tensor([[0.0, 0.5, 1.0]]))
    print(pred_y.shape, ut.shape, vt.shape)


