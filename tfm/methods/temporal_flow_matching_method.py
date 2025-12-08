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
from torchcfm.conditional_flow_matching import *
from fm_utils.unet_wrapper import UNetModelWrapper as UNetModel
import torch
from torchdiffeq import odeint, odeint_adjoint
from torch import nn


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, mask_time=False):
        super().__init__()
        self.model = model
        self.mask_time = mask_time

    def forward(self, t, x, *args, **kwargs):
        if len(t.size()) == 0:
            if isinstance(x, list):
                # since we additionally add the list
                t = t.repeat(x[0].shape[0])[:, None]
            else:
                t = t.repeat(x.shape[0])[:, None]
        if self.mask_time:
            t = torch.zeros(t.shape)
        return self.model(x, t, *args, **kwargs)


class TemporalFlowMatching(nn.Module):
    def __init__(self, in_shape=None, **kwargs):
        '''
        PAPER FREEZE: tfm
        :param network:
        :param in_shape: T,C,H,W,D
        :param kwargs:
        '''
        super().__init__()

        self.pre_seq = 3
        self.aft_seq = 1
        self.type_context = kwargs.get('fm_context', 'du')
        # self.no_context = True
        self.n_T = int(kwargs.get('number_evals', 50))  #50 # 250
        self.training_noise = kwargs.get('training_noise', 0.00)
        # vivit_model = ViVitModule(image_size, patch_size, num_frames)
        feature_size = kwargs.get('feature_size', 256)
        # self.u_net_type = 'fmu'
        self.u_net_type = kwargs.get('unet_type', 'fmu')
        self.fm_model_unet_expands = kwargs.get('fm_model_unet_expands', [1, 1, 2, 4])
        self.criterion = kwargs.get('loss_fn', nn.MSELoss())
        if self.u_net_type == 'fmu':
            self.u_net = UNetModel(dim=(in_shape[0],) + in_shape[2:], num_channels=feature_size, num_res_blocks=1,
                                   channel_mult=self.fm_model_unet_expands)
        else:
            print('choose valid Unet!')

        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=self.training_noise)
        self.node = NeuralODE(self.u_net, solver="dopri5", sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        self.hparams = kwargs
        self.fill_context = kwargs.get('fill_context', 1) > 0  # again, the argparser does not recognize False??
        self.mask_missing = kwargs.get('mask_missing', False)
        self.aggretation_mode = kwargs.get('aggretation_mode', 'mean')
        # swap around later??
        self.noise = 0.00
        self.device = self.hparams['device']

    def fill_missing_frames(self, x):
        '''
        Naively fill missing frames, in order for the model to understand what is happening
        :param x:
        :return:
        '''
        # x: Tensor of shape (B, T, C, H, D, W)
        B, T = x.shape[:2]
        # Determine which frames are valid (not all zeros).
        valid = x.abs().sum(dim=(2, 3, 4, 5)) != 0  # shape (B, T)
        #if not valid.any(dim=1).all():
        #    raise ValueError("Each batch must contain at least one valid frame.")
        idx = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        # Forward fill: for each time step, use the last valid index if available; else -1.
        forward_fill = torch.where(valid, idx, torch.full_like(idx, -1))
        forward_fill = forward_fill.cummax(dim=1)[0]
        # Backward fill: for each time step, use the next valid index if available; else T.
        backward_fill = torch.where(valid, idx, torch.full_like(idx, T))
        backward_fill = torch.flip(torch.flip(backward_fill, dims=[1]).cummin(dim=1)[0], dims=[1])
        # Use forward fill when available; otherwise, use backward fill.
        final_idx = torch.where(forward_fill != -1, forward_fill, backward_fill)
        batch_idx = torch.arange(B, device=x.device).unsqueeze(1).expand(B, T)
        return x[batch_idx, final_idx]

    def forward(self, batch_x, batch_y=None, **kwargs):
        ##################################
        # CFM
        ##################################
        B,T,C,D,H,W = batch_x.shape
        if self.fill_context:
            batch_x = self.fill_missing_frames(batch_x)
        # temporal broadcasting:

        batch_y = batch_y.expand(-1, batch_x.size(1), -1, -1, -1, -1)
        t, xt, ut = self.fm.sample_location_and_conditional_flow(batch_x, batch_y)

        xt = xt.reshape(B,T * C, D, H, W)
        vt = self.u_net(t, xt)
        vt = vt.reshape(B, T, C, D, H, W)
        return batch_x, ut, vt

    def training_step(self, batch, batch_idx):
        # batch_y, batch_x, batch_y_seg, batch_x_seg = batch
        batch_y = batch['target_img'].to(self.device)
        batch_x = batch['context'].to(self.device)
        # optionally if we want to use segmentation masks, not always given
        batch_y_seg = batch['target_seg']
        batch_x_seg = batch['context_seg']
        pred_y, ut, vt = self(batch_x, batch_y)
        # optionally mask out missing frames from loss calculation -> do not use jointly with fill_context
        if self.mask_missing:
            mask = batch_x.sum(dim=(2, 3, 4, 5)) > 1
            mask = mask[(...,) + (None,) * (vt.ndim - mask.ndim)]
            vt, ut = vt * mask, ut * mask
        loss = self.criterion(vt, ut)
        # loss += self.criterion(vt.to(device)[:, -1], ut.to(device)[:, -1]) * self.scale_last_flow
        return loss

    def validation_step(self, batch, batch_idx=None, time_points=None):
        # does not actually perform the validation, just does the prediction
        # misnomer
        t_span = torch.linspace(0, 1, self.n_T).to(batch.device)
        if self.fill_context:
            batch = self.fill_missing_frames(batch)
        traj = odeint(self.u_net, batch.squeeze(2), t_span, atol=1e-5, rtol=1e-5,
                      adjoint_params=self.u_net.parameters())
        # canonically we only return the final time point
        val_res = traj[-1]

        # ablation: validate on missing frames instead of filled frames
        if self.mask_missing:
            mask = batch.sum(dim=(2, 3, 4, 5)) > 1
            # check where we have valid frames
            mask = mask[(...,) + (None,) * (val_res.ndim - mask.ndim)]
            # should be bigger than 1 during pre-processing anyway, but just in case
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            return (val_res * mask).sum(dim=1, keepdim=True) / (denom + 1e-8)

        if self.aggretation_mode == 'mean':
            return val_res.mean(dim=1, keepdim=True)
        elif self.aggretation_mode == 'last':
            return val_res[:, [-1]]
        else:
            return val_res.mean(dim=1, keepdim=True)


if __name__ == "__main__":
    model = TemporalFlowMatching(
        in_shape=(3, 1, 16, 16, 16),
        device='gpu',
        feature_size=8,
        fm_model_unet_expands=[1, 1, 1, 1],  # keep it shallow
    )
    x = torch.randn(1, 3, 1, 16, 16, 16)
    y = torch.randn(1, 3, 1, 16, 16, 16)
    pred_y, ut, vt = model(x, y)
    print(pred_y.shape, ut.shape, vt.shape)
