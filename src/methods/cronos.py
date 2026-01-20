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
from methods.fm_utils.image_cond_unet import ConditionedUNet
import torch
from torchdiffeq import odeint, odeint_adjoint
from torch import nn
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
        patch_size = [int(shape / 4) for shape in in_shape[2:]]
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
        self.mask_time = kwargs.get('mask_time', 1.0)
        self.guidance_scale = kwargs.get('guidance_scale', 3.0)
        self.fm_model_unet_expands = kwargs.get('fm_model_unet_expands', [1, 1, 2, 4])
        # remove that later todo:
        embed = 'non_zero'
        from methods.fm_utils.fm_process_utils import process_fill_empty, process_batch_non_zero #todo: add the others as well!
        if embed == 'grid':
            self.process_batch = interpolate_images
        elif embed == 'gauss':
            self.process_batch = gaussian_smoothing
        elif embed == 'temporal_gauss':
            self.process_batch = temporal_gaussian_smoothing
        elif embed == 'non_zero':
            self.process_batch = process_batch_non_zero
        else:
            self.process_batch = process_fill_empty

        self.hparams = kwargs
        if kwargs.get('unet_type', 'fmu') == 'fmu':
            # from MedVP.utils.image_cond_unet import ConditionedUNet
            import inspect
            def filter_kwargs(cls, kwargs):
                sig = inspect.signature(cls.__init__)
                valid_params = sig.parameters.keys()
                return {k: v for k, v in kwargs.items() if k in valid_params}

            filtered_kwargs = filter_kwargs(ConditionedUNet, kwargs)
            # we use a different amount of input context frames, as we want to couple the time!!

            self.u_net = ConditionedUNet(dim=(in_shape[0],) + in_shape[2:], num_channels=feature_size, num_res_blocks=1,
                                   channel_mult=self.fm_model_unet_expands)#.to(kwargs['device'])

            #self.u_net = ConditionedUNet(channel_mult=kwargs.get('fm_model_unet_expands', [1,1,2,4]),
            #                             dim=(self.num_context, *in_shape[1:]),
            #                             num_res_blocks=1,
            #                             num_channels=feature_size, **filtered_kwargs)#.to(kwargs['device'])
        elif kwargs.get('unet_type', 'diff') == 'dit':
            # from MedVP.dit_util import NanoDiT
            patch_size = [p // 4 for p in patch_size]
            # should probably rename this
            self.u_net = NanoDiT(input_size=list(in_shape[2:]),
                                 patch_size=patch_size, in_channels=self.num_context, hidden_size=feature_size)
        else:
            raise ValueError('unknown context type')
        self.fm = ExactOptimalTransportConditionalFlowMatcher(sigma=self.training_noise)
        self.criterion = nn.MSELoss()

    def forward(self, batch_x, batch_y=None, time_points=None, give_vf=False, **kwargs):
        context_tensor = batch_x['image']
        B,T,C,D,H,W = context_tensor.shape
        if time_points is not None:
            time_points = time_points.to(context_tensor.device)
        target_tensor = batch_y.expand(-1, T, -1, -1, -1, -1)  #.squeeze(2)
        t, xt, ut = self.fm.sample_location_and_conditional_flow(context_tensor,
                                                                 target_tensor)
        if self.scale_time_embed:
            last_context_time, target_time = time_points[:, -2], time_points[:, -1]
            # delta_t = torch.diff(time_points, dim=1)
            t = t.view(-1, 1)  # shape (B, 1)
            scaled_flow_time = (1 - t) * time_points[:, :-1] + t * target_time.view(-1, 1)

            # for
            t = scaled_flow_time
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
        #
        conditioning_context = None
        image_and_condition = {'image': context, 'conditioning': conditioning_context}
        pred_y, predicted_velocity, true_velocity = self(image_and_condition, batch_y, time_points=processed_times)
        # do the contrastive stuff?
        loss = self.criterion(true_velocity.to((self.hparams['device'])), predicted_velocity.to(self.hparams['device']))
        return loss

    def validation_step(self, batch, batch_idx=None, time_points=None):
        # does not actually perform the validation, just does the prediction
        # I think
        t_span = torch.linspace(0, 1, self.n_T).to(batch.device)

        if self.scale_time_embed:
            time_points = time_points.to(batch.device)
            last_context_time, target_time = time_points[:, -2], time_points[:, -1]
            delta_t = target_time - last_context_time
            # scaled_flow_time = t*delta_t + last_context_time
        context, time_points = self.process_batch(batch, time_points=time_points, max_images=self.num_context)
        conditioning = None

        def u_net_wrapper(t, x):
            if self.scale_time_embed:
                last_context_time, target_time = time_points[:, -2], time_points[:, -1]
                # delta_t = torch.diff(time_points, dim=1)
                t = t.view(-1, 1)  # shape (B, 1)
                scaled_flow_time = (1 - t) * time_points[:, :-1] + t * target_time.view(-1, 1)
                # for simplicity
                t = scaled_flow_time
            else:
                t = t.view(-1, 1).repeat(1, context.shape[1])
            return self.u_net(t, [x, conditioning])  # [:,[-1]]

        if self.training_noise > 0:
            # https://github.com/nZhangx/TrajectoryFlowMatching/blob/main/src/model/FM_baseline.py#L431
            current_state = context.squeeze(2)
            mask = (context.squeeze(2) > 0.05).to(torch.float)  #.detach()
            trajectory = []
            dt = t_span[1] - t_span[0]
            for t in t_span:
                drift = u_net_wrapper(t, current_state)
                diffusion = self.training_noise * torch.ones_like(drift)
                noise = torch.randn_like(current_state) * torch.sqrt(dt) * mask
                current_state = current_state + drift * dt + diffusion * noise
                trajectory.append(current_state)
            val_res = trajectory[-1][:, -1]  #.mean(dim=1)

        else:
            with torch.no_grad():
                traj = odeint(u_net_wrapper, context.squeeze(2), t_span, atol=1e-5, rtol=1e-5,
                              adjoint_params=self.u_net.parameters())
                val_res = traj[-1][:, -1]  #.mean(dim=1)

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


