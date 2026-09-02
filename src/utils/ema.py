"""Exponential moving average of model weights.

A shadow copy of the model whose parameters are updated as
shadow = decay * shadow + (1 - decay) * model after every optimizer step.
EMA weights are typically smoother and generalise better than the raw
training weights, especially for diffusion/flow-matching objectives — see
SADM's own `use_ema`/`ema_decay` hparams (this mirrors that convention).
"""

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_p, p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for shadow_b, b in zip(self.shadow.buffers(), model.buffers()):
            shadow_b.copy_(b)

    def state_dict(self):
        return self.shadow.state_dict()
