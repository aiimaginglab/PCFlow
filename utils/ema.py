# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import warnings

import torch
from torch.optim import Optimizer


class EMA(Optimizer):
    def __init__(self, opt, ema_decay):
        super().__init__(opt.param_groups, defaults={})
        self.ema_decay = ema_decay
        self.apply_ema = self.ema_decay > 0.0
        self.optimizer = opt
        self.state = opt.state
        self.param_groups = opt.param_groups

    def step(self, *args, **kwargs):
        retval = self.optimizer.step(*args, **kwargs)

        if not self.apply_ema:
            return retval

        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.optimizer.state[p]
                if "ema" not in state:
                    state["ema"] = p.detach().clone()

                ema_v = state["ema"]
                ema_v.mul_(self.ema_decay).add_(p.detach(), alpha=1.0 - self.ema_decay)

        return retval

    def load_state_dict(self, state_dict):
        super(EMA, self).load_state_dict(state_dict)
        self.optimizer.state = self.state
        self.optimizer.param_groups = self.param_groups

    def swap_parameters_with_ema(self, store_params_in_ema):
        """This function swaps parameters with their ema values. It records original parameters in the ema
        parameters, if store_params_in_ema is true."""

        if not self.apply_ema:
            warnings.warn("swap_parameters_with_ema was called when there is no EMA weights.")
            return

        for group in self.optimizer.param_groups:
            for i, p in enumerate(group["params"]):
                if not p.requires_grad:
                    continue
                if p not in self.optimizer.state or "ema" not in self.optimizer.state[p]:
                    continue
                ema = self.optimizer.state[p]["ema"]
                if store_params_in_ema:
                    tmp = p.detach().clone() 
                    p.detach().copy_(ema)
                    ema.copy_(tmp)
                else:
                    p.detach().copy_(ema)
