import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm.autonotebook import tqdm
from functools import partial

# from models.UNetModel import UNetModel
from models.BBM_utils import extract, default
from tools.Register import Registers
from .model import *
# from .latent_space.autoencoder import AutoencoderKL
import cv2

# from .latent_space.util import disabled_train, instantiate_from_config
# from .latent_space.distributions import DiagonalGaussianDistribution

import matplotlib.pyplot as plt





@Registers.runners.register_with_name('BrownianBridge')
class BrownianBridgeModel(nn.Module):
    """
    The full code will be released to the public at a later time.
    """
    pass
    
@Registers.runners.register_with_name('RemoveBBM')
class RemoveBrownianBridgeModel(nn.Module):
    def __init__(self, model_config, diff_config, view_only=False):
        super().__init__()
        self.loss_type = diff_config.loss_type

        self.denoise_fn = Registers.runners[model_config.name](model_config)
        self.device = model_config.device

        if self.loss_type == 'BCE':
            self.BCELoss = torch.nn.BCELoss()
    
    def forward(self, objective, y, context):
        '''
        y: x_T;
        content: maybe multi-channel compared to y
        '''
        objective_recon = self.denoise_fn(context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        elif self.loss_type == 'focal':
            # print(torch.max(objective_recon), torch.min(objective_recon), torch.max(objective), torch.min(objective))
            recloss = self.FocalLoss(torch.sigmoid(objective_recon), torch.sigmoid(objective))
        elif self.loss_type == 'BCE':
            recloss = self.BCELoss(torch.sigmoid(objective_recon), torch.sigmoid(objective)) # 不加sigmoid无法训练
        else:
            raise NotImplementedError()
        
        return recloss
    
    @torch.no_grad()
    def sample(self, y, context, type=None):
        return self.denoise_fn(context).cpu().detach()


@Registers.runners.register_with_name('Diffusion')
class GaussianDiffusion(nn.Module):
    def __init__(
        self, model_config, diff_config, view_only=False
        # self,
        # model,
        # img_size,
        # img_channels,
        # betas,
        # device
    ):
        super().__init__()

        # self.model = model
        # self.ema_model = deepcopy(model)

        # self.ema = EMA(ema_decay)
        self.skip_sample = diff_config.skip_sample
        self.sample_type = diff_config.beta_schedule
        self.loss_type = diff_config.loss_type
        self.objective = diff_config.objective
        self.sample_type = diff_config.beta_schedule
        self.sample_step = diff_config.sample_step
        if not view_only:
            self.denoise_fn = Registers.runners[model_config.name](model_config) # UNetModel(**UNetParams)
            self.device = model_config.device 

        # self.img_size = img_size
        # self.img_channels = img_channels
        # self.device = device
        self.num_timesteps = diff_config.timesteps
        self.steps = None
        self.register_schedule()

    def register_schedule(self):
        T = self.num_timesteps

        betas = torch.linspace(0.0001, 0.02, T)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))

        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    @torch.no_grad()
    def remove_noise(self, x, y, t):

        return (
            (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, y, t)) *
            extract(self.reciprocal_sqrt_alphas, t, x.shape)
        )

    @torch.no_grad()
    def sample(self, y, context, clip_denoised=True, sample_mid_step=False, type='default'):
        x_t = torch.randn_like(y)  # TODO: device
        for i in range(len(self.steps)):  # 这里的step是测试的step
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            if self.objective == 'grad':
                x0_recon = x_t - objective_recon
            elif self.objective == 'x0':
                x0_recon = objective_recon
            else:
                raise NotImplementedError()
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            x_t = self.perturb_x(x0_recon, t, torch.randn_like(x_t))


        # for t in range(self.num_timesteps - 1, -1, -1):
        #     t_batch = torch.tensor([t], device=self.device).repeat(y.shape[0])
        #     x = self.remove_noise(x, y, t_batch)
        #     if t > 0:
        #         x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)        
        return x_t.cpu().detach()
              
 
    def perturb_x(self, x, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )

    def q_sample(self, x0, y, t, noise):
        perturbed_x = self.perturb_x(x0, t, noise)
        if self.objective == 'grad':
            objective = perturbed_x - x0
        # elif self.objective == 'ysubx':
        #     objective = noise - x0  # 这个应该不行吧
        elif self.objective == 'x0':
            objective = x0
        else:
            raise NotImplementedError()
        return perturbed_x, objective

    def get_losses(self, x0, y, context, t):
        noise = torch.randn_like(x0)

        # perturbed_x = self.perturb_x(x, t, noise) # TODO:
        x_t, objective = self.q_sample(x0, y, t, noise)
        objective_recon = self.denoise_fn(x_t, t=t, y=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()
        return recloss

    def forward(self, x, y, context):
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x.device)
        return self.get_losses(x, y, context, t)


    def view_q_sample(self, x0, y, path, name):
        noise = torch.randn_like(x0)
        for t in self.steps:
            t_batch = torch.tensor([t])
            # print(x0.shape, t.shape, y.shape)
            # exit()
            x_t = self.perturb_x(x0, t, noise)[0][0]
            # print(torch.max(x_t), torch.min(x_t)) # tensor(1.0358) tensor(-1.1701)
            # exit()
            x_t = (x_t.numpy().clip(-1, 1)+1)/2
            x_t = np.uint8(255 * x_t)
            # perturbed_x = np.broadcast_to(perturbed_x[:,:, np.newaxis],(128,128,3))
            cv2.imwrite(f"{path}/{name}_{t}.png", x_t)
            # exit()

    def view_q_sample_colors(self, x0, y, path, name):
        noise = torch.randn_like(x0)
        for t in self.steps:
            t_batch = torch.tensor([t])
            # print(x0.shape, t.shape, y.shape)
            # exit()
            x_t = self.perturb_x(x0, t_batch, noise)[0][0]
            # print(torch.max(x_t), torch.min(x_t)) # tensor(1.0358) tensor(-1.1701)
            # exit()
            x_t = (x_t.numpy().clip(-1, 1)+1)/2
            x_t = np.uint8(255 * x_t)
            # perturbed_x = np.broadcast_to(perturbed_x[:,:, np.newaxis],(128,128,3))
            plt.imsave(f"{path}/{name}_{t}.png", x_t, cmap=plt.get_cmap('bwr'))
            plt.clf()
            # exit()