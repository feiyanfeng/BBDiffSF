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
    def __init__(self, model_config, diff_config, view_only=False):
        super().__init__()
        self.num_timesteps = diff_config.timesteps
        self.mt_type = diff_config.beta_schedule
        self.max_var = diff_config.max_var if diff_config.__contains__("max_var") else 1
        self.var_scale = diff_config.var_scale
        self.eta = diff_config.eta if diff_config.__contains__("eta") else 1
        self.skip_sample = diff_config.skip_sample
        self.sample_type = diff_config.beta_schedule
        self.sample_step = diff_config.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = diff_config.loss_type
        self.objective = diff_config.objective
        self.BCE = torch.nn.BCELoss()

        if not view_only:
            self.denoise_fn = Registers.runners[model_config.name](model_config) # UNetModel(**UNetParams)
            self.device = model_config.device 


    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * self.var_scale * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

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



    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        elif self.objective == 'x0':
            objective = x0
        elif self.objective == 'xt-1':
            m_t_1 = extract(self.m_t, t-1, x0.shape)
            var_t_1 = extract(self.variance_t, t-1, x0.shape)
            sigma_t_1 = torch.sqrt(var_t_1)
            objective = (1. - m_t_1) * x0 + m_t_1 * y + sigma_t_1 * noise
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def view_q_sample(self, x0, y, path, name):
        noise = torch.randn_like(x0)
        for t in self.steps:
            t_batch = torch.tensor([t])
            x_t, _ = self.q_sample(x0, y, t_batch, noise)
            x_t = (x_t.numpy().clip(-1, 1)+1)/2
            x_t = np.uint8(255 * x_t[0][0])
            # perturbed_x = np.broadcast_to(perturbed_x[:,:, np.newaxis],(128,128,3))
            cv2.imwrite(f"{path}/{name}_{t}.png", x_t)
            # exit()

    def view_q_sample_colors(self, x0, y, path, name):
        noise = torch.randn_like(x0)
        for t in self.steps:
            t_batch = torch.tensor([t])
            x_t, _ = self.q_sample(x0, y, t_batch, noise)
            x_t = (x_t.numpy().clip(-1, 1)+1)/2
            x_t = np.uint8(255 * x_t[0][0])
            # perturbed_x = np.broadcast_to(perturbed_x[:,:, np.newaxis],(128,128,3))
            plt.imsave(f"{path}/{name}_{t}.png", x_t, cmap=plt.get_cmap('bwr'))
            plt.clf()
            # exit()

    def p_losses(self, x0, y, context, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, c, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        x_t, objective = self.q_sample(x0, y, t, noise)
        # print(x_t.shape, t, y.shape)
        objective_recon = self.denoise_fn(x_t, t=t, y=context)

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        elif self.loss_type == 'focal':
            # print(torch.max(objective_recon), torch.min(objective_recon), torch.max(objective), torch.min(objective))
            recloss = self.FocalLoss(torch.sigmoid(objective_recon), torch.sigmoid(objective))
        else:
            raise NotImplementedError()

        # x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)
        # log_dict = {
        #     "loss": recloss,
        #     "x0_recon": x0_recon
        # }
        return recloss # , log_dict


    def forward(self, x, y, context):
        '''
        y: x_T;
        content: maybe multi-channel compared to y
        '''
        b = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        return self.p_losses(x, y, context, t)



    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)  # TODO: variance_t
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        elif self.objective == 'x0':
            x0_recon = objective_recon
        else: # xt-1 不需要到这
            raise NotImplementedError
        return x0_recon



    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

 
    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        # if self.condition_key == "nocond":
        #     context = None
        # else:
        context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            # for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
            for i in range(len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            # for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):  # 这里的step是测试的step
            for i in range(len(self.steps)):  # 这里的step是测试的step
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img


    @torch.no_grad()
    def sample(self, y, context, clip_denoised=True, sample_mid_step=False, type='default'):
        if type=='default':
            return self.p_sample_loop(y, context, clip_denoised, sample_mid_step).cpu().detach()
        elif type=='x0': # 测试的x0和训练的x0不是一个概念。训练的x0应该属于default
            x_t = y
            t = torch.full((x_t.shape[0],), self.steps[0], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon.cpu().detach()
        elif type=='x0_to_xt-1':
            x_t=y
            for i in range(len(self.steps)-1):  # 这里的step是测试的step
                t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
                objective_recon = self.denoise_fn(x_t, t=t, y=context)
                x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
                if clip_denoised:
                    x0_recon.clamp_(-1., 1.)
                t_1 = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)
                x_t, _ = self.q_sample(x0_recon, y, t_1, noise=None) # xt-1
            t = torch.full((x_t.shape[0],), self.steps[-1], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            return self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon).cpu().detach()
        elif type == 'xt-1':
            x_t=y
            for i in range(len(self.steps)-1):
                t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
                x_t = self.denoise_fn(x_t, t=t, y=context)
            return x_t.cpu().detach()
        else:
            raise ValueError(type)
    
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
    ):
        super().__init__()

        self.skip_sample = diff_config.skip_sample
        self.sample_type = diff_config.beta_schedule
        self.loss_type = diff_config.loss_type
        self.objective = diff_config.objective
        self.sample_type = diff_config.beta_schedule
        self.sample_step = diff_config.sample_step
        if not view_only:
            self.denoise_fn = Registers.runners[model_config.name](model_config) # UNetModel(**UNetParams)
            self.device = model_config.device 

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
        elif self.objective == 'x0':
            objective = x0
        else:
            raise NotImplementedError()
        return perturbed_x, objective

    def get_losses(self, x0, y, context, t):
        noise = torch.randn_like(x0)

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
            x_t = self.perturb_x(x0, t, noise)[0][0]
            x_t = (x_t.numpy().clip(-1, 1)+1)/2
            x_t = np.uint8(255 * x_t)
            cv2.imwrite(f"{path}/{name}_{t}.png", x_t)
            # exit()

    def view_q_sample_colors(self, x0, y, path, name):
        noise = torch.randn_like(x0)
        for t in self.steps:
            t_batch = torch.tensor([t])
            x_t = self.perturb_x(x0, t_batch, noise)[0][0]
            x_t = (x_t.numpy().clip(-1, 1)+1)/2
            x_t = np.uint8(255 * x_t)
            plt.imsave(f"{path}/{name}_{t}.png", x_t, cmap=plt.get_cmap('bwr'))
            plt.clf()
