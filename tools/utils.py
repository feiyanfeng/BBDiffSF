# import argparse

import skimage.transform as trans
import numpy as np

import SimpleITK as sitk
# import cv2


import torch.optim as optim
import math


from tools.logger import Logger
import torch
import random

class Optimizer:

    def get_optimizer(self, config_optim, parameters):
        
        if config_optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay,
                            betas=(config_optim.beta1, 0.999), amsgrad=config_optim.amsgrad,
                            eps=config_optim.eps)
        elif config_optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay)
        elif config_optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=config_optim.lr, weight_decay=1e-4, momentum=0.9)
        else:
            raise NotImplementedError(
                'Optimizer {} not understood.'.format(config_optim.optimizer))

    def adjust_learning_rate(self, optimizer, epoch, config):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < config.training.warmup_epochs:
            lr = config.optim.lr * epoch / config.training.warmup_epochs
        else:
            lr = config.optim.min_lr + (config.optim.lr - config.optim.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - config.training.warmup_epochs) / (
                        config.training.epochs - config.training.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

def resample_3D_nii_to_Fixed_size(nii_image, image_new_size, resample_methold=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()

    image_original_size = nii_image.GetSize()  # 原始图像的尺寸
    image_original_spacing = nii_image.GetSpacing()  # 原始图像的像素之间的距离
    image_new_size = np.array(image_new_size, float)
    factor = image_original_size / image_new_size
    image_new_spacing = image_original_spacing * factor
    image_new_size = image_new_size.astype(np.int)

    resampler.SetReferenceImage(nii_image)  # 需要resize的图像（原始图像）
    resampler.SetSize(image_new_size.tolist())
    resampler.SetOutputSpacing(image_new_spacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resample_methold)

    return resampler.Execute(nii_image)


def nii_resize_2D(image, label, shape):
    """
    type of image,label: Image or array or None

    :return: array or None
    """
    # image
    if isinstance(image, sitk.SimpleITK.Image):  # image need type array, if not, transform it
        image = sitk.GetArrayFromImage(image)
    if image is not None:
        image = trans.resize(image, (shape, shape))
    # label
    if isinstance(label, np.ndarray):
        label = sitk.GetImageFromArray(label)  # label1 need type Image
    if label is not None:
        label = resample_3D_nii_to_Fixed_size(label, (shape, shape),
                                              resample_methold=sitk.sitkNearestNeighbor)
        label = sitk.GetArrayFromImage(label)
    return image, label


def resetting(cfg): 
    # # print(cfg)
    # print(cfg.__contains__('device'))
    # print(torch.device("cuda"))
    # cfg.device = torch.device("cuda")
    # print(cfg.device)
    # exit()
    completion = []
    warning = ''

    if not cfg.__contains__('name'): completion.append('name')
    if not cfg.data.__contains__('name'): completion.append('data.name')
    if not cfg.data.__contains__('root'): completion.append('data.root')
    if not cfg.data.__contains__('train_list'): completion.append('data.train_list')
    if not cfg.data.__contains__('test_list'): completion.append('data.test_list')
    if not cfg.model.__contains__('name'): completion.append('model.name')

    assert len(completion)==0, ', '.join(completion)+' should be given'
    # if cfg.test.batch_size:print('batch_size-----', cfg.test.batch_size)
    
    if cfg.diffusion.objective=='xt-1' and cfg.mode=='train': # 只有在训练时 这个参数才有意义。测试时 diffusion.sample_step 可能是个列表
        if cfg.test.type != 'xt-1':
            warning += f'Test type error:{cfg.test.type}, reset to xt-1\n'
            cfg.test.type = 'xt-1'
        if cfg.diffusion.timesteps != cfg.diffusion.sample_step:
            warning += 'Using training objective xt-1 requires consistency between timestep and sample_step\n'
            warning += f'diffusion.sample_step reset to {cfg.diffusion.timestep} ({cfg.diffusion.sample_step} before)\n'
            cfg.diffusion.sample_step = cfg.diffusion.timestep

    if cfg.test.batch_size is None:
        cfg.test.batch_size = cfg.training.batch_size*2

    # if cfg.name=='20240926nMC_MRI_128_f3_new': # TODO:
    #     warning += 'cfg.name reset from 20240926nMC_MRI_128_f3_new to 20240926nMC_MRI_128_f3\n'
    #     cfg.name = '20240926nMC_MRI_128_f3'

    if cfg.log.name is None: cfg.log.name = cfg.name
    if cfg.device is None:
        if torch.cuda.is_available(): cfg.device = "cuda"
        else:
            cfg.device = "cpu"
            warning += 'No GPU available, CPU is used!\n'
    cfg.diffusion.device = cfg.device
    cfg.model.device = cfg.device

    # if cfg.log.name in ['20240926pilot_128_nMC_f4', '20240926pilot_128_nMC_f5']: # TODO:
    #     warning += f'epoch reset from {cfg.training.epochs} to 300\n'
    #     cfg.training.epochs = 300
    return cfg, warning

def print_namespace(args, logger):
    for arg_name, arg_value in args.items(): logger.info(f'### {arg_name}: {arg_value}', '%(message)s')

def set_random_seed(SEED=1234, logger=None):
    if logger: logger.info('set random seed')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True  # 使用cuDNN库来加速计算，通常默认就是True
    torch.backends.cudnn.benchmark = False # 当benchmark为True时，意味着PyTorch会在程序开始时花费一些时间来寻找最适合当前配置（如输入数据大小和网络结构）的cuDNN算法
    torch.backends.cudnn.deterministic = True # 当deterministic为True时，每次计算的结果都是确定性的。即对于相同的输入，每次运行都会得到相同的输出。但是，设置为True可能会降低计算性能。默认值通常是 False，相同的输入可能在多次运行中得到不同的输出


def task_wrapper(task_func):
    def wrap(cfg):
        cfg, warning = resetting(cfg)
        logger = Logger(f'log/{cfg.log.name}.log')
        if warning: logger.warning(warning)
        print_namespace(cfg._content, logger)
        logger.info(f'{cfg.mode} to {cfg.log.name}\n')
        logger.info(f'\n{cfg.mode} start...\n')
        if cfg.random.open is False:
            set_random_seed(cfg.random.seed, logger)
        return task_func(cfg, logger)
    return wrap
