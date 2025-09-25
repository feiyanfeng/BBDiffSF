import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

from tools.utils import nii_resize_2D, resample_3D_nii_to_Fixed_size
from tools.metrics import Metirc

def convert_probability_to_mask_array(predict, choice):
    predict = sitk.GetArrayFromImage(predict)
    mask = np.zeros_like(predict, dtype='uint8')
    mask[predict > choice] = 1
    return mask


class printer:
    def info(self, message, format=None):
        print(message)
    def warning(self, message, format=None):
        print(message)

class Evaluation():
    def __init__(self, save_path, logger=None, choice=0, metrics='all', Global=False, resize=None):
        # 增加指标只需修改3处：value of metrics, result and fun
        '''
        Global: True, False, only
        '''
        if logger is None: logger=printer()
        logger.info('evaluation', '%(asctime)s ----%(message)s---')
        self.logger = logger
        if metrics == 'all':
            self.metrics = ['dice', 'Jaccard', 'recall', 'precision', 'FNR', 'FPR', 'HD95']  # 'RVD', 
        else:
            self.metrics = metrics

        self.result={
            'name':[],
            'dice':[],
            'Jaccard':[],
            'recall':[],
            'precision':[],
            # 'RVD':[],
            'FNR':[],
            'FPR':[],
            'HD95':[]
        }
        self.fun = {
            'dice':'metirc.dice_coef()',
            'Jaccard':'metirc.iou_score()',
            'recall':'metirc.recall()',
            'precision':'metirc.precision()',
            'RVD':'metirc.RVD()',
            'FNR':'metirc.FNR()',
            'FPR':'metirc.FPR()',
            'HD95':'metirc.HD95()'
        }
        self.Global = Global
        self.Single = False if Global=='only' else True
        self.PRE = []
        self.GT = []
        self.resize = resize
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            self.save_path = save_path
        self.choice = choice

    
    def cacu(self, name, predict, groundtruth):        
        pre2 = np.zeros_like(predict, dtype='uint8')
        pre2[predict > self.choice] = 1
        gt2 = np.zeros_like(groundtruth, dtype='uint8')
        gt2[groundtruth > 0] = 1
        if self.resize:
            _, pre2 = nii_resize_2D(None, pre2, self.resize)
        
        if self.Global:
            self.PRE.append(pre2)
            self.GT.append(gt2)
        if self.Single:
            self.result['name'].append(name)
            metirc = Metirc(pre2, gt2)  # 得有
            for m in self.metrics:
                self.result[m].append(eval(self.fun[m]))
                

    def view(self, get_dice=None):
        dice=None
        if self.Single:
            self.logger.info('Single:', '%(message)s')
            for m in self.metrics:
                self.logger.info(f'{m} : {np.nanmean(self.result[m])}', '%(message)s')
            if get_dice=="Single": dice=np.mean(self.result['dice'])
        if self.Global:
            self.logger.info(f'Global:({np.array(self.PRE).shape})({np.array(self.GT).shape})', '%(message)s')
            metirc = Metirc(np.array(self.PRE), np.array(self.GT))
            for m in self.metrics:
                self.logger.info(f'{m}, {eval(self.fun[m])}', '%(message)s')
            if get_dice=='Global': dice=metirc.dice_coef()
        return dice
    
    def get_dice(self, form):
        '''
        form (str): 'Single' or 'Global'.
        '''
        if form == 'Single' and self.Single:
            self.logger.info('Single:', '%(message)s')
            return np.mean(self.result['dice'])  # 即使已经添加了dice均值也不影响
        if form == 'Global' and self.Global:
            PRE, GT = np.array(self.PRE), np.array(self.GT) 
            self.logger.info(f'Global: {PRE.shape}, {GT.shape}', '%(message)s')
            metirc = Metirc(PRE, GT)
            return metirc.dice_coef()
        raise Exception(f'error combination. form:{form}, Single{self.Single}, Global:{self.Global}')
            
    def save(self, csv_name):
        if len(self.result['name'])==0 and len(self.PRE)==0:
            self.logger.warning('no result need to be saved: length=0', '%(message)s')
            return
        if self.Single:
            self.logger.info('Single:', '%(message)s')
            self.result['name'].append('means')
            for m in self.metrics:
                mean = np.nanmean(self.result[m])
                self.logger.info(f'{m} :{mean}', '%(message)s')
                self.result[m].append(mean)
        if self.Global:
            self.logger.info(f'Global:({np.array(self.PRE).shape})({np.array(self.GT).shape})', '%(message)s')
            self.result['name'].append('Global')
            metirc = Metirc(np.array(self.PRE), np.array(self.GT))
            for m in self.metrics:
                g = eval(self.fun[m])
                self.logger.info(f'{m}, {g}', '%(message)s')
                self.result[m].append(g)
        df = pd.DataFrame(self.result)
        df.to_csv(self.save_path + '/' + csv_name, index=False)
        # self.logger.info(df)

