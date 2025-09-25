# import SimpleITK as sitk
import numpy as np
# from utils import resample_3D_nii_to_Fixed_size

# from hausdorff import hausdorff_distance
from medpy.metric import binary


class Metirc():
    
    def __init__(self, output, target):  #output为预测结果 target为真实结果 X ∪ Y = TP + FN
        assert np.max(output)<=1
        assert np.max(target)<=1

        self.output = output.astype('uint8')
        self.target = target.astype('uint8')
        self.smooth = 1e-5 #防止0除
        
        self.intersection = (output * target).sum()
        self.union = (self.output | self.target).sum()

    # dice
    def dice_coef(self):
        return (2. * self.intersection + self.smooth) / (self.output.sum() + self.target.sum() + self.smooth)


    # IOU, also Jaccard Index, 重叠程度 
    # VOE=1-IOU
    def iou_score(self):
        intersection = (self.output & self.target).sum()
        return (intersection + self.smooth) / (self.union + self.smooth)


    # recall  TP/(TP+FN)   sensitivity
    def recall(self):
        return (self.intersection + self.smooth) / (self.target.sum() + self.smooth)


    # TP/(TP+FP)   ppv
    def precision(self):
        return (self.intersection + self.smooth) / (self.output.sum() + self.smooth)
        
    
    # RVD 体素相对误差, Relative Volume Difference
    # (|B|-|A|) / |A| （可正可负）
    def RVD(self):
        return (self.output.sum() - self.target.sum()) / (self.target.sum() + self.smooth)

    # FNR 欠分割率, False negative rate
    # FN/AUB
    def FNR(self):
        fn = self.target.sum() - self.intersection

        return fn / (self.union + self.smooth)


    # FPR 过分割率, False positive rate
    # FP/(AUB)
    def FPR(self):
        fp = self.output.sum() - self.intersection
        return fp / (self.union + self.smooth)


    def HD95(self):
        try:
            return binary.hd95(self.output, self.target)
        except:
            return np.nan
        # return binary.hd95(self.output, self.target)

# csv_save/model4_train01/epoch-299-model__DSC.csv
