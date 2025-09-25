from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics

import numpy as np
import pandas as pd


def load_data(args):
    # get ids 
    with open(f"{args.root}{args.data_path}/{args.train_list}.txt", "r") as f:  #打开文本  #, encoding='utf-8'
        train_list = f.read()   #读取文本
    train_list = train_list.split('\n')

    with open(f"{args.root}{args.data_path}/{args.test_list}.txt", "r") as f:  #打开文本  #, encoding='utf-8'
        val_list = f.read()   #读取文本
    val_list = val_list.split('\n')
    
    train_set = DataGen(args.root, args.data_path, train_list)
    val_set = DataGen(args.root, args.data_path, val_list)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    return train_loader, val_loader

def load_test(args):
    # get ids 
    with open(f"{args.root}{args.data_path}/{args.test_list}.txt", "r") as f:  #打开文本  #, encoding='utf-8'
        test_list = f.read()   #读取文本
    test_list = test_list.split('\n')

    test_set = TestGen(args.root, args.data_path, test_list)

    loader_args = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    return DataLoader(test_set, shuffle=False, drop_last=False, **loader_args)


class DataGen(Dataset):
    def __init__(self, root, data_path, name_list):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}', allow_pickle=True)
        return data[0], data[1], data[2]  # npy = [image, mask]
        # return (data[0]+1)/2, (data[1]+1)/2, data[2]
    
    def __len__(self):
        return len(self.name_list)
    

class TestGen(Dataset):
    def __init__(self, root, data_path, name_list):
        self.root = root
        self.data_path = data_path
        self.name_list = name_list

    def __getitem__(self, index):
        data = np.load(f'{self.root}{self.data_path}/{self.name_list[index]}', allow_pickle=True)
        return self.name_list[index], data[0], data[1]  # npy = [image, mask]
        # return self.name_list[index], (data[0]+1)/2, (data[1]+1)/2
    
    def __len__(self):
        return len(self.name_list)



def cacu(y_true, y_pred, see=False):
    print(metrics.confusion_matrix(y_true, y_pred))
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_pred)
    if see:
        print("accuracy:", accuracy)
        print("precision:", precision)
        print("recall:", recall)
        print("f1_score:", f1_score)
        print('auc:', auc)
    return accuracy, precision, recall, f1_score, auc

def cacu_multi_class(y_true, y_pred, see=False):
    print(metrics.confusion_matrix(y_true, y_pred))
    accuracy = metrics.accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    # auc = metrics.roc_auc_score(y_true, y_pred)
    if see:
        print("accuracy:", accuracy)
        print("precision:", precision)
        print("recall:", recall)
        print("f1_score:", f1_score)
        # print('auc:', auc)
    return accuracy, precision, recall, f1_score #, auc
    