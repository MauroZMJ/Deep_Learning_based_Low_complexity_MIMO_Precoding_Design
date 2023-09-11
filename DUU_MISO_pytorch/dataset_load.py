import numpy as np
from scipy import io
import torch
from torch import nn, optim
from torch.autograd import Variable
#from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torchvision.utils import save_image
import os
from scipy import io
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
from tqdm import tqdm, trange
from sklearn import preprocessing
from data_preprocess import data_process
class MyDataset(Dataset):
    def __init__(self, training_dataset, training_labelset,mode):
        self.dataset = training_dataset
        self.labelset = training_labelset
        self.mode = mode

    def __getitem__(self, index):
        # if mode=='su':
        channel_bar_iter = self.dataset[index,:]
        label_iter = self.labelset[index,:]
        return torch.from_numpy(channel_bar_iter),torch.from_numpy(label_iter)
    def __len__(self):
        return int(self.dataset.shape[0])


class Dataset_load():
    def __init__(self, dataset_root,SNR_channel_dB,SNR_dB,test_length,Nt,Nr,dk,K,mode='gpu'):
        dataset, test_dataset, dataset_bar, test_dataset_bar, H, test_H, H_noiseless, test_H_noiseless, labelset_su, test_labelset_su = \
            data_process(dataset_root, Nt, Nr, dk, K, SNR_dB, SNR_channel_dB, test_length, mode)

        train_su_dataset = MyDataset(dataset_bar,labelset_su,mode='su')
        self.train_su_dataset,self.valid_su_dataset = torch.utils.data.random_split(train_su_dataset,
                                                        [len(train_su_dataset)-len(train_su_dataset)//10,
                                                         len(train_su_dataset)//10])
        labelset_un = np.concatenate([dataset , H_noiseless], axis=-1)
        train_un_dataset = MyDataset(dataset_bar,labelset_un,mode = 'un')
        self.train_un_dataset, self.valid_un_dataset = torch.utils.data.random_split(train_un_dataset,
                                                                                     [len(train_un_dataset) -
                                                                                      len(train_un_dataset) // 10,
                                                                                      len(train_un_dataset) // 10])
        #self.train_su_dataset = MyDataset(dataset,labelset_su)
        test_labelset_un = np.concatenate([test_dataset , test_H_noiseless], axis=-1)
        self.test_su_dataset = MyDataset(test_dataset_bar,test_labelset_su,mode = 'un')
        self.test_un_dataset = MyDataset(test_dataset_bar,test_labelset_un,mode = 'un')