import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy import io

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data import DataLoader,Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.utils import save_image
import os

from scipy import io
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import os
from tqdm import tqdm,trange
import math
from dataset_load import Dataset_load

from option import parge_config
from model import BeamformNet,Loss_utils
args = parge_config()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
Nt = args.Nt
Nr = args.Nr
K = args.K
dk = args.dk
SNR_dB = args.SNR
SNR = 10**(SNR_dB/10)
p = 1
sigma_2 = 1/SNR
SNR_channel_dB = args.SNR_channel
data_mode = args.mode
batch_size = args.batch_size
epochs = args.epoch
test_length = args.test_length


#dataset_root = '/home/zmj/Desktop/precode/'
#data_root = dataset_root + 'data/DUU_MISO_dataset_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk, SNR_dB)
base_root = '/home/zmj/Desktop/precode/'
data_root = base_root + 'data/DUU_MISO_dataset_%d_%d_%d_%d_%d.mat'%(Nt,Nr,K,dk,SNR_dB)
train_mode = 'train'
model_name = 'CNN2D'

import random
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


setup_seed(2021)
if data_mode=='debug':
    epochs = 10
def validate(data_loader,model,criterion):
    model.eval()
    loss_list = []
    print("Validation started...")
    with torch.no_grad():
        for H_bar_data,label_data in data_loader:
            H_bar_data = H_bar_data.clone().detach().type(torch.float32)
            H_bar_data = H_bar_data.to(device)

            label_data = label_data.clone().detach().type(torch.float32)
            label_data = label_data.to(device)
            # forward
            model_output = model(H_bar_data)
            loss = criterion(label_data,model_output)
            loss_list.append(loss.item())
    model.train()
    return np.mean(loss_list)
def test(data_loader,model,criterion):
    model.eval()
    loss_list = []
    print("test started...")
    with torch.no_grad():
        for H_bar_data,label_data in data_loader:
            H_bar_data = H_bar_data.clone().detach().type(torch.float32)
            H_bar_data = H_bar_data.to(device)

            label_data = label_data.clone().detach().type(torch.float32)
            label_data = label_data.to(device)
            # forward
            model_output = model(H_bar_data)
            loss = -criterion(label_data,model_output)
            loss_list.append(loss.item())
    model.train()
    return np.mean(loss_list)
def train(data_loader, valid_data_loader, model, criterion,save_path,lr):
    min_loss = 1e5
    optimizier = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizier,
                                  factor=0.3,
                                  patience=5,
                                  mode='min',
                                  min_lr=1e-5,
                                  eps=1e-4,
                                  verbose=True)
    loss = None
    for epoch in range(epochs):
        step = 0
        with tqdm(data_loader, desc="epoch:" + str(epoch),
                  postfix={"train_loss": 0} if loss is None else {"train_loss": loss.data}) as iteration:
            for H_bar_data,label_data in iteration:
                iteration.set_description("epoch:" + str(epoch))
                H_bar_data = H_bar_data.clone().detach().type(torch.float32)
                H_bar_data = H_bar_data.to(device)

                label_data = label_data.clone().detach().type(torch.float32)
                label_data = label_data.to(device)
                # forward
                model_output = model(H_bar_data)
                loss = criterion(label_data,model_output)

                iteration.set_postfix(loss=('%.4f' % loss.data.item()))
                # backward

                optimizier.zero_grad()
                loss.backward()
                optimizier.step()
                step = step + 1
        for param_group in optimizier.param_groups:
            print(param_group['lr'])
        if optimizier.param_groups[0]['lr'] == 1e-5:
            print('early stop!')
            break
        valid_loss = validate(valid_data_loader, model, criterion)
        print('valid_loss:' + '%.7f' % valid_loss)
        scheduler.step(valid_loss)
        if valid_loss < min_loss and abs(valid_loss - min_loss) > 1e-5:
            min_loss = valid_loss
            print("model has been saved")
            torch.save(model, save_path)
if __name__ == "__main__":

    total_dataset = Dataset_load(data_root,SNR_channel_dB=SNR_channel_dB,SNR_dB=SNR_dB,test_length = test_length,
                                 Nt=Nt,Nr=Nr,dk=dk,K=K, mode=data_mode)
    Loss_util = Loss_utils(Nt,Nr,dk,K,p,sigma_2)
    MSE_loss = Loss_util.MSE_loss
    SMR_loss = Loss_util.DUU_MISO
    model = BeamformNet(model_name, Nt, Nr, dk, K)
    device = torch.device(0)
    if torch.cuda.is_available():
        model.to(device)

    from convert_tf2pt import convert2pytorch
    model = convert2pytorch(model,Nt,Nr,dk,K,SNR_dB)


