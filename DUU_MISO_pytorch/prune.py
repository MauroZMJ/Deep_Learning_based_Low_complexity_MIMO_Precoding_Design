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
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from option import parge_config
from model import BeamformNet,Loss_utils

from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, L2FilterPruner, FPGMPruner
from nni.algorithms.compression.pytorch.pruning import SimulatedAnnealingPruner, ADMMPruner, NetAdaptPruner, AutoCompressPruner
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params

import json

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
def train(model, device, train_loader, criterion, optimizer, epoch, callback=None):
    model.train()
    batch_idx = 0
    log_interval = 100
    for H_bar_data, label_data in train_loader:
        H_bar_data = H_bar_data.clone().detach().type(torch.float32)
        H_bar_data = H_bar_data.to(device)

        label_data = label_data.clone().detach().type(torch.float32)
        label_data = label_data.to(device)
        # forward
        model_output = model(H_bar_data)
        loss = criterion(label_data, model_output)
        loss.backward()
        if callback:
            callback()
        optimizer.step()
        batch_idx = batch_idx + 1
        if batch_idx % log_interval == 0:
            print('Train Epoch: %d,%f'%(epoch,loss.item()))






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

    '''supervised learning'''
    best_su_model_path = base_root + 'model/DUU_pytorch/DUU_models_%d_%d_%d_%d_%d_su.pth'%(Nt,Nr,K,dk,SNR_dB)
    train_su_dataloader = DataLoader(total_dataset.train_su_dataset, shuffle=False, batch_size=batch_size, drop_last=True)

    valid_su_dataloader = DataLoader(total_dataset.valid_su_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    #train(train_su_dataloader,valid_su_dataloader, model, criterion=MSE_loss,save_path = best_su_model_path,  lr = 1e-2)
    #print('supervised learning complete!')

    '''unsupervised learning'''
    best_un_model_path = base_root + 'model/DUU_pytorch/DUU_models_%d_%d_%d_%d_%d_un.pth' % (Nt, Nr, K, dk, SNR_dB)
    train_un_dataloader = DataLoader(total_dataset.train_un_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    valid_un_dataloader = DataLoader(total_dataset.valid_un_dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    #train(train_un_dataloader,valid_un_dataloader, model, criterion=SMR_loss,save_path = best_un_model_path, lr = 1e-3)
    #print('unsupervised learning complete!')

    '''test'''
    test_dataloader = DataLoader(total_dataset.test_un_dataset, shuffle=False, batch_size=batch_size,
                                     drop_last=True)
    model = torch.load(best_un_model_path)


    config_list = [{
        'sparsity': 0.3,
        'op_types': ['Conv2d']
    }]
    def trainer(model, optimizer, criterion, epoch, callback):
        return train(model, device, train_su_dataloader, MSE_loss, optimizer, epoch=epoch, callback=callback)
    def evaluator(model):
        return test(valid_un_dataloader,model,criterion=SMR_loss)

    dummy_input = torch.randn([batch_size, (K*dk)**2]).to(device)
    experiment_data_dir = base_root + 'prune'
    pruner = AutoCompressPruner(
        model, config_list, trainer=trainer, evaluator=evaluator, dummy_input=dummy_input,
        num_iterations=3, optimize_mode='maximize', base_algo='l1',
        cool_down_rate=0.9, admm_num_iterations=30, admm_training_epochs=5,
        experiment_data_dir=experiment_data_dir)
    #pruner = L2FilterPruner(model, config_list)
    result = {'flops': {}, 'params': {}, 'performance': {}}
    model = pruner.compress()

    evaluation_result = evaluator(model)
    print('Evaluation result (masked model): %s' % evaluation_result)
    result['performance']['pruned'] = evaluation_result
    pruner.export_model(
        os.path.join(experiment_data_dir, 'model_masked.pth'), os.path.join(experiment_data_dir, 'mask.pth'))
    print('Masked model saved to %s' % experiment_data_dir)

    flops, params, _ = count_flops_params(model, (1,(K*dk)**2))
    result['flops']['speedup'] = flops
    result['params']['speedup'] = params

    fine_tune_epochs = 10
    best_acc = 0
    for epoch in range(fine_tune_epochs):
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1e-3)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        train(model, device, train_un_dataloader, criterion=SMR_loss, optimizer = optimizer, epoch=epoch, callback=None)
        scheduler.step()
        acc = evaluator(model)
        if acc > best_acc:
            best_acc = acc
            print(best_acc)
            torch.save(model.state_dict(), os.path.join(experiment_data_dir, 'model_fine_tuned.pth'))
    print('Evaluation result (fine tuned): %s' % best_acc)
    print('Fined tuned model saved to %s' % experiment_data_dir)
    result['performance']['finetuned'] = best_acc
    with open(os.path.join(experiment_data_dir, 'result.json'), 'w+') as f:
        json.dump(result, f)


#python train.py --Nt 64 --Nr 4 --K 10 --dk 2 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 1 --test_length 2000

