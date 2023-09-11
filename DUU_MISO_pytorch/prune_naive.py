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

import torch_pruning as tp
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

base_root = '/mnts2d/diis_data1/zmj/LCP_dataset/dataset/'
data_root = base_root + 'data/DUU_MISO_dataset_%d_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk,1, SNR_dB)
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
def train(data_loader, valid_data_loader, model, criterion,save_path,lr,epochs,mode='su'):
    min_loss = 1e5
    if mode=='su':
        threshold = 1e-6
    else:
        threshold = 1e-4
    optimizier = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizier,
                                  factor=0.3,
                                  patience=5,
                                  mode='min',
                                  min_lr=1e-5,
                                  eps=1e-5,
                                  verbose=True,                                  
                                  threshold = 1e-5,
                                  threshold_mode = 'abs')
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
    # model = torch.load(best_su_model_path)
    # su_performance = test(test_dataloader,model,criterion=SMR_loss)
    #
    model = torch.load(best_un_model_path)
    un_performance = test(test_dataloader,model,criterion=SMR_loss)


    import logging

    logger = logging.getLogger('mytest')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh = logging.FileHandler(base_root + 'model/DUU_pytorch/Prune_models_%d_%d_%d_%d_%d.log'%(Nt,Nr,K,dk,SNR_dB))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    # logger.info('supervised learning performance:' + str(su_performance))
    # logger.info('unsupervised learning performance:' + str(un_performance))
    # Pruning
   # from convert_tf2pt import convert2pytorch
   #model = convert2pytorch(model,Nt,Nr,dk,K,SNR_dB)
    amount_CNN1 = 0  # Pruning factor
    amount_CNN2 = 0
    amount_CNN3 = 0
    # amount_CNN4 = 0
    for amount_CNN1 in [8,12,15]:
        for amount_CNN2 in [4,6,7]:
            for amount_CNN3 in [0]:
                # Load pre-training model
                model = torch.load(best_un_model_path)
                amount_CNN1_factor = ((amount_CNN1) / model.bfnet.CNN1.weight.shape[0])
                amount_CNN2_factor = ((amount_CNN2) / model.bfnet.CNN2.weight.shape[0])
                amount_CNN3_factor = ((amount_CNN3) / model.bfnet.CNN3.weight.shape[0])
                # Pruned model save path
                org_pruned_model_path = base_root + 'prune/model/org_pruned_model_%d_%d_%d_%d_%d_L1_%d_%d_%d.pth' \
                                        % (Nt,Nr,K,dk,SNR_dB,amount_CNN1, amount_CNN2, amount_CNN3)
                # Fine-tuned model save path
                fine_tune_model_path = base_root + 'prune/model/fine_tune_model_%d_%d_%d_%d_%d_L1_%d_%d_%d.pth' \
                                       % (Nt,Nr,K,dk,SNR_dB,amount_CNN1, amount_CNN2, amount_CNN3)
                # Pruning strategy
                DG = tp.DependencyGraph()
                DG.build_dependency(model, example_inputs=torch.randn(1, (K*dk)**2))
                strategy = tp.strategy.L1Strategy()

                # Prune CNN1
                pruning_idxs = strategy(model.bfnet.CNN1.weight, amount=amount_CNN1_factor)
                pruning_plan = DG.get_pruning_plan(model.bfnet.CNN1, tp.prune_conv, idxs=pruning_idxs)
                pruning_plan.exec()
                logger.info("CNN1: prune %d channels" % amount_CNN1)
                # Prune CNN2
                pruning_idxs = strategy(model.bfnet.CNN2.weight, amount=amount_CNN2_factor)
                pruning_plan = DG.get_pruning_plan(model.bfnet.CNN2, tp.prune_conv, idxs=pruning_idxs)
                pruning_plan.exec()
                logger.info("CNN2: prune %d channels" % amount_CNN2)
                # Prune CNN3
                pruning_idxs = strategy(model.bfnet.CNN3.weight, amount=amount_CNN3_factor)
                pruning_plan = DG.get_pruning_plan(model.bfnet.CNN3, tp.prune_conv, idxs=pruning_idxs)
                pruning_plan.exec()
                logger.info("CNN3: prune %d channels" % amount_CNN3)

                torch.save(model, org_pruned_model_path)
                # Fine-tune
                model.to(device)
                train(train_su_dataloader,valid_su_dataloader,model,criterion=MSE_loss,save_path=fine_tune_model_path,lr=1e-2,epochs = 300,mode='su')
                train(train_un_dataloader, valid_un_dataloader, model, criterion=SMR_loss,
                      save_path=fine_tune_model_path,
                      lr=1e-3,epochs = 10,mode='un')
                logger.info('fine-tune complete!')
                # Test
                model.to(device)
                prune_performance = test(test_dataloader, model, criterion=SMR_loss)
                logger.info('test complete! pruned model performance:' + str(prune_performance))
                logger.info('compared with original model:' + str(prune_performance / un_performance))
