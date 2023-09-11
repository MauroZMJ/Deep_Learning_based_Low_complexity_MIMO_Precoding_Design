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
def complex_det(A):
    A_real = A.real
    A_imag = A.imag
    upper_matrix = torch.cat((A_real,-A_imag),dim = 2)
    lower_matrix = torch.cat((A_imag,A_real),dim = 2)
    Matrix = torch.cat(((upper_matrix,lower_matrix)),dim = 1)
    det_result = torch.linalg.det(Matrix)
    return torch.sqrt(det_result)

class Loss_utils():
    def __init__(self,Nt,Nr,dk,K,p,sigma_2):
        super(Loss_utils).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.dk = dk
        self.K = K
        self.p = p
        self.sigma_2 = sigma_2
    def MSE_loss(self,y_true,y_pred):
        Nt = self.Nt
        Nr = self.Nr
        dk = self.dk
        K = self.K
        p = self.p
        sigma_2 = self.sigma_2

        loss = F.mse_loss(y_pred, y_true, reduction='mean')
        return loss
    def SMR_loss(self,y_true,y_pred):
        Nt = self.Nt
        Nr = self.Nr
        dk = self.dk
        K = self.K
        p = self.p
        sigma_2 = self.sigma_2
        batch_size = y_true.shape[0]
        #H_noiseless = torch.view_as_complex(y_true[:,:(2*Nt*Nr*K)].reshape((-1,Nt,Nr,2,K)).permute(0,1,2,4,3).contiguous())
        H = torch.view_as_complex(y_true.reshape((-1,Nt,Nr,2,K)).permute(0,1,2,4,3).contiguous())

        #restore V
        V = torch.view_as_complex(y_pred.reshape((-1,Nt,dk,K,2)).contiguous())
        '''precode matrix normalize'''
        V_flatten = V.reshape((-1,Nt*dk*K))
        energy_scale = torch.linalg.norm(V_flatten,axis=1).reshape((-1,1,1,1)).repeat(1,Nt,dk,K).type_as(H)
        V = V/energy_scale
        #V = self.DUU_EZF(H,p_list_pred,q_list_pred,mrt_list_pred)
        '''need to change for normal runing'''
        sum_rate = torch.zeros(1).cuda()
        for user in range(K):
            H_k = H[:,:,:,user].permute(0,2,1)
            V_k = V[:,:,:,user]
            signal_k = torch.matmul(H_k, V_k)
            signal_k_energy = torch.matmul(signal_k,torch.conj(signal_k.permute(0,2,1)))
            interference_k_energy = sigma_2 * torch.eye(Nr).type_as(H).reshape((1,Nr,Nr)).repeat(batch_size,1,1)
            for j in range(K):
                if j!=user:
                    V_j = V[:, :, :, j]
                    interference_j = torch.matmul(H_k, V_j)
                    interference_k_energy = interference_k_energy + torch.matmul(interference_j,torch.conj(interference_j.permute(0,2,1)))
                SINR_k = torch.matmul(signal_k_energy, torch.linalg.inv(interference_k_energy))
                rate_k = torch.log2(complex_det(SINR_k + torch.eye(Nr).type_as(H).reshape((1,Nr,Nr)).repeat(batch_size,1,1)))
            sum_rate = sum_rate + rate_k
        sum_rate = - sum_rate
        return torch.mean(sum_rate)
    def pq2V(self,channel,p_list,q_list):
        Nt = self.Nt
        Nr = 1
        dk = 1
        K = self.K * self.dk
        p = self.p
        sigma_2 = self.sigma_2
        batch_size = channel.shape[0]
        channel = torch.view_as_complex(channel.reshape((-1,Nt,Nr,2,K)).permute(0,1,2,4,3).contiguous())
        #channel = torch.conj(channel)
        P = channel[:,:,0,:]
        p_list = p_list.type_as(channel)
        q_list = q_list.type_as(channel)
        weighted_P = P * torch.sqrt(q_list).reshape((-1, 1, K)).repeat(1, Nt, 1)
        B = sigma_2 * torch.eye(K).type_as(channel).reshape((1, K, K)).repeat(batch_size, 1, 1)
        temp = weighted_P
        B = B + torch.matmul(torch.conj(temp.permute(0, 2, 1)), temp)
        P = torch.matmul(weighted_P, torch.linalg.inv(B))
        V = []
        for user in range(K):
                V_temp = P[:, :, user]/torch.sqrt(q_list[:, user]+1e-24).reshape((-1, 1)).repeat(1, Nt)
                V_temp = torch.sqrt(p_list[:, user]).reshape((-1, 1)).repeat(1, Nt) * V_temp/(torch.norm(V_temp,dim=1).reshape(-1, 1).repeat(1, Nt) + 1e-48)
                V.append(V_temp)
        V = torch.conj(torch.stack(V, dim=2).reshape((-1, Nt, dk, K,1)))
        V = torch.cat((V.real,V.imag),dim=-1)
        return V
    def DUU_MISO(self,y_true,y_pred):
        Nt = self.Nt
        Nr = self.Nr
        dk = self.dk
        K = self.K
        p = self.p
        sigma_2 = self.sigma_2
        batch_size = y_true.shape[0]
        channel_miso = y_true[:, :2 * dk * Nt * K]
        channel = y_true[:, 2 * dk * Nt * K:]
        p_list_pred = y_pred[:, :K * dk]
        q_list_pred = y_pred[:, K * dk:2 * K * dk]
        V_restore = self.pq2V(channel=channel_miso , p_list = p_list_pred , q_list = q_list_pred)
        V_restore = V_restore.reshape(-1,Nt,dk,K,2)#tf.reshape(V_restore, (-1, Nt, dk, K, 2))
        loss = self.SMR_loss(channel,V_restore)

        return loss
class CNN_2D_net(nn.Module):
    def __init__(self, Nt, Nr, dk, K):
        super(CNN_2D_net, self).__init__()
        self.Nt = Nt
        self.Nr = Nr
        self.dk = dk
        self.K = K
        first_filter_num = 16
        second_filter_num = 8
        third_filter_num = 4
        self.CNN1 = torch.nn.Conv2d(in_channels=1, out_channels=first_filter_num, kernel_size=7, stride=1, padding=0)
        #padding=(kernel_size-1)/2 )
        self.bn1 = nn.BatchNorm2d(first_filter_num)
        self.CNN2 = torch.nn.Conv2d(in_channels=first_filter_num, out_channels=second_filter_num, kernel_size=5, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(second_filter_num)
        self.CNN3 = torch.nn.Conv2d(in_channels=second_filter_num, out_channels=third_filter_num, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(third_filter_num)
        #self.CNN4 = torch.nn.Conv1d(in_channels=Nr * 2 * K, out_channels=dk * 2 * K, kernel_size=3, stride=1, padding=1)
        flatten_dim = (dk*K-12)**2 * third_filter_num
        self.fcp = nn.Linear(flatten_dim,dk*K)
        self.fcq = nn.Linear(flatten_dim, dk*K)
        #self.fcr = nn.Linear(flatten_dim, 1)
    def forward(self,x):
        #x = x.permute(0,3,1,2)
        x = x.reshape((-1,self.K*self.dk,self.K*self.dk,1)).permute(0,3,1,2)
        #x = nn.ReLU(inplace =True)(self.bn1(self.CNN1(x)))
        #x = nn.ReLU(inplace=True)(self.bn2(self.CNN2(x)))
        #x = nn.ReLU(inplace=True)(self.bn3(self.CNN3(x)))
        x = F.leaky_relu(self.bn1(self.CNN1(x)), negative_slope=0.3)
        x = F.leaky_relu(self.bn2(self.CNN2(x)), negative_slope=0.3)
        x = F.leaky_relu(self.bn3(self.CNN3(x)), negative_slope=0.3)

        #x = F.relu(self.bn1(self.CNN1(x)))
        #x = F.relu(self.bn2(self.CNN2(x)))
        #x = F.relu(self.bn3(self.CNN3(x)))


        #x = self.CNN4(x)
        #x = x.permute(0,2,1).reshape((-1,self.dk*self.K*2*self.Nt))

        x_flatten = x.reshape((x.shape[0],-1))
        p_pred = self.fcp(x_flatten)
        p_pred = F.softmax(p_pred, dim=1)
        p_pred = p_pred/(torch.sum(p_pred,dim=1).reshape(p_pred.shape[0],1).repeat(1,self.K*self.dk))
        q_pred = self.fcq(x_flatten)
        q_pred = F.softmax(q_pred,dim=1)
        q_pred = q_pred / (torch.sum(q_pred, dim=1).reshape(q_pred.shape[0], 1).repeat(1, self.K * self.dk))
        pq_pred = torch.cat((p_pred,q_pred),dim=-1)
        # mrt_pred = torch.sigmoid(self.fcr(x_flatten))
        # mrt_pred = 0.005 + 0.025 * mrt_pred
        return pq_pred

class BeamformNet(nn.Module):
    def __init__(self, model_name, Nt, Nr, dk, K):
        super(BeamformNet, self).__init__()
        self.bfnet = CNN_2D_net(Nt,Nr,dk,K)
    def forward(self,x):
        return self.bfnet(x)
