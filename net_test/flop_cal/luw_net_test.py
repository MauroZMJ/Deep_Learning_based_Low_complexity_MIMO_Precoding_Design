import numpy as np

np.random.seed(2021)
from scipy import io
from sklearn import preprocessing
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, Flatten, Permute, Reshape, Input, \
    BatchNormalization, Concatenate, Add, Lambda, GlobalAveragePooling1D, Concatenate, GlobalAvgPool1D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as BK

import os
import hdf5storage
#tf.config.run_functions_eagerly(True)

from option import parge_config
from init_network_preprocessing import data_process
dataset_root = '/data/zhangmaojun/dataset/'
train_mode = 'train'
net_name = 'init_net'
args = parge_config()
Nt = args.Nt
Nr = args.Nr
K = args.K
dk = args.dk
SNR_dB = args.SNR
B = args.B
SNR = 10 ** (SNR_dB / 10)
p = 1
sigma_2 = 1 / SNR
SNR_channel_dB = args.SNR_channel
data_mode = args.mode
batch_size = args.batch_size
epochs = args.epoch
prefix = 'learn_from_H_bar'
##################
# %% supervised training
def backbone(data, Nt, Nr, dk, K, B):
    def vector_norm_l1(vec):
        # vector shape:B*L
        v_norm, _ = tf.linalg.normalize(vec, ord=1, axis=1)
        return v_norm

    def vector_norm_l2(vec):
        v_norm, _ = tf.linalg.normalize(vec, ord=2, axis=1)
        return v_norm

    net = 'sigmoid'
    if net == 'sigmoid':
        data = Reshape((K * Nr * B, K * Nr * B, 1))(data)
        # data = Permute((1,3,2))(data)
        # import ipdb;ipdb.set_trace()
        if K * dk > 12:
            x = Conv2D(filters=16, kernel_size=(7, 7))(data)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        else:
            x = data
        x = Conv2D(filters=8, kernel_size=(5, 5))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        for i in range(3):
            x = Conv2D(filters=4, kernel_size=(3, 3))(x)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        x = Conv2D(filters=2, kernel_size=(3, 3))(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        x = Flatten()(x)

        x_tp = BatchNormalization()(x)
        w_list = Dense(K * dk * dk * B)(x_tp)
        w_list = BatchNormalization()(w_list)
        w_list = Activation('relu')(w_list)
        w_list = Dense(K * dk * dk * B)(w_list)
        w_list = Activation('sigmoid')(w_list)
        w_list = Lambda(vector_norm_l1)(w_list)

        x_q = BatchNormalization()(x)
        u_list = Dense(K * Nr * dk * B * 2)(x_q)
        u_list = BatchNormalization()(u_list)
        u_list = Activation('relu')(u_list)
        u_list = Dense(K * Nr * dk * B * 2)(u_list)
        u_list = Activation('tanh')(u_list)
        # u_list = Lambda(vector_norm_l2)(u_list)
        prediction = Concatenate(axis=1)([u_list, w_list])
    return prediction
#@tf.function
def weighted_mse_loss(y_true, y_pred):
    pa_list_pred = y_pred[:, :K * dk]
    lambda_list_pred = y_pred[:, K * dk:(K*dk + K*dk*B)]
    q_list_pred = y_pred[:,(K*dk + K*dk*B):]

    pa_list_true = y_true[:, :K * dk]
    lambda_list_true = y_true[:, K * dk: (K*dk + K*dk*B)]
    q_list_true = y_true[:,(K*dk + K*dk*B):]

    mse_pa = tf.reduce_mean(tf.square((pa_list_true - pa_list_pred)), axis=-1)
    mse_lambda = tf.reduce_mean(tf.square((lambda_list_true - lambda_list_pred)), axis=-1)
    mse_q = tf.reduce_mean(tf.square((q_list_true - q_list_pred)), axis=-1)
    loss = mse_pa + mse_lambda + mse_q
    return loss


def su_net(Nt,Nr, K, B, dk, lr):
    data = Input(shape=(K*Nr*Nr*K*B*B))
    prediction = backbone(data,Nt,Nr,dk,K,B)
    model = Model(inputs=data, outputs=prediction)
    model.compile(loss=weighted_mse_loss, optimizer=Adam(lr=lr))
    model.summary()
    return model




lr = 1e-2



net_flops_list = []
lcp_flops_list = []
if data_mode=='debug':
    dk_list = [1,2,3,4]
    K_list = [10]
    B = 4
    for dk in dk_list:
        for K in K_list:
            model = su_net(Nt, Nr, K, B, dk, lr)
            from keras_flops import get_flops
            net_flops = get_flops(model,batch_size=1)
            print('DL_flops:'+str(net_flops))
            RB = B
            # if RB>1:
            #     EZF_flops = 8*K**3*dk**3/3 + 16*K**2*Nt*dk**2 + 6*K**2*dk**2 + 10*K*dk/3 + 4*K*(6*Nt**3 + dk*(2*Nt**2 + Nt)) + 4*K*(2*Nr**2*Nt*RB + Nt**2*(RB - 1))
            # else:
            #     EZF_flops = 8*K**3*dk**3/3 + 16*K**2*Nt*dk**2 + 6*K**2*dk**2 + 10*K*dk/3 + 4*K*(4*Nr**3 + 2*Nr**2*Nt + dk*(2*Nr*Nt + Nt))
            if RB>1:
                restore_flops = (8*K**2*Nr**2*Nt*RB**2 + 4*K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2)))
            else:
                M = K*dk
                restore_flops = (8*K**2*Nr**2*Nt*RB**2 + 4*K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2)))
            WMMSE_flops = 4*50*(K*RB*(2*Nr*Nt*dk + 2*Nt*dk**2 + 2*dk**3 + dk**2) + K*RB*(K*(2*Nr**2*dk + 2*Nr*Nt*dk) + K*(2*Nt**2*dk + Nt - 1) + 2*Nr**3/3 + 2*Nr**2*Nt + Nr**2*(K - 1) + 7*Nr**2/2 + 2*Nr*Nt*dk + 5*Nr/6) + K*(K*RB*(2*Nr*Nt**2 + 4*Nr*Nt*dk + 2*Nt*dk**2) + K*RB*(2*Nr**2*dk + 2*Nr*dk**2 + Nr - 1) + 2*Nt**3 + 2*Nt**2*dk + Nt**2*(K*RB - 1) + Nt**2 + Nt*dk*(RB - 1) + RB*(2*Nr*Nt*dk + 2*Nt*dk**2)))
            total_flops =net_flops + restore_flops
            net_flops_list.append(net_flops)
            lcp_flops_list.append(total_flops)
    print('RB:%d K:%d dk:%d '%(RB,K,dk))
    print(net_flops_list)
    print(lcp_flops_list)
    # print('EZF_flops:'+str(EZF_flops))
    # print('Net ratio: %.3f'%(net_flops/EZF_flops))
    #
    # print('restore ratio: %.3f'%(restore_flops/EZF_flops))
    # print('LCP ratio: %.3f'%(total_flops/EZF_flops))
    # print('WMMSE ratio: %.3f'%(WMMSE_flops/EZF_flops))
    #import ipdb;ipdb.set_trace()
