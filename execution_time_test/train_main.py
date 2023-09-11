import numpy as np

np.random.seed(2021)
from scipy import io
from sklearn import preprocessing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
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
from data_preprocess import data_process

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
if data_mode=='debug':
    epochs = 1
test_length = 2000
dataset_root = '/mnts2d/diis_data1/zmj/LCP_dataset/dataset/'
data_root = dataset_root + 'data/DUU_MISO_dataset_%d_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk,B, SNR_dB)
train_mode = 'train'

dataset,test_dataset,dataset_bar,test_dataset_bar,H,test_H,H_noiseless,test_H_noiseless,labelset_su,test_labelset_su = \
                                        data_process(data_root,Nt,Nr,dk,K,SNR_dB,SNR_channel_dB,test_length,data_mode)
prefix = 'learn_from_H_bar'
##################
# %% supervised training
def backbone(data,Nt,Nr,dk,K):
    def vector_norm(vec):
        #vector shape:B*L

        v_norm,_ = tf.linalg.normalize(vec,ord=1,axis=1)
        return v_norm
    net = 'sigmoid'
    if net=='softmax':
        data = Reshape((Nt,2*K*dk))(data)
        x = Conv1D(filters=int(40),kernel_size=7)(data)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv1D(filters=int(20), kernel_size=5)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv1D(filters=int(10), kernel_size=3)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        tp_list = Dense(K * dk)(x)
        tp_list = Activation('softmax')(tp_list)
        up_list = Dense(K * dk)(x)
        up_list = Activation('softmax')(up_list)
        prediction = Concatenate(axis=1)([tp_list, up_list])
    else:
        data = Reshape((K*dk,K*dk,1))(data)
        #import ipdb;ipdb.set_trace()
        if K*dk>12:
            x = Conv2D(filters=16, kernel_size=(7, 7))(data)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        else:
            x = data
        try:
            x = Conv2D(filters=8, kernel_size=(5, 5))(x)
        except:
            x = Conv2D(filters=8, kernel_size=(5, 5),padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        try:
            x = Conv2D(filters=4, kernel_size=(3, 3))(x)
        except:
            x = Conv2D(filters=4, kernel_size=(3, 3),padding='same')(x)    
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        tp_list = Dense(K * dk)(x)
        tp_list = Activation('sigmoid')(tp_list)
        tp_list = Lambda(vector_norm)(tp_list)
        up_list = Dense(K * dk)(x)
        up_list = Activation('sigmoid')(up_list)
        up_list = Lambda(vector_norm)(up_list)
        prediction = Concatenate(axis=1)([tp_list, up_list])
    return prediction

def weighted_mse_loss(y_true, y_pred):
    p_list_pred = y_pred[:, :K * dk]
    q_list_pred = y_pred[:, K * dk:2 * K * dk]
    p_list_true = y_true[:, :K * dk]
    q_list_true = y_true[:, K * dk:2 * K * dk]
    mse_p = tf.reduce_mean(tf.square((p_list_true - p_list_pred)), axis=-1)
    mse_q = tf.reduce_mean(tf.square((q_list_true - q_list_pred)), axis=-1)
    loss = mse_p + mse_q
    return loss


def su_net(Nt,Nr, K, dk, lr):
    data = Input(shape=(K*dk*K*dk))
    prediction = backbone(data,Nt,Nr,dk,K)
    model = Model(inputs=data, outputs=prediction)
    model.compile(loss=weighted_mse_loss, optimizer=Adam(lr=lr))
    model.summary()
    return model


lr = 1e-2

model = su_net(Nt,Nr, K, dk, lr)

su_model_path = dataset_root + 'model/DUU_MISO_models_%d_%d_%d_%d_%d_%d_su.h5' % (Nt, Nr, K, dk, SNR_dB,SNR_channel_dB)
checkpointer = ModelCheckpoint(su_model_path, verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-5, min_lr=1e-5)

if train_mode == 'train':
    model.fit(dataset_bar, labelset_su, epochs=epochs, batch_size=batch_size, verbose=1, \
              validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])


# %% unsupervised training
def minus_sum_rate_loss(y_true, y_pred):
    '''
                y_true is the channels
                y_pred is the predicted beamformers
                notice that, y_true has to be the same shape as y_pred
    '''
    ## construct complex data  channel shape:Nt,Nr,2*K   y_pred shape:Nt,dk,K,2
    y_true = tf.cast(tf.reshape(y_true, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = y_true[:, :, :, :K] + 1j * y_true[:, :, :, K:]
    y_pred = tf.cast(y_pred, tf.complex128)
    V0 = y_pred[:, :, :, :, 0] + 1j * y_pred[:, :, :, :, 1]

    ## power normalization of the predicted beamformers
    # VV = tf.matmul(V0,tf.transpose(V0,perm=[0,2,1],conjugate = True))
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[:, :, :, user], tf.transpose(V0[:, :, :, user], perm=[0, 2, 1], conjugate=True)))

    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    energy_scale = tf.tile(tf.reshape(energy_scale, (-1, 1, 1, 1)), (1, Nt, dk, K))
    V = V0 * tf.cast(energy_scale, tf.complex128)
    sum_rate = 0.0
    # import ipdb;ipdb.set_trace()
    for k in range(K):
        H_k = tf.transpose(H[:, :, :, k], perm=[0, 2, 1])  # NrxNt
        V_k = V[:, :, :, k]  # Ntx1
        signal_k = tf.matmul(H_k, V_k)
        signal_k_energy = tf.matmul(signal_k, tf.transpose(signal_k, perm=[0, 2, 1], conjugate=True))
        interference_k_energy = 0.0
        for j in range(K):
            if j != k:
                V_j = V[:, :, :, j]
                interference_j = tf.matmul(H_k, V_j)
                interference_k_energy = interference_k_energy + tf.matmul(interference_j,
                                                                          tf.transpose(interference_j, perm=[0, 2, 1],
                                                                                       conjugate=True))
        SINR_k = tf.matmul(signal_k_energy,
                           tf.linalg.inv(interference_k_energy + sigma_2 * tf.eye(Nr, dtype=tf.complex128)))
        rate_k = tf.math.log(tf.linalg.det(tf.eye(Nr, dtype=tf.complex128) + SINR_k)) / tf.cast(tf.math.log(2.0),
                                                                                                dtype=tf.complex128)
        sum_rate = sum_rate + rate_k
    sum_rate = tf.cast(tf.math.real(sum_rate), tf.float32)
    # loss
    loss = -sum_rate
    return loss


#@tf.function
def pq2V(channel,Nt,Nr,dk,K,p,sigma_2,q_list,p_list):
    channel = tf.cast(channel,tf.complex128)
    channel = tf.reshape(channel,(-1,Nt,Nr,K*2))
    channel = channel[:,:,0,:K] + 1j*channel[:,:,0,K:]  #new channel shape:(B,Nt,K)
    p_list = tf.cast(p_list,tf.complex128)
    q_list = tf.cast(q_list, tf.complex128)
    V_norm = list()
    temp_inverse = tf.eye(Nt,dtype=tf.complex128)
    for i in range(K):
        Hi = channel[:,:,i:i+1]
        temp_inverse = temp_inverse + 1/sigma_2 * tf.tile(tf.reshape(q_list[:,i],[-1,1,1]),(1,Nt,Nt)) * \
                       tf.matmul(Hi,tf.transpose(Hi, [0, 2, 1], conjugate=True))
    temp_inverse = tf.linalg.inv(temp_inverse)
    for k in range(K):
        Hk = channel[:,:,k:k+1]
        V_temp = tf.matmul(temp_inverse,Hk)
        '''''There may be a problem should be changed'''
        V_temp,_ = tf.linalg.normalize(V_temp[:,:,0],axis=1)
        V_norm.append(V_temp)
    V_norm = tf.stack(V_norm,2)

    V = list()
    for k in range(K):
        V_temp = tf.tile(tf.reshape(tf.sqrt(p_list[:,k]),(-1,1)),(1,Nt)) * V_norm[:,:,k]
        V.append(V_temp)
    V = tf.stack(V,2) #(B,Nt,K)
    V = tf.math.conj(tf.reshape(V,(-1,Nt,1,K,1)))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V

def pq2V_low_complexity(channel,Nt,Nr,dk,K,p,sigma_2,q_list,p_list):
    channel = tf.cast(channel,tf.complex128)
    channel = tf.reshape(channel,(-1,Nt,Nr,K*2))
    channel = channel[:,:,0,:K] + 1j*channel[:,:,0,K:]  #new channel shape:(B,Nt,K)
    p_list = tf.cast(p_list,tf.complex128)+1e-16
    q_list = tf.cast(q_list, tf.complex128)+1e-16
    V_norm = list()
    B = sigma_2 * tf.eye(K,dtype = tf.complex128)
    weight_P = channel * tf.tile(tf.reshape(tf.sqrt(q_list),[-1,1,K*dk]),(1,Nt,1))
    #weight_P = tf.matmul(channel,tf.sqrt(tf.linalg.diag(q_list)),b_is_sparse=True)
    B = B + tf.matmul(weight_P,weight_P,adjoint_a=True)
    P = tf.matmul(weight_P,tf.linalg.inv(B))
    V = list()
    for user in range(K):
            V_temp = P[:,:,user] / tf.tile(tf.reshape(tf.sqrt(q_list[:,user]),[-1,1]),(1,Nt))
            V_temp = tf.tile(tf.reshape(tf.sqrt(p_list[:,user]),[-1,1]),(1,Nt))*V_temp/tf.tile(tf.reshape(tf.norm(V_temp,axis = 1),[-1,1]),(1,Nt))
            V.append(V_temp)
    V = tf.stack(V,2) #(B,Nt,K)
    V = tf.math.conj(tf.reshape(V,(-1,Nt,1,K,1)))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V
#@tf.function
def DUU_EZF_loss(y_true, y_pred):
    # import ipdb;ipdb.set_trace()
    channel_miso = y_true[:, :2 * dk * Nt * K]
    channel = y_true[:, 2 * dk * Nt * K:]
    # channel = y_true
    p_list_pred = y_pred[:, :K * dk]
    q_list_pred = y_pred[:, K * dk:2 * K * dk]
    V_restore = pq2V_low_complexity(channel_miso,Nt=Nt,Nr=1,dk=1,K=K*dk,p=1,sigma_2=sigma_2,q_list=q_list_pred,p_list=p_list_pred)
    V_restore = tf.reshape(V_restore,(-1,Nt,dk,K,2))
    loss = minus_sum_rate_loss(channel, V_restore)
    return loss


def un_net(Nt,Nr, K, dk, lr):
    data = Input(shape=(K*dk*K*dk))
    prediction = backbone(data,Nt,Nr,dk,K)

    model = Model(inputs=data, outputs=prediction)
    model.compile(loss=DUU_EZF_loss, optimizer=Adam(lr=lr))
    model.summary()

    return model


lr = 1e-3

model = un_net(Nt,Nr, K, dk, lr)
model.load_weights(su_model_path)

un_model_path = dataset_root + 'model/DUU_MISO_models_%d_%d_%d_%d_%d_%d_un.h5' % (Nt, Nr, K, dk, SNR_dB,SNR_channel_dB)
checkpointer = ModelCheckpoint(un_model_path, verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=25)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-4, min_lr=1e-5)

data_num = len(H)


H = np.reshape(H, (data_num, -1))


labelset_un = np.concatenate([dataset , H_noiseless], axis=-1)
if train_mode == 'train':
    model.fit(dataset_bar, labelset_un, epochs=epochs, batch_size=batch_size, verbose=1, \
              validation_split=0.1, callbacks=[checkpointer, reduce_lr, early_stopping])
class pq2V_low_complexity_for_restore(tf.keras.layers.Layer):
    def __init__(self):
        super(pq2V_low_complexity_for_restore, self).__init__()
    def call(self,channel,Nt,Nr,dk,K,p,sigma_2,prediction):
        channel = tf.cast(channel, tf.complex128)
        channel = tf.reshape(channel, (-1, Nt, Nr, K * 2))
        channel = channel[:, :, 0, :K] + 1j * channel[:, :, 0, K:]  # new channel shape:(B,Nt,K)
        p_list = prediction[:, :K * dk]
        q_list = prediction[:, K * dk:2 * K * dk]
        p_list = tf.cast(p_list, tf.complex128) + 1e-16
        q_list = tf.cast(q_list, tf.complex128) + 1e-16
        V_norm = list()
        B = sigma_2 * tf.eye(K, dtype=tf.complex128)
        weight_P = channel * tf.tile(tf.reshape(tf.sqrt(q_list), [-1, 1, K * dk]), (1, Nt, 1))
        # weight_P = tf.matmul(channel,tf.sqrt(tf.linalg.diag(q_list)),b_is_sparse=True)
        B = B + tf.matmul(weight_P, weight_P, adjoint_a=True)
        P = tf.matmul(weight_P, tf.linalg.inv(B))
        V = list()
        for user in range(K):
            V_temp = P[:, :, user] / tf.tile(tf.reshape(tf.sqrt(q_list[:, user]), [-1, 1]), (1, Nt))
            V_temp = tf.tile(tf.reshape(tf.sqrt(p_list[:, user]), [-1, 1]), (1, Nt)) * V_temp / tf.tile(
                tf.reshape(tf.norm(V_temp, axis=1), [-1, 1]), (1, Nt))
            V.append(V_temp)
        V = tf.stack(V, 2)  # (B,Nt,K)
        V = tf.math.conj(tf.reshape(V, (-1, Nt, 1, K, 1)))
        V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
        return V
# def pq2V_low_complexity_for_restore(channel,q_list,p_list):
#     channel = tf.cast(channel,tf.complex128)
#     channel = tf.reshape(channel,(-1,Nt,Nr,K*2))
#     channel = channel[:,:,0,:K] + 1j*channel[:,:,0,K:]  #new channel shape:(B,Nt,K)
#     p_list = tf.cast(p_list,tf.complex128)+1e-16
#     q_list = tf.cast(q_list, tf.complex128)+1e-16
#     V_norm = list()
#     B = sigma_2 * tf.eye(K,dtype = tf.complex128)
#     weight_P = channel * tf.tile(tf.reshape(tf.sqrt(q_list),[-1,1,K*dk]),(1,Nt,1))
#     #weight_P = tf.matmul(channel,tf.sqrt(tf.linalg.diag(q_list)),b_is_sparse=True)
#     B = B + tf.matmul(weight_P,weight_P,adjoint_a=True)
#     P = tf.matmul(weight_P,tf.linalg.inv(B))
#     V = list()
#     for user in range(K):
#             V_temp = P[:,:,user] / tf.tile(tf.reshape(tf.sqrt(q_list[:,user]),[-1,1]),(1,Nt))
#             V_temp = tf.tile(tf.reshape(tf.sqrt(p_list[:,user]),[-1,1]),(1,Nt))*V_temp/tf.tile(tf.reshape(tf.norm(V_temp,axis = 1),[-1,1]),(1,Nt))
#             V.append(V_temp)
#     V = tf.stack(V,2) #(B,Nt,K)
#     V = tf.math.conj(tf.reshape(V,(-1,Nt,1,K,1)))
#     V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
#     return V


def restore(channel, prediction):
    p_list_pred = prediction[:, :K * dk]
    q_list_pred = prediction[:, K * dk:2 * K * dk]
    V_restore = pq2V_low_complexity_for_restore(channel,Nt=Nt,Nr=1,dk=1,K=K*dk,p=1,sigma_2=sigma_2,q_list=q_list_pred,p_list=p_list_pred)
    V_restore = tf.reshape(V_restore, (-1, Nt, dk, K, 2))
    return V_restore


# class MIMO2MISO(tf.keras.layers.Layer):
#     def __init__(self):
#         super(MIMO2MISO, self).__init__()
#     def call(self, channel):
#         channel = tf.reshape(channel, (-1, Nt, Nr, K * 2))
#         channel = channel[:, :, 0, :K] + 1j * channel[:, :, 0, K:]
#         H = tf.cast(tf.transpose(channel, [0, 2, 1, 3]), tf.complex128)
#         P = list()
#         for user in range(K):
#             H_this_user = H[:, :, :, user]
#             s, _, v = tf.linalg.svd(H_this_user)
#             v = tf.matmul(v, tf.cast(tf.linalg.diag(s), tf.complex128))
#             P.append((tf.math.conj(v[:, :, :dk])))
#         P = tf.stack(P, axis=3)
#         P = tf.reshape(P, [-1, Nt, 1, K * dk])
#         H_ensemble = tf.reshape(tf.transpose(P, (0, 3, 2, 1)), (-1, K * dk * 1, Nt))
#         H_bar = tf.matmul(H_ensemble, H_ensemble, adjoint_b=True)
#         H_miso_bar = np.triu(np.real(H_bar)) + np.tril(np.imag(H_bar))
#         return H_miso_bar
def MIMO2MISO(channel):
    channel = tf.cast(tf.reshape(channel, (-1, Nt, Nr, K * 2)),tf.complex128)
    channel = channel[:, :, :, :K] + 1j * channel[:, :, :, K:]
    H = tf.cast(tf.transpose(channel, [0, 2, 1, 3]), tf.complex128)
    P = list()
    for user in range(K):
        H_this_user = H[:, :, :, user]
        s, _, v = tf.linalg.svd(H_this_user)
        v = tf.matmul(v, tf.cast(tf.linalg.diag(s), tf.complex128))
        P.append((tf.math.conj(v[:, :, :dk])))
    P = tf.stack(P, axis=3)
    P = tf.reshape(P, [-1, Nt, 1, K * dk])
    H_ensemble = tf.reshape(tf.transpose(P, (0, 3, 2, 1)), (-1, K * dk * 1, Nt))
    H_bar = tf.matmul(H_ensemble, H_ensemble, adjoint_b=True)
    H_miso_bar = tf.linalg.band_part(tf.math.real(H_bar),-1,0) + tf.linalg.band_part(tf.math.imag(H_bar),0,-1)
    #H_miso_bar = tf.experimental.numpy.triu(tf.math.real(H_bar)) + tf.experimental.numpy.tril(tf.math.imag(H_bar))
    P = tf.cast(tf.concat([tf.math.real(P), tf.math.imag(P)], axis=3), dtype=tf.float32)
    #H_miso_bar = np.triu(np.real(H_bar)) + np.tril(np.imag(H_bar))
    return P,H_miso_bar




import time

channel = test_H
channel_noiseless = test_H_noiseless
model.load_weights(un_model_path)

#t = time.time()
model_output = model.predict(test_dataset_bar)
lcp_run_time = []

#print(time.time() - t)

def total_net(Nt,Nr, K, dk, lr,origin_model):
    data = Input(shape=(K*Nt*Nr*2))
    data_miso,data_bar = Lambda(MIMO2MISO)(data)
    prediction = backbone(data_bar,Nt,Nr,dk,K)
    precoding_matrix = pq2V_low_complexity_for_restore()(data_miso,Nt=Nt,Nr=1,dk=1,K=K*dk,p=p,sigma_2=sigma_2,prediction=prediction)
    model = Model(inputs=data, outputs=precoding_matrix)
    model.compile(loss=DUU_EZF_loss, optimizer=Adam(lr=lr))
    model.summary()

    return model
@tf.function
def serve(x):
  return total_model(x, training=False)

total_model = total_net(Nt,Nr, K, dk, lr,model)
model_output = serve(test_H[0:1,:])
#model_output = total_model.predict(test_H)
test_num = 50
#test_dataset_bar = MIMO2MISO(test_H)
#model_output = serve(test_dataset_bar)
# model.call = tf.function(model.call)

t = time.time()
for i in range(test_num):
    model_output = serve(test_H[i:i+1,:])
    #model_output = total_model(test_H[i:i+1,:],training=False)
lcp_run_time=time.time() - t
#print(time.time() - t)

#tf_dataset = tf.data.Dataset.from_tensor_slices(test_dataset_bar)

# DUU_EZF_output = restore(test_dataset, model_output)
#
# DUU_EZF_performance = np.mean(minus_sum_rate_loss(channel_noiseless, DUU_EZF_output))
#
# t = time.time()
# DUU_MISO_output = restore(test_dataset, test_labelset_su)
# lcp_run_time.append(time.time() - t)
#
#
# t = time.time()
# DUU_MISO_output = restore(test_dataset, test_labelset_su)
# lcp_run_time.append(time.time() - t)
# #print(time.time() - t)
# DUU_MISO_performance = np.mean(minus_sum_rate_loss(channel_noiseless, DUU_MISO_output))
@tf.function
def EZF(channel):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = H[:, :, :, :K] + 1j * H[:, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3])
    P = list()
    for user in range(K):
        H_this_user = H[:, :, :, user]
        _, _, v = tf.linalg.svd(H_this_user)
        P.append(v[:, :, :dk])
    P = tf.stack(P, axis=3)
    P = tf.reshape(P, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    V = tf.matmul(P, tf.linalg.inv(tf.matmul(tf.transpose(P, [0, 2, 1], conjugate=True), P)))  # B*Nt*Kdk
    V = tf.reshape(V, [-1, Nt, dk, K, 1])
    # import ipdb;ipdb.set_trace()
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V


def update_RMMWSE_U(H_bar, X):
    U_list = list()
    trace_Hbar_XX = 0
    for user_index in range(K):
        Xk = X[:, :, :, user_index]
        trace_Hbar_XX = trace_Hbar_XX + tf.linalg.trace(
            tf.matmul(H_bar, tf.matmul(Xk, tf.transpose(Xk, [0, 2, 1], conjugate=True))))
    for user_index in range(K):
        HXXH = tf.zeros([Nr, Nr], dtype=tf.complex128)
        Xk = X[:, :, :, user_index]
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        for i in range(K):
            Xi = X[:, :, :, i]  #
            HX = tf.matmul(H_bark, Xi)
            HXXH = HXXH + tf.matmul(HX, tf.transpose(HX, perm=[0, 2, 1], conjugate=True))
        U_this_user = tf.matmul(tf.matmul(tf.linalg.inv(
            sigma_2 / p * tf.tile(tf.reshape(trace_Hbar_XX, (-1, 1, 1)), [1, Nr, Nr]) * tf.eye(Nr,
                                                                                               dtype=tf.complex128) + HXXH),
                                          H_bark), Xk)
        U_list.append(U_this_user)
    U = tf.stack(U_list, 3)  # B*Nr*K
    return U


def update_RMMWSE_W(H_bar, X, U):
    W_list = list()
    for user_index in range(K):
        Uk = U[:, :, :, user_index]
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        Xk = X[:, :, :, user_index]
        W_this_user = tf.linalg.inv(tf.eye(dk, dtype=tf.complex128) - tf.matmul(
            tf.matmul(tf.transpose(Uk, perm=[0, 2, 1], conjugate=True), H_bark), Xk))
        W_list.append(W_this_user)
    W = tf.stack(W_list, 3)
    return W


def update_RMMWSE_X(H_bar, U, W):
    X_list = list()
    B = H_bar * 0
    for user_index in range(K):
        Uk = U[:, :, :, user_index]
        Wk = W[:, :, :, user_index]
        Mk = tf.matmul(tf.matmul(Uk, Wk), tf.transpose(Uk, perm=[0, 2, 1], conjugate=True))
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        B = B + sigma_2 / p * tf.tile(tf.reshape(tf.linalg.trace(Mk), (-1, 1, 1)),
                                      [1, K * Nr, K * Nr]) * H_bar + tf.matmul(
            tf.matmul(tf.transpose(H_bark, perm=[0, 2, 1], conjugate=True), Mk), H_bark)
    B_inverse = tf.linalg.inv(B)
    for user_index in range(K):
        Uk = U[:, :, :, user_index]
        Wk = W[:, :, :, user_index]
        H_bark = H_bar[:, user_index * Nr:(user_index + 1) * Nr, :]
        HUWk = tf.matmul(tf.matmul(tf.transpose(H_bark, perm=[0, 2, 1], conjugate=True), Uk), Wk)
        X_this_user = tf.matmul(B_inverse, HUWk)
        X_list.append(X_this_user)
    X = tf.stack(X_list, 3)
    return X


def tf_pinv(a, rcond=None):
    """Taken from
    https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/ops/linalg/linalg_impl.py
    """
    dtype = a.dtype.as_numpy_dtype

    if rcond is None:
        def get_dim_size(dim):
            dim_val = a.shape[dim]
            if dim_val is not None:
                return dim_val
            return tf.shape(a)[dim]

        num_rows = get_dim_size(-2)
        num_cols = get_dim_size(-1)
        if isinstance(num_rows, int) and isinstance(num_cols, int):
            max_rows_cols = float(max(num_rows, num_cols))
        else:
            max_rows_cols = tf.cast(tf.maximum(num_rows, num_cols), dtype)
        rcond = 10. * max_rows_cols * np.finfo(dtype).eps

    rcond = tf.convert_to_tensor(rcond, dtype=dtype, name='rcond')
    # Calculate pseudo inverse via SVD.
    # Note: if a is Hermitian then u == v. (We might observe additional
    # performance by explicitly setting `v = u` in such cases.)
    [
        singular_values,  # Sigma
        left_singular_vectors,  # U
        right_singular_vectors,  # V
    ] = tf.linalg.svd(
        a, full_matrices=False, compute_uv=True)

    # Saturate small singular values to inf. This has the effect of make
    # `1. / s = 0.` while not resulting in `NaN` gradients.
    cutoff = tf.cast(rcond, dtype=singular_values.dtype) * tf.reduce_max(singular_values, axis=-1)
    singular_values = tf.where(
        singular_values > cutoff[..., None], singular_values,
        np.array(np.inf, dtype))

    # By the definition of the SVD, `a == u @ s @ v^H`, and the pseudo-inverse
    # is defined as `pinv(a) == v @ inv(s) @ u^H`.
    a_pinv = tf.matmul(
        right_singular_vectors / tf.cast(singular_values[..., None, :], dtype=dtype),
        left_singular_vectors,
        adjoint_b=True)

    if a.shape is not None and a.shape.rank is not None:
        a_pinv.set_shape(a.shape[:-2].concatenate([a.shape[-1], a.shape[-2]]))

    return a_pinv


def WMMSE(channel):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, 2 * K]), tf.complex128)
    H = H[:, :, :, :K] + 1j * H[:, :, :, K:]  # B*Nt*Nr*K
    H_ensemble = tf.reshape(tf.transpose(H, [0, 3, 2, 1]), [-1, K * Nr, Nt])  # B*KNr*Nt
    H_bar = tf.matmul(H_ensemble, tf.transpose(H_ensemble, [0, 2, 1], conjugate=True))
    V = EZF(channel)
    V = tf.cast(V, tf.complex128)
    V = V[:, :, :, :, 0] + 1j * V[:, :, :, :, 1]  # B*Nt*dk*K
    X_list = []
    # import ipdb;ipdb.set_trace()
    H_ensemble_pinv = tf_pinv(tf.transpose(H_ensemble, [0, 2, 1], conjugate=True))
    for user_index in range(K):
        Vk = V[:, :, :, user_index]
        X_list.append(tf.matmul(H_ensemble_pinv, Vk))
    X = tf.stack(X_list, 3)  # B*KNr*dk*K
    U = update_RMMWSE_U(H_bar, X)
    W = update_RMMWSE_W(H_bar, X, U)
    for i in range(50):
        new_X = update_RMMWSE_X(H_bar, U, W)
        new_U = update_RMMWSE_U(H_bar, new_X)
        new_W = update_RMMWSE_W(H_bar, new_X, new_U)
        if tf.norm(new_W - W, 2).numpy() < 0.0001:
            break
        U = new_U
        W = new_W
        X = new_X
    V_list = list()
    for user_index in range(K):
        Xk = X[:, :, :, user_index]
        Vk = tf.matmul(tf.transpose(H_ensemble, [0, 2, 1], conjugate=True), Xk)
        V_list.append(Vk)
    V = tf.stack(V_list, 3)
    V = tf.reshape(V, [-1, Nt, dk, K, 1])
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)

    U = np.reshape(U, (-1, K * Nr * dk))
    X = np.reshape(X, (-1, K * Nr * dk * K))
    #    print(W_list)

    W = np.triu(np.real(W)) + np.tril(np.imag(W))
    W = np.reshape(W, (-1, K * dk * dk))
    return U, W, V, X



EZF_output = EZF(channel[i:i+1,:])
t = time.time()
for i in range(test_num):
    EZF_output = EZF(channel[i:i+1,:])
ezf_run_time = time.time() - t
EZF_performance = np.mean(minus_sum_rate_loss(channel_noiseless, EZF_output))
_, _, WMMSE_output, _ = WMMSE(channel[i:i+1,:])
t = time.time()
for i in range(test_num):
    print(i)
    _, _, WMMSE_output, _ = WMMSE(channel[i:i+1,:])
wmmse_run_time = time.time() - t
WMMSE_performance = np.mean(minus_sum_rate_loss(channel_noiseless, WMMSE_output))

import logging

logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(dataset_root + 'result/execution_time_test.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('%s_Nt_%d_Nr_%d_K_%d_dk_%d_SNR_%d_SNR_channel_%d'%(prefix,Nt, Nr, K, dk, SNR_dB,SNR_channel_dB))
logger.info('EZF running time:%.5f' % np.mean(ezf_run_time))
logger.info('LCP running time:%.5f' % np.mean(lcp_run_time))
#logger.info(' sum rate:%.5f' % (np.mean(DUU_MISO_performance)))
logger.info('WMMSE running time:%.5f' % np.mean(wmmse_run_time))
# print('EZF sum rate:%.5f'%np.mean(EZF_performance))
# print('DUU EZF sum rate:%.5f'%np.mean(DUU_EZF_performance))
# print('RWMMSE sum rate:%.5f'%np.mean(RWMMSE_performance))


# CUDA_VISIBLE_DEVICES=0 python learn_from_bar.py  --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2
