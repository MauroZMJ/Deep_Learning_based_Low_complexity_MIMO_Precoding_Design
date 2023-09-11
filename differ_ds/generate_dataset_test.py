import numpy as np
import os
np.random.seed(2020)
import tensorflow as tf

from scipy import io
# import os
import os
import hdf5storage
import random
from sklearn import preprocessing
import logging

#tf.config.run_functions_eagerly(True)

# %% load and construct data
from option import parge_config

args = parge_config()
Nt = args.Nt
Nr = args.Nr
K = args.K
B = args.B
dk = args.dk
SNR_dB = args.SNR
SNR = 10 ** (SNR_dB / 10)
p = 1
sigma_2 = 1 / SNR
SNR_channel_dB = args.SNR_channel
data_mode = args.mode

mode = 'train'
data_mode = 'debug'

dataset_root = '/data/zhangmaojun/dataset/'
data_root = dataset_root + 'channel_dataset.mat'
if data_mode =='debug':
    channels = hdf5storage.loadmat(data_root)['H_list'][2000:4000, :, :, :K, :B]
else:
    channels = hdf5storage.loadmat(data_root)['H_list'][:, :, :, :K, :B]
channels = np.transpose(channels,(0,1,2,4,3))
model_root = './model/'
factor = 1
test_batch = 200
ds_list = np.ones((channels.shape[0],K))
for data_index in range(channels.shape[0]//test_batch):
    ds_this_list = np.ones((K,1))
    total_stream = K*dk - K
    user_dex = 0
    while total_stream>0:
        d_extra = np.min((np.random.randint(Nr),total_stream))
        while ds_this_list[user_dex,0] + d_extra > Nr:
            user_dex = (user_dex+1)%K      
        ds_this_list[user_dex,0] += d_extra
        user_dex = (user_dex+1)%K 
        total_stream = total_stream - d_extra
    ds_list[data_index*200 : (data_index+1)*200,:] = np.repeat(ds_this_list,test_batch,axis = 1).T
#import ipdb;ipdb.set_trace()
def minus_sum_rate_loss(y_true, y_pred,Nt,Nr,dk,K,B,p,sigma_2):
    '''
    y_true is the channels
    y_pred is the predicted beamformers
    notice that, y_true has to be the same shape as y_pred
    '''
    # V shape:Nt*(K*dk)*2
    ## construct complex data  channel shape:Nt,Nr,2*K   y_pred shape:Nt,dk,K,2
    ds_list = tf.cast(y_true[0,-K:],tf.int32)
    y_true = y_true[:,:-K]
    y_true = tf.cast(tf.reshape(y_true, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = y_true[:, :, :, :, :K] + 1j * y_true[:, :, :, :, K:]
    #where the V should be a list of the beamform vector of all users
    V0 = y_pred
    V = list()
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[user], tf.transpose(V0[user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    #import ipdb;ipdb.set_trace()
    for user in range(K):
        energy_scale_temp = tf.tile(tf.reshape(energy_scale, (-1, 1, 1)), (1, Nt, ds_list[user]))
        V.append(V0[user] * tf.cast(energy_scale_temp, tf.complex128))
    sum_rate = 0.0
    # import ipdb;ipdb.set_trace()
    for k in range(K):
        for rb in range(B):
            H_k = tf.transpose(H[:, :, :, rb, k], perm=[0, 2, 1])  # NrxNt
            V_k = V[k]  # Ntx1
            signal_k = tf.matmul(H_k, V_k)
            signal_k_energy = tf.matmul(signal_k, tf.transpose(signal_k, perm=[0, 2, 1], conjugate=True))
            interference_k_energy = 0.0
            for j in range(K):
                if j != k:
                    V_j = V[j]
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
    loss = sum_rate
    return loss
def EZF(channel,Nt,Nr,dk,K,B,p,sigma_2,P_return=False):
    ds_list = tf.cast(channel[0,-K:],tf.int32)
    channel = channel[:,:-K]
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B,2 * K]), tf.complex128)
    H = H[:, :, :, :,:K] + 1j * H[:, :, :,:, K:]
    H = tf.transpose(H, [0, 2, 1, 3, 4])
    P = list()
    for user in range(K):
        H_this_user = tf.matmul(tf.transpose(H[:, :, :, 0, user], [0, 2, 1], conjugate=True), H[:, :, :, 0, user])
        for rb in range(1,B):
            H_this_user = H_this_user + tf.matmul(tf.transpose(H[:, :, :, rb, user], [0, 2, 1], conjugate=True), H[:, :, :, rb, user])
        _, _, v = tf.linalg.svd(H_this_user)
        P.append(v[:, :, :ds_list[user]])
    #P = tf.stack(P, axis=3)
    P = tf.concat([vi for vi in P],axis=2)
    P = tf.reshape(P, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    V = tf.matmul(P, tf.linalg.inv(tf.matmul(tf.transpose(P, [0, 2, 1], conjugate=True), P)))  # B*Nt*Kdk
    V = tf.reshape(V, [-1, Nt, dk*K])
    V0 = list()
    ds_acc = 0
    for user in range(K):
        V0.append(V[:,:,ds_acc:ds_acc+ds_list[user]])
        ds_acc += ds_list[user]

    V = list()
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[user], tf.transpose(V0[user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    #import ipdb;ipdb.set_trace()
    for user in range(K):
        energy_scale_temp = tf.tile(tf.reshape(energy_scale, (-1, 1, 1)), (1, Nt, ds_list[user]))
        V.append(V0[user] * tf.cast(energy_scale_temp, tf.complex128))
    # import ipdb;ipdb.set_trace()
    if P_return:
        P = tf.reshape(P, [-1, Nt, 1, K * dk])
        P = tf.cast(tf.concat([tf.math.real(P), tf.math.imag(P)], axis=3), dtype=tf.float32)
        return V,P
    else:
        return V
def MIMO2MISO(channel,Nt,Nr,dk,K,B):
    ds_list = tf.cast(channel[0,-K:],tf.int32)
    channel = channel[:,:-K]
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3,4])
    P = list()
    for user in range(K):
        P_this_user_list = list()
        for rb in range(B):
            H_this_user = H[:, :, :, rb, user]
            s, _, v = tf.linalg.svd(H_this_user)
            v = tf.matmul(v,tf.cast(tf.linalg.diag(s),tf.complex128))
            P_this_user_list.append((tf.math.conj(v[:, :, :ds_list[user]])))
        P.append(tf.stack(P_this_user_list,axis=2))
    P = tf.concat([vi for vi in P],axis=3)
    P = tf.reshape(P,[-1,Nt,1,B,K*dk])
    P = tf.cast(tf.concat([tf.math.real(P), tf.math.imag(P)], axis=4), dtype=tf.float32)
    return P

def MMSE_DUAL_PA(channel,Nt,Nr,dk,K,B,p,sigma_2,V_wmmse,U_wmmse,W_wmmse):
    #sinr_list shape: (B,K*dk)
    #channel shape:complex (B,Nt,1,K)
    lambda_list = list()
    q_list = list()
    pa_list = list()
    for user in range(K):
        pa_list.append(tf.norm(tf.reshape(V_wmmse[user],[channel.shape[0],-1]),axis=1)**2)
    pa_list = tf.stack(pa_list,axis=1)
    for user in range(K):
        for rb in range(B):
            q_temp = tf.matmul(U_wmmse[user][rb], W_wmmse[user][rb])
            lambda_temp = tf.matmul(q_temp,U_wmmse[user][rb],adjoint_b = True)
            lambda_list.append(lambda_temp)
            q_list.append(q_temp)

    lambda_list = tf.reshape(tf.stack(lambda_list,1),[-1,K,B])
    lambda_list = p * lambda_list / tf.tile(tf.cast(tf.reshape(tf.norm(tf.reshape(lambda_list,[-1,K*B]),ord=1,axis=1),[-1,1,1]),tf.complex128),(1,K,B))
    q_list = tf.reshape(tf.stack(q_list,1),[-1,K,B])
    q_list = p * q_list / tf.tile(tf.cast(tf.reshape(tf.norm(tf.reshape(q_list,[-1,K*B]),ord=2,axis=1),[-1,1,1]),tf.complex128),(1,K,B))
    lambda_list = tf.cast(lambda_list,dtype=tf.float32)
    pa_list = tf.cast(pa_list,dtype=tf.float32)
    q_list = tf.reshape(q_list,[-1,K,B,1])
    q_list = tf.cast(tf.concat([tf.math.real(q_list),tf.math.imag(q_list)],axis=3),dtype=tf.float32)

    return lambda_list,pa_list,q_list

def pq2V(channel_ds,Nt,Nr,dk,K,B,p,sigma_2,lambda_list,pa_list,q_list):
    ds_list = tf.cast(channel_ds[0,-K:],tf.int32)
    channel = channel_ds[:,:-K]
    channel = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    # recover
    channel = channel[:,:,:,:,:K] + 1j*channel[:,:,:,:,K:]  #new channel shape:(B,Nt,K)
    channel = tf.transpose(channel,[0,2,1,3,4])


    lambda_list = tf.cast(lambda_list,dtype=tf.complex128)


    pa_list = tf.cast(pa_list,dtype=tf.complex128)
    q_list = tf.cast(q_list,dtype=tf.complex128)
    q_list = q_list[:,:,:,0] + 1j*q_list[:,:,:,1]
    temp_inverse = sigma_2 * tf.eye(Nt, dtype=tf.complex128)
    for user in range(K):
        for rb in range(B):
            temp_inverse = temp_inverse + tf.tile(tf.reshape(lambda_list[:,user,rb],[-1,1,1]),[1,Nt,Nt])*tf.matmul(channel[:,:,:,rb,user],channel[:,:,:,rb,user],adjoint_a=True)
    temp_inverse = tf.linalg.inv(temp_inverse)

    V_norm = list()
    for user in range(K):
        HUW_this_user = tf.zeros([Nt,dk],dtype=tf.complex128)
        for rb in range(B):
            HUW_this_user = HUW_this_user + tf.tile(tf.reshape(q_list[:,user,rb],[-1,1,1]),[1,Nt,Nr])*tf.transpose(channel[:,:,:,rb,user],[0,2,1],conjugate=True)
        V_temp = tf.matmul(temp_inverse,HUW_this_user)
        V_temp, _ = tf.linalg.normalize(V_temp[:, :, 0] + 1e-16, axis=1)
        V_norm.append(V_temp)
    V_norm = tf.stack(V_norm,2)
    V = list()
    for k in range(K):
        V_temp = tf.tile(tf.reshape(tf.sqrt(pa_list[:,k]),(-1,1)),(1,Nt)) * V_norm[:,:,k]
        V.append(V_temp)
    V = tf.stack(V,2) #(B,Nt,K)
    V = tf.reshape(V,(-1,Nt,1,K,1))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V


def WMMSE(channel_ds,Nt,Nr,dk,K,B,p,sigma_2,UW_return):
    def update_WMMSE_U(H,V):
        U = list()
        trace_VV = 0
        for user in range(K):
            trace_VV = trace_VV + tf.linalg.trace(
                tf.matmul(V[user], V[user],adjoint_b=True))
        for user in range(K):
            U_this_user_list = list()
            for rb in range(B):
                HVVH = tf.zeros([Nr, Nr], dtype=tf.complex128)
                for k in range(K):
                    HV = tf.matmul(H[:, :, :, rb, user], V[k])
                    HVVH = HVVH + tf.matmul(HV, HV,adjoint_b=True)
                inverse_temp = tf.linalg.inv(sigma_2 / p * tf.tile(tf.reshape(trace_VV, (-1, 1, 1)), [1, Nr, Nr]) * tf.eye(Nr,dtype=tf.complex128) + HVVH)
                U_this_user = tf.matmul(tf.matmul(inverse_temp,H[:, :, :, rb,user]), V[user])
                U_this_user_list.append(U_this_user)
            U.append(U_this_user_list)
        return U
    def update_WMMSE_W(H,U,V):
        W = list()
        for user in range(K):
            W_this_user_list = list()
            for rb in range(B):
                HV = tf.matmul(H[:, :, :, rb, user], V[user])
                W_this_user = tf.linalg.inv(tf.eye(V[user].shape[2], dtype=tf.complex128) - tf.matmul(
                    U[user][rb], HV,adjoint_a=True))
                W_this_user_list.append(W_this_user)
            W.append(W_this_user_list)
        return W
    def update_WMMSE_V(H,U,W):
        temp_B = tf.zeros([Nt, Nt], dtype=tf.complex128)
        for user in range(K):
            for rb in range(B):
                HHU = tf.matmul(H[:, :, :, rb, user], U[user][rb],adjoint_a=True)  # b*Nt*dk
                trace_UWU = sigma_2 / p * tf.linalg.trace(tf.matmul(tf.matmul(U[user][rb], W[user][rb]),
                                                                    U[user][rb],adjoint_b=True))
                temp_B = temp_B + tf.tile(tf.reshape(trace_UWU, (-1, 1, 1)), [1, Nt, Nt]) * tf.eye(Nt,
                                                                                                   dtype=tf.complex128) + tf.matmul(
                    tf.matmul(HHU, W[user][rb]), HHU,adjoint_b=True)

        temp_B_inverse = tf.linalg.inv(temp_B)

        V0 = list()
        VV = tf.zeros([batch_size, Nt, Nt], dtype=tf.complex128)

        for user in range(K):
            HUW = tf.matmul(tf.matmul(H[:, :, :, 0, user], U[user][0],adjoint_a=True),W[user][0])
            for rb in range(1,B):
                HUW = HUW + tf.matmul(tf.matmul(H[:, :, :, rb, user], U[user][rb],adjoint_a=True),W[user][rb])  # b*Nt*dk
            V0_this_user = tf.matmul(temp_B_inverse, HUW)
            V0.append(V0_this_user)

        V = list()
        trace_VV = 0
        for user in range(K):
            trace_VV = trace_VV + tf.linalg.trace(
                tf.matmul(V0[user], tf.transpose(V0[user], perm=[0, 2, 1], conjugate=True)))
        energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
        #import ipdb;ipdb.set_trace()
        for user in range(K):
            energy_scale_temp = tf.tile(tf.reshape(energy_scale, (-1, 1, 1)), (1, Nt, ds_list[user]))
            V.append(V0[user] * tf.cast(energy_scale_temp, tf.complex128))
        return V
    ds_list = tf.cast(channel_ds[0,-K:],tf.int32)
    channel = channel_ds[:,:-K]
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]  # B*Nt*Nr*K
    H = tf.transpose(H,(0,2,1,3,4))
    V = EZF(channel_ds, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B, p=p, sigma_2=sigma_2)
    for i in range(50):
        U = update_WMMSE_U(H,V)
        W = update_WMMSE_W(H,U,V)
        V = update_WMMSE_V(H,U,W)
    V0 = V
    V = list()
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[user], tf.transpose(V0[user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    #import ipdb;ipdb.set_trace()
    for user in range(K):
        energy_scale_temp = tf.tile(tf.reshape(energy_scale, (-1, 1, 1)), (1, Nt, ds_list[user]))
        V.append(V0[user] * tf.cast(energy_scale_temp, tf.complex128))
    if UW_return:
        return U,W,V
    else:
        return V
def reconstruct_V(V_restore_miso,ds_list):
    V_restore_miso = tf.cast(V_restore_miso,tf.complex128)
    V_restore_miso = V_restore_miso[:,:,:,:,0] + 1j*V_restore_miso[:,:,:,:,1]
    V = list()
    ds_list = tf.cast(ds_list[0,:],tf.int32)
    ds_acc = 0
    for ds in ds_list:
        V.append(V_restore_miso[:,:,0,ds_acc:ds_acc+ds])
        ds_acc +=ds
    return V
        
data_num = len(channels)
#dataset = dataset[:data_num]
channels = channels[:data_num]
labelset = np.zeros((data_num, 2*dk*K))

transmit_pa = np.zeros((data_num, dk * K))
upload_pa = np.zeros((data_num, dk * K * B))
rb_allocate_vec = np.zeros((data_num, dk * K * B*2))
#channels_miso = np.zeros((data_num,Nt,1,B,2*K*dk))

init_rate_list = []
final_rate_list = []

batch_size = test_batch
total_iter = len(channels) // batch_size
# import ipdb;ipdb.set_trace()
EZF_performance = []
WMMSE_performance = []
DUU_MISO_performance = []
for i in range(total_iter):
    print('iteration:' + str(i))
    channel_iter = channels[i * batch_size:(i + 1) * batch_size, :]
    import ipdb;ipdb.set_trace()
    ds_iter = ds_list[i * batch_size:(i + 1) * batch_size, :]
    ds_iter = np.repeat(np.array([[4,3,1,3,4,1,1,1,1,1]]).T,test_batch,axis = 1).T
    channel_iter = np.concatenate([np.reshape(np.concatenate([np.real(channel_iter), np.imag(channel_iter)], axis=-1),(channel_iter.shape[0],-1)),ds_iter],axis = 1)
    EZF_output = EZF(channel_iter,Nt=Nt,Nr=Nr,dk=dk,K=K,B=B,p=p,sigma_2=sigma_2)
    EZF_performance.append(np.mean(minus_sum_rate_loss(channel_iter, EZF_output,Nt,Nr,dk,K,B,p,sigma_2))/B)
    WMMSE_output = WMMSE(channel_iter,Nt=Nt,Nr = Nr,dk=dk,K=K,B=B,p=p,sigma_2=sigma_2,UW_return=False)
    WMMSE_performance.append(np.mean(minus_sum_rate_loss(channel_iter, WMMSE_output,Nt,Nr,dk,K,B,p,sigma_2))/B)

    channel_iter_miso = MIMO2MISO(channel_iter,Nt=Nt,Nr=Nr,dk=dk,K=K,B=B)
    ds_iter_miso = np.ones((batch_size,K*dk))
    channel_iter_miso = np.concatenate([np.reshape(channel_iter_miso,(channel_iter_miso.shape[0],-1)),ds_iter_miso],axis=1)
    U_iter_miso, W_iter_miso , V_iter_miso = WMMSE(channel_iter_miso,Nt=Nt,Nr=1,dk=1,K=K*dk,B=B,p = p,sigma_2=sigma_2,UW_return=True)

    lambda_list,pa_list,q_list = MMSE_DUAL_PA(channel_iter_miso,Nt=Nt,Nr=1,dk=1,K = K*dk,B=B,p=p,sigma_2=sigma_2,
                                                  V_wmmse=V_iter_miso,U_wmmse = U_iter_miso,W_wmmse=W_iter_miso)
    '''restore way'''
    V_restore_miso = pq2V(channel_iter_miso, Nt, 1, 1, K * dk,B, p, sigma_2, lambda_list, pa_list,q_list)
    V_dual_pa = reconstruct_V(V_restore_miso,ds_iter)
    #import ipdb;ipdb.set_trace()
    DUU_MISO_performance.append(np.mean(minus_sum_rate_loss(channel_iter, V_dual_pa,Nt,Nr,dk,K,B,p,sigma_2))/B)
    # import ipdb;ipdb.set_trace()

    #import ipdb;ipdb.set_trace()
    #RWMMSE_performance.append(np.mean(minus_sum_rate_loss(channel_iter, V_iter,Nt,Nr,dk,K,p,sigma_2)))
    # import ipdb;ipdb.set_trace()
    lambda_list = tf.reshape(lambda_list,[-1,K*B*dk])
    q_list = tf.reshape(q_list,[-1,K*B*2*dk])
    #channels_miso[i * 1000:(i + 1) * 1000, :] = channel_iter_miso.numpy()
    transmit_pa[i * batch_size:(i + 1) * batch_size, :] = pa_list.numpy()
    upload_pa[i * batch_size:(i + 1) * batch_size, :] = lambda_list.numpy()
    rb_allocate_vec[i * batch_size:(i + 1) * batch_size, :] = q_list.numpy()

data_save_root = dataset_root + 'data/DUU_MISO_dataset_differ_ds_%d_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk,B, SNR_dB)
hdf5storage.savemat(data_save_root,
           {'transmit_power_allocation': transmit_pa, 'upload_power_allocation': upload_pa, 'resource_block_allocation':rb_allocate_vec,
            'H': channels,'ds_list':ds_list})


logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(dataset_root + 'data/DUU_MISO_dataset_differ_ds_%d_%d_%d_%d_%d_%d.log' % (Nt, Nr, K, dk,B, SNR_dB))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('EZF sum rate:%.5f' % np.mean(EZF_performance))
logger.info('DUU MISO sum rate:%.5f' % np.mean(DUU_MISO_performance))
logger.info('WMMSE sum rate:%.5f' % np.mean(WMMSE_performance))
#logger.info('RWMMSE sum rate:%.5f' % np.mean(RWMMSE_performance))

# python generate_DUU_EZF.py --Nt 64 --Nr 4 --dk 4 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 1000 --epoch 1000 --factor 2
#python generate_dataset.py --Nt 64 --Nr 4 --K 10 --dk 2 --B 4 --SNR 0 --SNR_channel 100 --gpu 0 --mode debug --batch_size 200 --epoch 1000 --factor 1