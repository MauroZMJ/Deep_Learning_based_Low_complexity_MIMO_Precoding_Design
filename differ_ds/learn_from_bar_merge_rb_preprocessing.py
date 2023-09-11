import numpy as np
import tensorflow as tf
from scipy import io
import hdf5storage
np.random.seed(2021)

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
    #P = tf.reshape(tf.transpose(tf.reshape(P, [-1, K,B,Nt, dk]),[0,3,2,4,1]),[-1,Nt,1,B,K*dk])
    H_ensemble = tf.reshape(tf.transpose(P,(0,3,4,1,2)),(-1,B*K*dk*1,Nt))
    H_bar = tf.matmul(H_ensemble, H_ensemble, adjoint_b=True)
    H_miso_bar = np.triu(np.real(H_bar)) + np.tril(np.imag(H_bar))
    H_miso_bar = tf.transpose(tf.reshape(H_miso_bar,[-1,1,B*K*dk,B*K*dk]),[0,2,3,1])
    P = tf.cast(tf.concat([tf.math.real(P), tf.math.imag(P)], axis=4), dtype=tf.float32)
    return P,H_miso_bar
def data_process(data_root,Nt,Nr,dk,K,B,SNR_dB,SNR_channel_dB,test_length,data_mode):
    total_dataset = hdf5storage.loadmat(data_root)
    if data_mode == 'debug':
        data_length = 4000
    else:
        data_length = len(total_dataset['H'])
    H = total_dataset['H'][:data_length, :]
    data_stream_list = total_dataset['ds_list'][:data_length, :]
    transmit_pa = total_dataset['transmit_power_allocation'][:data_length, :]
    upload_pa = total_dataset['upload_power_allocation'][:data_length, :]
    rb_allocate_vec = total_dataset['resource_block_allocation'][:data_length, :]
    labelset_su = np.concatenate((transmit_pa, upload_pa,rb_allocate_vec), axis=-1)
    print(labelset_su.shape)
    print(H.shape)
    data_num = len(H)
    SNR_channel = 10 ** (SNR_channel_dB / 10)
    SNR = 10 ** (SNR_dB / 10)
    p = 1
    sigma_2 = 1 / SNR
    noise_energy = np.var(H) / SNR_channel
    channel_noise = np.sqrt(1 / 2 * noise_energy) * (
            np.random.randn(data_num, Nt, Nr, B, K) + 1j * np.random.randn(data_num, Nt, Nr, B, K))
    H_noiseless = H
    H = H_noiseless + channel_noise

    H_miso = np.zeros((data_num,Nt,1,B,2*K*dk))
    H_miso_bar = np.zeros((data_num,K*dk*B,K*dk*B,1))
    batch_size = 200
    total_iter = len(H_miso)//batch_size
    for i in range(total_iter):
        H_iter = H[i * batch_size:(i + 1) * batch_size, :]
        ds_iter = data_stream_list[i * batch_size:(i + 1) * batch_size, :]
        H_iter = np.concatenate([np.reshape(np.concatenate([np.real(H_iter), np.imag(H_iter)], axis=-1),(H_iter.shape[0],-1)),ds_iter],axis = 1)
        #H_iter = np.concatenate([np.real(H_iter), np.imag(H_iter)], axis=-1)
        
        H_miso_iter,H_miso_bar_iter = MIMO2MISO(channel=H_iter, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B)
        H_miso[i * batch_size:(i + 1) * batch_size, :] = H_miso_iter.numpy()
        H_miso_bar[i * batch_size:(i + 1) * batch_size, :] = H_miso_bar_iter
    H_miso = tf.reshape(H_miso,(data_num,-1))
    H_miso_bar = np.reshape(H_miso_bar,(data_num,-1))
    #H_miso = np.reshape(np.concatenate([np.real(H_miso), np.imag(H_miso)], axis=-1), (data_num, -1))

    #H_miso_bar = H2H_bar(H_miso, Nt, 1, 1, K=K * dk, data_num=data_num)
    #H_miso_bar = np.triu(np.real(H_miso_bar)) + np.tril(np.imag(H_miso_bar))

    dataset = H_miso

    test_dataset = dataset[-test_length:, :]
    dataset = dataset[:-test_length, :]

    dataset_bar = H_miso_bar
    test_dataset_bar = dataset_bar[-test_length:, :]
    dataset_bar = dataset_bar[:-test_length, :]

    H = np.reshape(np.concatenate([np.real(H), np.imag(H)], axis=-1), (data_num, -1))
    H_noiseless = np.reshape(np.concatenate([np.real(H_noiseless), np.imag(H_noiseless)], axis=-1), (data_num, -1))

    test_H = H[-test_length:, :]
    H = H[:-test_length, :]

    #H_noiseless = np.concatenate([H_noiseless,data_stream_list],axis=1)
    test_H_noiseless = H_noiseless[-test_length:, :]
    H_noiseless = H_noiseless[:-test_length, :]

    test_labelset_su = labelset_su[-test_length:, :]
    labelset_su = labelset_su[:-test_length, :]
    test_ds_list = data_stream_list[-test_length:, :]
    ds_list = data_stream_list[:-test_length, :]
    return dataset,test_dataset,H,test_H,H_noiseless,test_H_noiseless,labelset_su,test_labelset_su,dataset_bar,test_dataset_bar,ds_list,test_ds_list


