import numpy as np
import tensorflow as tf
from scipy import io
import hdf5storage
np.random.seed(2021)

def MIMO2MISO(channel,Nt,Nr,dk,K,B):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3,4])
    P = list()
    for user in range(K):
        for rb in range(B):
            H_this_user = H[:, :, :, rb, user]
            s, _, v = tf.linalg.svd(H_this_user)
            v = tf.matmul(v,tf.cast(tf.linalg.diag(s),tf.complex128))
            P.append((tf.math.conj(v[:, :, :dk])))
    P = tf.stack(P, axis=1)
    P = tf.reshape(tf.transpose(tf.reshape(P, [-1, K,B,Nt, dk]),[0,3,2,4,1]),[-1,Nt,1,B,K*dk])
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
    H = total_dataset['H'][-data_length:, :]
    transmit_pa = total_dataset['transmit_power_allocation'][-data_length:, :]
    upload_pa = total_dataset['upload_power_allocation'][-data_length:, :]
    rb_allocate_vec = total_dataset['resource_block_allocation'][-data_length:, :]
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
    total_iter = len(H_miso)//1000
    for i in range(total_iter):
        H_iter = H[i * 1000:(i + 1) * 1000, :]
        H_iter = np.concatenate([np.real(H_iter), np.imag(H_iter)], axis=-1)
        H_miso_iter,H_miso_bar_iter = MIMO2MISO(channel=H_iter, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B)
        H_miso[i * 1000:(i + 1) * 1000, :] = H_miso_iter.numpy()
        H_miso_bar[i * 1000:(i + 1) * 1000, :] = H_miso_bar_iter
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

    test_H_noiseless = H_noiseless[-test_length:, :]
    H_noiseless = H_noiseless[:-test_length, :]

    test_labelset_su = labelset_su[-test_length:, :]
    labelset_su = labelset_su[:-test_length, :]
    return dataset,test_dataset,H,test_H,H_noiseless,test_H_noiseless,labelset_su,test_labelset_su,dataset_bar,test_dataset_bar

from generate_dataset_mini import generate_dataset_custom_func
def data_process_fill_zero(test_channel,Nt,Nr,dk,K,B,SNR_dB,SNR_channel_dB,test_length,data_mode,add_user_num):
    data_length = len(test_channel)
    test_channel = np.reshape(test_channel,(data_length,Nt,Nr,B,-1))
    total_dataset = generate_dataset_custom_func(test_channel,Nt,Nr,dk,K-add_user_num,B,1,1/(10 ** (SNR_dB / 10)))
    # H:data*Nt*Nr*B*K
    H = total_dataset['H'][-data_length:, :]
    if add_user_num>0:
        H = np.concatenate([H,np.zeros((H.shape[0],H.shape[1],H.shape[2],H.shape[3],add_user_num))+1j*np.zeros((H.shape[0],H.shape[1],H.shape[2],H.shape[3],add_user_num))],axis = -1)
    K_origin = K - add_user_num
    # pa:data_num*K
    transmit_pa = total_dataset['transmit_power_allocation'][-data_length:, :]
    if add_user_num>0:
        transmit_pa = np.reshape(np.concatenate([np.reshape(transmit_pa,(-1,dk,K_origin)),np.zeros((transmit_pa.shape[0],dk,add_user_num))],axis = 2),(-1,dk*K))
    # lambda data_num*K*B
    upload_pa = total_dataset['upload_power_allocation'][-data_length:, :]
    if add_user_num>0:
        upload_pa = np.reshape(np.concatenate([np.reshape(upload_pa,(-1,dk,K_origin,B)),np.zeros((upload_pa.shape[0],dk,add_user_num,B))],axis = 2),(-1,(K_origin+add_user_num)*B*dk))
    #rb_allocate data_num*K*B*2
    rb_allocate_vec = total_dataset['resource_block_allocation'][-data_length:, :]
    if add_user_num>0:
        rb_allocate_vec = np.reshape(np.concatenate([np.reshape(rb_allocate_vec,(-1,dk,K_origin,B,2)),np.zeros((rb_allocate_vec.shape[0],dk,add_user_num,B,2))],axis = 2),(-1,(K_origin+add_user_num)*B*2*dk))
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
            np.random.randn(data_num, Nt, Nr, B, K-add_user_num) + 1j * np.random.randn(data_num, Nt, Nr, B, K-add_user_num))
    channel_noise = np.concatenate([channel_noise,np.zeros((channel_noise.shape[0],channel_noise.shape[1],channel_noise.shape[2],channel_noise.shape[3],add_user_num))+1j*np.zeros((channel_noise.shape[0],channel_noise.shape[1],channel_noise.shape[2],channel_noise.shape[3],add_user_num))],axis = -1)
    H_noiseless = H
    H = H_noiseless + channel_noise

    H_miso = np.zeros((data_num,Nt,1,B,2*K*dk))
    H_miso_bar = np.zeros((data_num,K*dk*B,K*dk*B,1))
    total_iter = len(H_miso)//1000
    for i in range(total_iter):
        H_iter = H[i * 1000:(i + 1) * 1000, :]
        H_iter = np.concatenate([np.real(H_iter), np.imag(H_iter)], axis=-1)
        H_miso_iter,H_miso_bar_iter = MIMO2MISO(channel=H_iter, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B)
        H_miso[i * 1000:(i + 1) * 1000, :] = H_miso_iter.numpy()
        H_miso_bar[i * 1000:(i + 1) * 1000, :] = H_miso_bar_iter
    H_miso = tf.reshape(H_miso,(data_num,-1))
    H_miso_bar = np.reshape(H_miso_bar,(data_num,-1))
    dataset = H_miso

    test_dataset = dataset


    dataset_bar = H_miso_bar
    test_dataset_bar = dataset_bar


    H = np.reshape(np.concatenate([np.real(H), np.imag(H)], axis=-1), (data_num, -1))
    H_noiseless = np.reshape(np.concatenate([np.real(H_noiseless), np.imag(H_noiseless)], axis=-1), (data_num, -1))

    test_H = H


    test_H_noiseless = H_noiseless

    test_labelset_su = labelset_su

    return test_dataset,test_H,test_H_noiseless,test_dataset_bar,test_labelset_su


