import numpy as np
import tensorflow as tf
from scipy import io
import hdf5storage
np.random.seed(2021)

def MIMO2MISO(channel,Nt,Nr,dk,K,B):
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]
    H_ensemble = tf.reshape(H, [-1, Nt, Nr * B * K])
    H_bar = tf.matmul(H_ensemble, H_ensemble, adjoint_a=True)
    H_bar = np.triu(np.real(H_bar)) + np.tril(np.imag(H_bar))
    return H_bar
def data_process(data_root,Nt,Nr,dk,K,B,SNR_dB,SNR_channel_dB,test_length,data_mode):
    total_dataset = hdf5storage.loadmat(data_root)
    if data_mode == 'debug':
        data_length = 4000
    else:
        data_length = len(total_dataset['H'])
    H = total_dataset['H'][:data_length, :]
    u_flatten_dataset = total_dataset['U'][:data_length, :]
    w_flatten_dataset = total_dataset['W'][:data_length, :]
    w_flatten_dataset = np.reshape(w_flatten_dataset,(-1,dk,dk,B,2*K))
    w_flatten_dataset = w_flatten_dataset[:,:,:,:,:K]
    w_flatten_dataset = np.reshape(w_flatten_dataset,(-1,dk*dk*B*K))

    labelset_su = np.concatenate((u_flatten_dataset,w_flatten_dataset), axis=-1)

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

    H_bar = np.zeros((data_num,K*Nr*B,K*Nr*B))
    total_iter = len(H_bar)//1000
    for i in range(total_iter):
        H_iter = H[i * 1000:(i + 1) * 1000, :]
        H_iter = np.concatenate([np.real(H_iter), np.imag(H_iter)], axis=-1)
        H_bar_iter = MIMO2MISO(channel=H_iter, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B)

        H_bar[i * 1000:(i + 1) * 1000, :] = H_bar_iter
    H_bar = np.reshape(H_bar,(data_num,-1))
    dataset_bar = H_bar

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

    return H,test_H,H_noiseless,test_H_noiseless,labelset_su,test_labelset_su,dataset_bar,test_dataset_bar


