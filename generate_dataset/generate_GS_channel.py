import numpy as np 
import hdf5storage
def gen_OFDMmodel(N_rx,N_tx,N_sc,N_path,f_s,tap_range):
    H_freq = (np.random.randn(N_rx,N_tx,N_sc) + 1j * np.random.randn(N_rx,N_tx,N_sc)) / np.sqrt(2)
    return H_freq
def generate_channel_dataset(data_num,Nt,Nr,K,B,Lp,f_s,tap_range):
    channel_dataset = np.zeros((data_num,Nt,Nr,K,B)) + 1j* np.zeros((data_num,Nt,Nr,K,B))
    for data_index in range(data_num):
        if data_index %1000==0:
            print(str(data_index+1)+'has been generated.')
        channel_shards = np.zeros((Nt,Nr,K,B)) + 1j*np.zeros((Nt,Nr,K,B))
        for user in range(K):
            ofdm_channel = gen_OFDMmodel(N_rx=Nr,N_tx=Nt,N_sc = B,N_path=10,f_s = f_s,tap_range=tap_range)
            for rb in range(B):
                channel_shards[:,:,user,rb] = ofdm_channel[:,:,rb].T
        channel_dataset[data_index,:] = channel_shards  
    return channel_dataset
if __name__ == "__main__":
    print('begin to generated!!')
    channel = generate_channel_dataset(50000,Nt = 64,Nr = 4, K = 20, B = 4, Lp = 10, f_s = 720*1e3, tap_range = 100)
    save_root = '/data/zhangmaojun/GS_dataset/channel_dataset.mat'
    hdf5storage.savemat(save_root,{'H_list':channel})
    #import ipdb;ipdb.set_trace()

