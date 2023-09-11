import numpy as np 
import hdf5storage
def gen_OFDMmodel(N_rx,N_tx,N_sc,N_path,f_s,tap_range):
    delta_f = f_s / N_sc
    gamma = (np.random.randn(N_path,1) + 1j * np.random.randn(N_path,1)) / np.sqrt(2)
    tau = np.random.rand(N_path,1) * tap_range * 1e-9

    AoA_theta = 2 * np.pi * (np.random.rand(N_path,1)-0.5)
    AoD_theta = 2 * np.pi * (np.random.rand(N_path,1)-0.5)
    A_steer_rx = np.zeros((N_rx,N_path)) + 1j*np.zeros((N_rx,N_path))
    for ii in range(N_path):
        A_steer_rx[:,ii] = np.exp(-1j*np.pi*(np.arange(0,N_rx,1)-1).T * np.sin(AoA_theta[ii]))
    A_steer_tx = np.zeros((N_tx,N_path)) + 1j*np.zeros((N_tx,N_path))
    for ii in range(N_path):
        A_steer_tx[:,ii] = np.exp(-1j*np.pi*(np.arange(0,N_tx,1)-1).T * np.sin(AoD_theta[ii]))
    H_freq = np.zeros((N_rx,N_tx,N_sc)) + 1j*np.zeros((N_rx,N_tx,N_sc))
    for ii in range(N_sc):
        for jj in range(N_path):
            H_freq[:,:,ii] = H_freq[:,:,ii] + gamma[jj]*A_steer_rx[:,jj:jj+1] * A_steer_tx[:,jj:jj+1].T * np.exp(-1j * 2*np.pi * ii * tau[jj]*delta_f)
    return H_freq/np.sqrt(N_path)
def generate_channel_dataset(data_num,Nt,Nr,K,B,Lp,f_s,tap_range):
    channel_dataset = np.zeros((data_num,Nt,Nr,K,B)) + 1j* np.zeros((data_num,Nt,Nr,K,B))
    for data_index in range(data_num):
        if data_index %1000==0:
            print(str(data_index+1)+'has been generated.')
        channel_shards = np.zeros((Nt,Nr,K,B)) + 1j*np.zeros((Nt,Nr,K,B))
        for user in range(K):
            ofdm_channel = gen_OFDMmodel(N_rx=Nr,N_tx=Nt,N_sc = 12*B,N_path=10,f_s = f_s,tap_range=tap_range)
            for rb in range(B):
                channel_shards[:,:,user,rb] = ofdm_channel[:,:,rb*12].T
        channel_dataset[data_index,:] = channel_shards  
    return channel_dataset
if __name__ == "__main__":
    print('begin to generated!!')
    channel = generate_channel_dataset(4000,Nt = 16,Nr = 2, K = 10, B = 4, Lp = 10, f_s = 0.32*1e9, tap_range = 100)
    save_root = './dataset/channel_dataset.mat'
    hdf5storage.savemat(save_root,{'H_list':channel})
    #import ipdb;ipdb.set_trace()


