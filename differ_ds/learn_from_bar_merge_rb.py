import numpy as np

np.random.seed(2021)
from scipy import io
from sklearn import preprocessing
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense, Conv1D, Conv2D, Flatten, Permute, Reshape, Input, \
    BatchNormalization, Concatenate, Add, Lambda, GlobalAveragePooling1D, Concatenate, GlobalAvgPool1D, Multiply
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as BK
from custom_train import train_func
import os
import hdf5storage

#tf.config.run_functions_eagerly(True)

from option import parge_config
from learn_from_bar_merge_rb_preprocessing import data_process

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
if data_mode == 'debug':
    epochs = 5
test_length = 2000
dataset_root = '/mnts2d/diis_data1/zmj/LCP_dataset/dataset/'
data_root = dataset_root + 'data/DUU_MISO_dataset_differ_ds_%d_%d_%d_%d_%d_%d.mat' % (Nt, Nr, K, dk, B, SNR_dB)
train_mode = 'train'
# data_root = './data/su_dataset_%d_%d_%d_%d_%d.mat'%(Nt,Nr,K,dk,SNR)
dataset, test_dataset, H, test_H, H_noiseless, test_H_noiseless, labelset_su, test_labelset_su, dataset_bar, test_dataset_bar, ds_dataset, test_ds_dataset = \
    data_process(data_root,Nt, Nr, dk, K, B, SNR_dB, SNR_channel_dB, test_length, data_mode)
import gc

gc.collect()
prefix = 'learn_from_bar'


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
        data = Reshape((K*dk*B,K*dk*B,1))(data)
        #data = Permute((1,3,2))(data)
        #import ipdb;ipdb.set_trace()
        if K*dk>12:
            x = Conv2D(filters=16, kernel_size=(7, 7))(data)
            x = BatchNormalization()(x)
            x = LeakyReLU()(x)
        else:
            x = data
        x = Conv2D(filters=8, kernel_size=(5, 5))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(filters=4, kernel_size=(3, 3))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Flatten()(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU()(x)
        #x = Flatten()(x)

        x_tp = BatchNormalization()(x)
        # tp_list = Dense(K * dk)(x_tp)
        # tp_list = BatchNormalization()(tp_list)
        # tp_list = Activation('relu')(tp_list)
        tp_list = Dense(K * dk )(x_tp)
        tp_list = Activation('sigmoid')(tp_list)
        tp_list = Lambda(vector_norm_l1)(tp_list)

        x_up = BatchNormalization()(x)
        # up_list = Dense(K * dk * B)(x_up)
        # up_list = BatchNormalization()(up_list)
        # up_list = Activation('relu')(up_list)
        up_list = Dense(K * dk *B)(x_up)
        up_list = Activation('sigmoid')(up_list)
        up_list = Lambda(vector_norm_l1)(up_list)

        # x_q = BatchNormalization()(x)
        # # q_list = Dense(K * dk * B * 2)(x_q)
        # # q_list = BatchNormalization()(q_list)
        # # q_list = Activation('relu')(q_list)
        # q_list = Dense(K * dk * B * 2)(x_q)
        # q_list = Activation('tanh')(q_list)
        #q_list = Lambda(vector_norm_l2)(q_list)
        prediction = Concatenate(axis=1)([tp_list, up_list])
    return prediction


def weighted_mse_loss(y_true, y_pred):
    pa_list_pred = y_pred[:, :K * dk]
    lambda_list_pred = y_pred[:, K * dk:(K * dk + K * dk * B)]
    #q_list_pred = y_pred[:, (K * dk + K * dk * B):]

    pa_list_true = y_true[:, :K * dk]
    lambda_list_true = y_true[:, K * dk: (K * dk + K * dk * B)]
    #q_list_true = y_true[:, (K * dk + K * dk * B):]

    mse_pa = tf.reduce_mean(tf.square((pa_list_true - pa_list_pred)), axis=-1)
    mse_lambda = tf.reduce_mean(tf.square((lambda_list_true - lambda_list_pred)), axis=-1)
    #mse_q = tf.reduce_mean(tf.square((q_list_true - q_list_pred)), axis=-1)
    loss = mse_pa + mse_lambda #+ mse_q
    return loss


def su_net(Nt, Nr, K, B, dk, lr):
    data = Input(shape=(K * dk * dk * K * B * B))
    prediction = backbone(data, Nt, Nr, dk, K, B)
    model = Model(inputs=data, outputs=prediction)
    model.compile(loss=weighted_mse_loss, optimizer=Adam(lr=lr))
    model.summary()
    return model


lr = 1e-2

model = su_net(Nt, Nr, K, B, dk, lr)
if data_mode == 'debug_test':
    from keras_flops import get_flops

    total_flops = get_flops(model, batch_size=1)
    print('DL_flops:' + str(total_flops))
    EZF_flops = (K * (2 * Nt * Nr ** 2 + 11 * Nr ** 3) + (K * dk) ** 2 * (2 * K * dk + 4 * Nt)) * (2 + 6) / 2
    print('EZF_flops:' + str(EZF_flops))
    print('ratio:' + str(total_flops / EZF_flops))
    import ipdb;

    ipdb.set_trace()

su_model_path = dataset_root + 'DUU_MISO_models_%d_%d_%d_%d_%d_%d_%d_su.h5' % (Nt, Nr, K, B, dk, SNR_dB, SNR_channel_dB)
checkpointer = ModelCheckpoint(su_model_path, verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=25)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-5, min_lr=1e-5)

if train_mode == 'train':
    model.fit(dataset_bar, labelset_su, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=1, \
              callbacks=[checkpointer, reduce_lr, early_stopping])
gc.collect()

def minus_sum_rate_loss_for_train(y_true, y_pred, Nt, Nr, dk, K, B, p, sigma_2):
    y_true = tf.cast(tf.reshape(y_true, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = y_true[:, :, :, :, :K] + 1j * y_true[:, :, :, :, K:]
    y_pred = tf.cast(y_pred, tf.complex128)
    V0 = y_pred[:, :, :, :, 0] + 1j * y_pred[:, :, :, :, 1]

    ## power normalization of the predicted beamformers
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[:, :, :, user], tf.transpose(V0[:, :, :, user], perm=[0, 2, 1], conjugate=True)))

    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    energy_scale = tf.tile(tf.reshape(energy_scale, (-1, 1, 1, 1)), (1, Nt, dk, K))
    V = V0 * tf.cast(energy_scale, tf.complex128)
    sum_rate = 0.0

    for k in range(K):
        for rb in range(B):
            H_k = tf.transpose(H[:, :, :, rb, k], perm=[0, 2, 1])  # NrxNt
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
    loss = sum_rate
    return loss

# %% unsupervised training
#@tf.function
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
    #import ipdb;ipdb.set_trace()
    #print(len(V0))
    trace_VV = 0
    for user in range(K):
        #print(len(V0))
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[user], V0[user],adjoint_b=True))
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


# @tf.function
def pq2V(channel, Nt, Nr, dk, K, B, p, sigma_2, lambda_list, pa_list):
    # recover
    channel = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    # recover
    channel = channel[:, :, :, :, :K] + 1j * channel[:, :, :, :, K:]  # new channel shape:(B,Nt,K)
    channel = tf.transpose(channel, [0, 2, 1, 3, 4])

    lambda_list = tf.reshape(tf.cast(lambda_list, dtype=tf.complex128), [-1, K, B])

    pa_list = tf.cast(pa_list, dtype=tf.complex128)

    temp_inverse = sigma_2 * tf.eye(Nt, dtype=tf.complex128)
    for user in range(K):
        for rb in range(B):
            temp_inverse = temp_inverse + tf.tile(tf.reshape(lambda_list[:, user, rb], [-1, 1, 1]),
                                                  [1, Nt, Nt]) * tf.matmul(channel[:, :, :, rb, user],
                                                                           channel[:, :, :, rb, user], adjoint_a=True)
    temp_inverse = tf.linalg.inv(temp_inverse)

    V_norm = list()
    for user in range(K):
        HUW_this_user = tf.zeros([Nt, dk], dtype=tf.complex128)
        for rb in range(B):
            HUW_this_user = HUW_this_user + tf.transpose(channel[:, :, :, rb, user], [0, 2, 1],
                                                                                conjugate=True)
        V_temp = tf.matmul(temp_inverse, HUW_this_user)
        V_temp, _ = tf.linalg.normalize(V_temp[:, :, 0] + 1e-16, axis=1)
        V_norm.append(V_temp)
    V_norm = tf.stack(V_norm, 2)
    V = list()
    for k in range(K):
        V_temp = tf.tile(tf.reshape(tf.sqrt(pa_list[:, k]), (-1, 1)), (1, Nt)) * V_norm[:, :, k]
        V.append(V_temp)
    V = tf.stack(V, 2)  # (B,Nt,K)
    V = tf.reshape(V, (-1, Nt, 1, K, 1))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    return V


def pq2V_low_complexity(channel, Nt, Nr, dk, K, p, sigma_2, q_list, p_list):
    channel = tf.cast(channel, tf.complex128)
    channel = tf.reshape(channel, (-1, Nt, Nr, K * 2))
    channel = channel[:, :, 0, :K] + 1j * channel[:, :, 0, K:]  # new channel shape:(B,Nt,K)
    p_list = tf.cast(p_list, tf.complex128)
    q_list = tf.cast(q_list, tf.complex128)
    V_norm = list()
    B = sigma_2 * tf.eye(K, dtype=tf.complex128)
    weight_P = tf.matmul(channel, tf.sqrt(tf.linalg.diag(q_list)))
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


# @tf.function
def reconstruct_V(V_restore_miso, ds_list):
    V_restore_miso = tf.cast(V_restore_miso, tf.complex128)
    V_restore_miso = V_restore_miso[:, :, :, :, 0] + 1j * V_restore_miso[:, :, :, :, 1]
    V = list()
    ds_list = tf.cast(ds_list[0, :], tf.int32)
    ds_acc = 0
    for ds in ds_list:
        V.append(V_restore_miso[:, :, 0, ds_acc:ds_acc + ds])
        ds_acc += ds
    return V


# @tf.function
def DUU_EZF_loss(y_true, y_pred):
    # import ipdb;ipdb.set_trace()
    channel_miso = y_true[:, :2 * dk * Nt * K * B]
    channel = y_true[:, 2 * dk * Nt * K * B: (2 * dk * Nt * K * B + 2 * Nr * Nt * K * B)]
    data_stream_distribution = y_true[:, -K:]
    # channel_miso = tf.concat([channel_miso,data_stream_distribution],axis=1)
    channel = tf.concat([channel, data_stream_distribution], axis=1)
    # channel = y_true
    pa_list_pred = y_pred[:, :K * dk]
    lambda_list_pred = y_pred[:, K * dk:(K * dk + K * dk * B)]
    #q_list_pred = y_pred[:, (K * dk + K * dk * B):]
    V_restore = pq2V(channel=channel_miso, Nt=Nt, Nr=1, dk=1, K=K * dk, B=B, p=p, sigma_2=sigma_2,
                     lambda_list=lambda_list_pred, pa_list=pa_list_pred)
    #V_restore = reconstruct_V(V_restore, data_stream_distribution)

    loss = -minus_sum_rate_loss_for_train(channel_miso, V_restore, Nt=Nt, Nr=1, dk=1, K=K*dk, B=B, p=p, sigma_2=sigma_2) / B
    return loss


def un_net(Nt, Nr, K, B, dk, lr):
    data = Input(shape=(K * dk * dk * K * B * B))
    prediction = backbone(data, Nt, Nr, dk, K, B)

    model = Model(inputs=data, outputs=prediction)
    model.compile(loss=DUU_EZF_loss, optimizer=Adam(lr=lr))
    model.summary()

    return model


lr = 1e-3
model = un_net(Nt, Nr, K, B, dk, lr)
model.load_weights(su_model_path)
un_model_path = dataset_root + 'DUU_MISO_models_%d_%d_%d_%d_%d_%d_%d_un.h5' % (Nt, Nr, K, B, dk, SNR_dB, SNR_channel_dB)
checkpointer = ModelCheckpoint(un_model_path, verbose=1, save_best_only=True, save_weights_only=True)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=25)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_delta=1e-4, min_lr=1e-5)
# data_num = len(H)
# H = np.reshape(H, (data_num, -1))

# labelset_un = np.concatenate([dataset, H_noiseless, ds_dataset], axis=-1)
# del dataset, H_noiseless
# gc.collect()
#
# if train_mode == 'train':
#     model.fit(dataset_bar, labelset_un, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=1, \
#               callbacks=[checkpointer, reduce_lr, early_stopping])

# train_func(model=model,loss_func = DUU_EZF_loss,epochs = epochs,lr =lr,model_path = un_model_path,argument_list=argument_list,batch_size=batch_size,train_dataset_root = dataset_unsu[:-1],valid_dataset_root = dataset_unsu[-1:])

# train_dataset,valid_dataset = generate_unsu_dataset_from_numpy(dataset_bar,labelset_un,batch_size)
# if train_mode == 'train':
#     model.fit(train_dataset, validation_data = valid_dataset,epochs=epochs, batch_size=batch_size, verbose=1, \
#             callbacks=[checkpointer, reduce_lr, early_stopping])
labelset_un = np.concatenate([dataset, H_noiseless, ds_dataset], axis=-1)
del dataset, H_noiseless
gc.collect()
# labelset_un = H_noiseless
if data_mode == 'gpu':
    train_sample_num = int(dataset_bar.shape[0] // 10000 * 10000)
    valid_sample_num = dataset_bar.shape[0] - train_sample_num
else:
    train_sample_num = int(dataset_bar.shape[0] / 2)
    valid_sample_num = int(dataset_bar.shape[0] / 2)
# to save momory, first store the train and valid dataset to
train_func(model=model, loss_func=DUU_EZF_loss, epochs=epochs, lr=lr, model_path=un_model_path, batch_size=batch_size,
           train_dataset_bar=dataset_bar[:train_sample_num, :], train_labelset_un=labelset_un[:train_sample_num, :],
           valid_dataset_bar=dataset_bar[-valid_sample_num:, :], valid_labelset_un=labelset_un[-valid_sample_num:, :],
           split_ratio=4)
gc.collect()
def restore(channel, prediction, data_stream_distribution):
    # channel = y_true
    pa_list_pred = prediction[:, :K * dk]
    lambda_list_pred = prediction[:, K * dk:(K * dk + K * dk * B)]
    #q_list_pred = prediction[:, (K * dk + K * dk * B):]

    V_restore = pq2V(channel=channel, Nt=Nt, Nr=1, dk=1, K=K * dk, B=B, p=p, sigma_2=sigma_2,
                     lambda_list=lambda_list_pred, pa_list=pa_list_pred)
    V_restore = reconstruct_V(V_restore, data_stream_distribution)
    # V_restore = tf.reshape(V_restore,(-1,Nt,dk,K,2))

    return V_restore


def EZF(channel, Nt, Nr, dk, K, B, p, sigma_2, P_return=False):
    ds_list = tf.cast(channel[0, -K:], tf.int32)
    channel = channel[:, :-K]
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]
    H = tf.transpose(H, [0, 2, 1, 3, 4])
    P = list()
    for user in range(K):
        H_this_user = tf.matmul(tf.transpose(H[:, :, :, 0, user], [0, 2, 1], conjugate=True), H[:, :, :, 0, user])
        for rb in range(1, B):
            H_this_user = H_this_user + tf.matmul(tf.transpose(H[:, :, :, rb, user], [0, 2, 1], conjugate=True),
                                                  H[:, :, :, rb, user])
        _, _, v = tf.linalg.svd(H_this_user)
        P.append(v[:, :, :ds_list[user]])
    # P = tf.stack(P, axis=3)
    P = tf.concat([vi for vi in P], axis=2)
    P = tf.reshape(P, [-1, Nt, K * dk])
    # import ipdb;ipdb.set_trace()
    V = tf.matmul(P, tf.linalg.inv(tf.matmul(tf.transpose(P, [0, 2, 1], conjugate=True), P)))  # B*Nt*Kdk
    V = tf.reshape(V, [-1, Nt, dk * K])
    V0 = list()
    ds_acc = 0
    for user in range(K):
        V0.append(V[:, :, ds_acc:ds_acc + ds_list[user]])
        ds_acc += ds_list[user]

    V = list()
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[user], tf.transpose(V0[user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    # import ipdb;ipdb.set_trace()
    for user in range(K):
        energy_scale_temp = tf.tile(tf.reshape(energy_scale, (-1, 1, 1)), (1, Nt, ds_list[user]))
        V.append(V0[user] * tf.cast(energy_scale_temp, tf.complex128))
    # import ipdb;ipdb.set_trace()
    if P_return:
        P = tf.reshape(P, [-1, Nt, 1, K * dk])
        P = tf.cast(tf.concat([tf.math.real(P), tf.math.imag(P)], axis=3), dtype=tf.float32)
        return V, P
    else:
        return V


def WMMSE(channel_ds, Nt, Nr, dk, K, B, p, sigma_2, UW_return=False):
    def update_WMMSE_U(H, V):
        U = list()
        trace_VV = 0
        for user in range(K):
            trace_VV = trace_VV + tf.linalg.trace(
                tf.matmul(V[user], V[user], adjoint_b=True))
        for user in range(K):
            U_this_user_list = list()
            for rb in range(B):
                HVVH = tf.zeros([Nr, Nr], dtype=tf.complex128)
                for k in range(K):
                    HV = tf.matmul(H[:, :, :, rb, user], V[k])
                    HVVH = HVVH + tf.matmul(HV, HV, adjoint_b=True)
                inverse_temp = tf.linalg.inv(
                    sigma_2 / p * tf.tile(tf.reshape(trace_VV, (-1, 1, 1)), [1, Nr, Nr]) * tf.eye(Nr,
                                                                                                  dtype=tf.complex128) + HVVH)
                U_this_user = tf.matmul(tf.matmul(inverse_temp, H[:, :, :, rb, user]), V[user])
                U_this_user_list.append(U_this_user)
            U.append(U_this_user_list)
        return U

    def update_WMMSE_W(H, U, V):
        W = list()
        for user in range(K):
            W_this_user_list = list()
            for rb in range(B):
                HV = tf.matmul(H[:, :, :, rb, user], V[user])
                W_this_user = tf.linalg.inv(tf.eye(V[user].shape[2], dtype=tf.complex128) - tf.matmul(
                    U[user][rb], HV, adjoint_a=True))
                W_this_user_list.append(W_this_user)
            W.append(W_this_user_list)
        return W

    def update_WMMSE_V(H, U, W):
        temp_B = tf.zeros([Nt, Nt], dtype=tf.complex128)
        for user in range(K):
            for rb in range(B):
                HHU = tf.matmul(H[:, :, :, rb, user], U[user][rb], adjoint_a=True)  # b*Nt*dk
                trace_UWU = sigma_2 / p * tf.linalg.trace(tf.matmul(tf.matmul(U[user][rb], W[user][rb]),
                                                                    U[user][rb], adjoint_b=True))
                temp_B = temp_B + tf.tile(tf.reshape(trace_UWU, (-1, 1, 1)), [1, Nt, Nt]) * tf.eye(Nt,
                                                                                                   dtype=tf.complex128) + tf.matmul(
                    tf.matmul(HHU, W[user][rb]), HHU, adjoint_b=True)

        temp_B_inverse = tf.linalg.inv(temp_B)

        V0 = list()
        VV = tf.zeros([batch_size, Nt, Nt], dtype=tf.complex128)

        for user in range(K):
            HUW = tf.matmul(tf.matmul(H[:, :, :, 0, user], U[user][0], adjoint_a=True), W[user][0])
            for rb in range(1, B):
                HUW = HUW + tf.matmul(tf.matmul(H[:, :, :, rb, user], U[user][rb], adjoint_a=True),
                                      W[user][rb])  # b*Nt*dk
            V0_this_user = tf.matmul(temp_B_inverse, HUW)
            V0.append(V0_this_user)

        V = list()
        trace_VV = 0
        for user in range(K):
            trace_VV = trace_VV + tf.linalg.trace(
                tf.matmul(V0[user], tf.transpose(V0[user], perm=[0, 2, 1], conjugate=True)))
        energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
        # import ipdb;ipdb.set_trace()
        for user in range(K):
            energy_scale_temp = tf.tile(tf.reshape(energy_scale, (-1, 1, 1)), (1, Nt, ds_list[user]))
            V.append(V0[user] * tf.cast(energy_scale_temp, tf.complex128))
        return V

    ds_list = tf.cast(channel_ds[0, -K:], tf.int32)
    channel = channel_ds[:, :-K]
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]  # B*Nt*Nr*K
    H = tf.transpose(H, (0, 2, 1, 3, 4))
    V = EZF(channel_ds, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B, p=p, sigma_2=sigma_2)
    for i in range(50):
        U = update_WMMSE_U(H, V)
        W = update_WMMSE_W(H, U, V)
        V = update_WMMSE_V(H, U, W)
    V0 = V
    V = list()
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[user], tf.transpose(V0[user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    # import ipdb;ipdb.set_trace()
    for user in range(K):
        energy_scale_temp = tf.tile(tf.reshape(energy_scale, (-1, 1, 1)), (1, Nt, ds_list[user]))
        V.append(V0[user] * tf.cast(energy_scale_temp, tf.complex128))
    if UW_return:
        return U, W, V
    else:
        return V
    H = tf.cast(tf.reshape(channel, [-1, Nt, Nr, B, 2 * K]), tf.complex128)
    H = H[:, :, :, :, :K] + 1j * H[:, :, :, :, K:]  # B*Nt*Nr*K
    H = tf.transpose(H, (0, 2, 1, 3, 4))
    V = EZF(channel, Nt=Nt, Nr=Nr, dk=dk, K=K, B=B, p=p, sigma_2=sigma_2)
    V = tf.cast(V, tf.complex128)
    V = V[:, :, :, :, 0] + 1j * V[:, :, :, :, 1]  # B*Nt*dk*K
    for i in range(50):
        U = update_WMMSE_U(H, V)
        W = update_WMMSE_W(H, U, V)
        V = update_WMMSE_V(H, U, W)
    V0 = V
    trace_VV = 0
    for user in range(K):
        trace_VV = trace_VV + tf.linalg.trace(
            tf.matmul(V0[:, :, :, user], tf.transpose(V0[:, :, :, user], perm=[0, 2, 1], conjugate=True)))
    energy_scale = tf.sqrt(p / tf.cast(trace_VV, tf.float32))
    energy_scale = tf.tile(tf.reshape(energy_scale, (-1, 1, 1, 1)), (1, Nt, dk, K))
    V = V0 * tf.cast(energy_scale, tf.complex128)
    V = tf.reshape(V, (-1, Nt, dk, K, 1))
    V = tf.cast(tf.concat([tf.math.real(V), tf.math.imag(V)], axis=4), dtype=tf.float32)
    if UW_return:
        return U, W, V
    else:
        return V


channel = np.concatenate([test_H, test_ds_dataset], axis=1)
channel_noiseless = test_H_noiseless
channel_noiseless = np.concatenate([channel_noiseless, test_ds_dataset], axis=1)
model.load_weights(un_model_path)
model_output = model.predict(test_dataset_bar)

batch_size = 200
iter_num = len(channel) // batch_size
DUU_EZF_performance = np.zeros(iter_num)
DUU_MISO_performance = np.zeros(iter_num)
EZF_performance = np.zeros(iter_num)
RWMMSE_performance = np.zeros(iter_num)
for it in range(iter_num):
    DUU_EZF_output = restore(test_dataset[it * batch_size:(it + 1) * batch_size, :],
                             model_output[it * batch_size:(it + 1) * batch_size, :],
                             test_ds_dataset[it * batch_size:(it + 1) * batch_size, :])
    DUU_EZF_performance[it] = np.mean(
        minus_sum_rate_loss(channel_noiseless[it * batch_size:(it + 1) * batch_size, :], DUU_EZF_output, Nt=Nt, Nr=Nr,
                            dk=dk, K=K, B=B, p=p, sigma_2=sigma_2))

    DUU_MISO_output = restore(test_dataset[it * batch_size:(it + 1) * batch_size, :],
                              test_labelset_su[it * batch_size:(it + 1) * batch_size, :],
                              test_ds_dataset[it * batch_size:(it + 1) * batch_size, :])
    DUU_MISO_performance[it] = np.mean(
        minus_sum_rate_loss(channel_noiseless[it * batch_size:(it + 1) * batch_size, :], DUU_MISO_output, Nt=Nt, Nr=Nr,
                            dk=dk, K=K, B=B, p=p, sigma_2=sigma_2))
    # import ipdb;ipdb.set_trace()

    EZF_output = EZF(channel[it * batch_size:(it + 1) * batch_size, :], Nt, Nr, dk, K, B, p, sigma_2)
    EZF_performance[it] = np.mean(
        minus_sum_rate_loss(channel_noiseless[it * batch_size:(it + 1) * batch_size, :], EZF_output, Nt=Nt, Nr=Nr,
                            dk=dk, K=K, B=B, p=p, sigma_2=sigma_2))
    WMMSE_output = WMMSE(channel[it * batch_size:(it + 1) * batch_size, :], Nt, Nr, dk, K, B, p, sigma_2)
    RWMMSE_performance[it] = np.mean(
        minus_sum_rate_loss(channel_noiseless[it * batch_size:(it + 1) * batch_size, :], WMMSE_output, Nt=Nt, Nr=Nr,
                            dk=dk, K=K, B=B, p=p, sigma_2=sigma_2))

import logging

logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(
    dataset_root + 'result/tensorflow_%s_%d_%d_%d_%d_%d_%d_%d.log' % (prefix, Nt, Nr, K, B, dk, SNR_dB, SNR_channel_dB))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('EZF sum rate:%.5f' % (np.mean(EZF_performance) / B))
logger.info('DUU EZF sum rate:%.5f' % (np.mean(DUU_EZF_performance) / B))
logger.info('DUU MISO sum rate:%.5f' % (np.mean(DUU_MISO_performance) / B))
logger.info('RWMMSE sum rate:%.5f' % (np.mean(RWMMSE_performance) / B))
# print('EZF sum rate:%.5f'%np.mean(EZF_performance))
# print('DUU EZF sum rate:%.5f'%np.mean(DUU_EZF_performance))
# print('RWMMSE sum rate:%.5f'%np.mean(RWMMSE_performance))


# CUDA_VISIBLE_DEVICES=0 python learn_from_bar.py  --Nt 64 --Nr 4 --dk 2 --K 10 --SNR 0 --SNR_channel 100 --gpu 0 --mode gpu --batch_size 200 --epoch 1000 --factor 2
