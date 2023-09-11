import tensorflow as tf
import numpy as np
from tensorflow.python.keras.backend import stop_gradient
import hdf5storage


@tf.function
def printbar():
    ts = tf.timestamp()
    today_ts = ts % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return (tf.strings.format("0{}", m))
        else:
            return (tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                                  timeformat(second)], separator=":")
    tf.print("==========" * 8, end="")
    tf.print(timestring)


@tf.function
def train_step(model, features, labels, loss_func, optimizer, train_loss):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)


@tf.function
def valid_step(model, features, labels, loss_func, valid_loss):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)


def generate_unsu_dataset_from_numpy(dataset_bar, labelset_un, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset_bar[:-dataset_bar.shape[0] // 10, :], labelset_un[:-dataset_bar.shape[0] // 10, :])).batch(batch_size)
    # valid_dataset = tf.data.Dataset.from_tensor_slices((dataset_bar[-dataset_bar.shape[0]//10:,:],labelset_un[-dataset_bar.shape[0]//10:,:])).batch(batch_size)
    return train_dataset


def generate_su_dataset_from_numpy(dataset_bar, labelset_su, batch_size):
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset_bar[:-dataset_bar.shape[0] // 10, :], labelset_su[:-dataset_bar.shape[0] // 10, :])).batch(batch_size)
    valid_dataset = tf.data.Dataset.from_tensor_slices(
        (dataset_bar[-dataset_bar.shape[0] // 10:, :], labelset_su[-dataset_bar.shape[0] // 10:, :])).batch(batch_size)
    return train_dataset, valid_dataset


# train_func(model=model,loss_func = DUU_EZF_loss,epochs = epochs,lr =lr,model_path = un_model_path,batch_size=batch_size,train_dataset_bar = dataset_bar[:train_sample_num,:],train_labelset_un = labelset_un[:train_sample_num,:],valid_dataset_bar = [-valid_sample_num:,:],valid_labelset_un = labelset_un[-valid_sample_num:,:],split_ratio = 4)
def train_func(model, loss_func, epochs, lr, model_path, batch_size, train_dataset_bar, train_labelset_un,
               valid_dataset_bar, valid_labelset_un, split_ratio):
    optimizer = tf.keras.optimizers.Adam(lr)
    # import ipdb;ipdb.set_trace()
    loss_value = None
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')

    min_loss = 100
    lr_patient = 10
    stop_patient = 25
    cur_lr_patient = lr_patient
    cur_stop_patient = stop_patient
    # import ipdb;ipdb.set_trace()
    valid_dataset = generate_unsu_dataset_from_numpy(valid_dataset_bar, valid_labelset_un, batch_size)
    sum_data_num = int(len(train_dataset_bar) // split_ratio)
    for epoch in range(1, epochs + 1):
        for sum_epoch in range(split_ratio):
            train_dataset = generate_unsu_dataset_from_numpy(
                train_dataset_bar[int(sum_epoch * sum_data_num):int((sum_epoch + 1) * sum_data_num), :],
                train_labelset_un[int(sum_epoch * sum_data_num):int((sum_epoch + 1) * sum_data_num), :], batch_size)
            for features, labels in train_dataset:
                train_step(model, features, labels, loss_func, optimizer, train_loss)
        for features, labels in valid_dataset:
            valid_step(model, features, labels, loss_func, valid_loss)
        valid_loss_value = valid_loss.result()
        logs = 'Epoch={},Loss:{},Valid Loss:{}'
        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(logs,
                                       (epoch, train_loss.result(), valid_loss.result())))
        # import ipdb;ipdb.set_trace()
        if valid_loss_value < min_loss and abs(valid_loss_value - min_loss) > 1e-4:
            tf.print(
                'val_loss improved from %.5f to %.5f, saving model to %s!!' % (min_loss, valid_loss_value, model_path))
            min_loss = valid_loss_value
            model.save_weights(model_path)
            cur_lr_patient = lr_patient
            cur_stop_patient = stop_patient
        else:
            cur_lr_patient = cur_lr_patient - 1
            cur_stop_patient = cur_stop_patient - 1
        if cur_lr_patient <= 0:
            # optimizer._decayed_lr(tf.Variable(0.1))
            optimizer.lr.assign(0.1 * optimizer.lr.read_value().numpy())
            tf.print("Learning rate has reduced to " + str(optimizer.lr.read_value().numpy()))
            cur_lr_patient = lr_patient
        if optimizer.lr.read_value() <= 1e-5 or cur_stop_patient <= 0:
            tf.print('Early stop!!')
            break
        train_loss.reset_states()
        valid_loss.reset_states()
        tf.print("")