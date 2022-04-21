import time
import numpy as np
import tensorflow as tf
from model import *
from channel import *
from ofdm import *
from tqdm import tqdm

def data_generate():
    # Bit Generate---------------------------------------------------------
    tx_data = np.zeros((sys.num_datas, sys.num_sc, sys.bps))
    test_data_ = np.zeros((sys.num_datas // 6, sys.num_sc, sys.bps))
    for i in range(sys.num_datas):
        tx_time_bits_s = bitsGenerate()
        tx_time_bits_p = s2p(tx_time_bits_s)
        tx_data[i, :, :] = tx_time_bits_p

    for i in range(sys.num_datas // 6):
        test_time_bits_s = bitsGenerate()
        test_time_bits_p = s2p(test_time_bits_s)
        test_data_[i, :, :] = test_time_bits_p

    train_data = tx_data.reshape(sys.num_datas, sys.num_sc * sys.bps)
    test_data = test_data_.reshape(sys.num_datas // 6, sys.num_sc * sys.bps)
    # Bit Generate---------------------------------------------------------

    return train_data,test_data

def train(train_ver,model,train_data,test_data):

    train_data = tf.data.Dataset.from_tensor_slices(train_data).shuffle(sys.num_datas).batch(nn.batch_size,
                                                                                          drop_remainder=True)
    test_data = tf.data.Dataset.from_tensor_slices(test_data).shuffle(sys.num_datas // 6).batch(nn.batch_size,
                                                                                                 drop_remainder=True)

    train_loss_1 = tf.keras.metrics.Mean(name="train_loss_1") # BER
    train_loss_2 = tf.keras.metrics.Mean(name="train_loss_2") # PAPR
    test_loss_1 = tf.keras.metrics.Mean(name="test_loss_1") # BER
    test_loss_2 = tf.keras.metrics.Mean(name="test_loss_2") # PAPR

    # loss_1-----------------------
    if train_ver== 0 :
        for epoch in range(nn.epochs):
          start_time = time.time()

          for x_batch in enumerate(tqdm(train_data)):
             loss_1 = model.train_1(x_batch[1])
             train_loss_1(loss_1)

          for x_batch_test in enumerate(test_data):
             loss_1_t = model.compute_loss_1(x_batch_test[1])
             test_loss_1(loss_1_t)

          template = 'Epoch {:d}/{:d}, Train-Loss: [{:2.4f}], Test-Loss: [{:2.4f}]'

          print(template.format(epoch+1,nn.epochs,train_loss_1.result(),test_loss_1.result(),))
          print("Time taken: %.2fs" %(time.time()-start_time))

          train_loss_1.reset_states()
          test_loss_1.reset_states()

          nn.wait +=1
          if train_loss_1 > nn.best_1:
              nn.best_1 = train_loss_1
              nn.wait=0
          if nn.wait>=nn.patience: break

        return model

    # loss_1 + papr ------------------------------------------------
    elif train_ver == 1:
        for epoch in range(nn.epochs):
            start_time = time.time()

            for x_batch in enumerate(tqdm(train_data)):
                loss_1,loss_2 = model.train_2(x_batch)
                train_loss_1(loss_1);train_loss_2(loss_2)

            for x_batch_test in enumerate(test_data):
                loss_1_t,loss_2_t = model.compute_loss_2(x_batch_test)
                test_loss_1(loss_1_t);test_loss_2(loss_2_t)

            template = 'Epoch {:d}/{:d}, Train-Loss: [{:2.4f},{:2.4f}], Test-Loss: [{:2.4f},{:2.4f}]'

            print(template.format(epoch + 1, nn.epochs, train_loss_1.result(), train_loss_2.result(),
                                  test_loss_1.result(), test_loss_2.result(), ))
            print("Time taken: %.2fs" % (time.time() - start_time))

            train_loss_1.reset_states()
            train_loss_2.reset_states()
            test_loss_1.reset_states()
            test_loss_2.reset_states()

            nn.wait += 1
            if train_loss_2 > nn.best_2:
                nn.best_2 = train_loss_2
                nn.wait = 0
            if nn.wait >= nn.patience: break

        return model