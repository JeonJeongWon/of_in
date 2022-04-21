import tensorflow as tf
from ofdm import *
from variables import *

@tf.function
def channel(tx_time_sym):
    ch_tap_arr = tf.cast(tf.range(ch.num_paths),dtype=tf.dtypes.float32)
    ch_pwr_profile = tf.complex(tf.math.exp(-0.5*ch_tap_arr),tf.zeros(ch.num_paths))
    ch_tap_profile = tf.complex(tf.random.normal([nn.batch_size,ch.num_paths],0,1),
                                tf.random.normal([nn.batch_size,ch.num_paths],0,1))/tf.math.sqrt(tf.complex(2.0,0.0))

    ch_time_coef = tf.math.multiply(ch_pwr_profile,ch_tap_profile)

    ch_zero_tensor = tf.zeros((nn.batch_size,(sys.num_sc-ch.num_paths)),dtype=tf.dtypes.complex64)

    ch_time_coef = tf.concat(((ch_time_coef,ch_zero_tensor)),axis=1)
    ch_freq_coef = tf.signal.fft(ch_time_coef)

    ch_time_circ_mat_ = tf.linalg.LinearOperatorCirculant(ch_freq_coef)
    ch_time_circ_mat = ch_time_circ_mat_.to_dense()

    ch_tx_result = ch_mat_operation(tx_time_sym,ch_time_circ_mat)

    sig_mean_pwr = tf.math.reduce_mean(tf.math.abs(tx_time_sym**2))
    noise_pwr_profile = sig_mean_pwr * 10 **(-ch.snrdb/10)

    noise = tf.cast(tf.math.sqrt(noise_pwr_profile/2),dtype=tf.complex64)*tf.complex(tf.random.normal([nn.batch_size,sys.num_sc],0,1),tf.random.normal([nn.batch_size,sys.num_sc],0,1))


    return ch_tx_result + noise

@tf.function
def ch_mat_operation(tx_time_sym, ch_time_circ_mat):
    for i in range(nn.batch_size):
        if i == 0:
            x = tf.reshape(tf.linalg.matmul(ch_time_circ_mat[i],tf.reshape(tx_time_sym[i],shape=(sys.num_sc,-1))),shape=(-1,sys.num_sc))
        else:
            x = tf.concat([x,
                           tf.reshape(tf.linalg.matmul(ch_time_circ_mat[i],tf.reshape(tx_time_sym[i], shape=(sys.num_sc, -1))),shape=(-1, sys.num_sc))], axis=0)
            if i == (sys.num_sc-1) : return x


