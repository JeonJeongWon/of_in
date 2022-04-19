import numpy as np
import tensorflow as tf
from variables import *

def bitsGenerate():
    tx_data_bits = np.random.binomial(n=1,p=0.5,size=(sys.num_sc*sys.bps))
    return tx_data_bits

def s2p (tx_data_bits):
    return tx_data_bits.reshape(sys.num_sc,sys.bps)

def p2s (rx_data_bits):
    return rx_data_bits.reshape((-1,))

# def papr(tx_data_sym):
#     sigsq=np.power(np.abs(tx_data_sym),2)
#     paprdB = 10.0*np.log10(np.divide(np.max(sigsq,axis=-1),np.mean(sigsq,axis=-1)))
#     return paprdB
#-------------------------------------------------------------------------------

def flt2com(tx_data_sym_div):
    tx_data_sym_div = tf.reshape(tx_data_sym_div,shape=(nn.batch_size,sys.num_sc,sys.bps))
    return tf.complex(tx_data_sym_div[:,:,0],tx_data_sym_div[:,:,1])
    #return tf.complex(tx_data_sym_div[:,:sys.num_sc],tx_data_sym_div[:,sys.num_sc:])

def IFFT(tx_data_sym):
    return tf.signal.ifft(tx_data_sym)

def FFT(rx_time_sym):
    return tf.siganl_fft(rx_time_sym)

def com2flt(rx_fre_sym_comb):
    rx_fre_sym_div_real = tf.math.real(rx_fre_sym_comb)
    rx_fre_sym_div_imag = tf.math.imag(rx_fre_sym_comb)
    x = tf.concat((rx_fre_sym_div_real,rx_fre_sym_div_imag),axis=1)
    return x