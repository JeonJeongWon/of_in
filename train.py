import time
import numpy as np
import tensorflow as tf
from model import *
from channel import *
from ofdm import *
from tqdm import tqdm

def train():
    #BitGenerate---------------------------------------------------------
    tx_time_bits = np.zeros((sys.num_datas,sys.num_sc,sys.bps))

    for i in range(sys.num_datas):
        tx_time_bits_s = bitsGenerate()
        tx_time_bits_p = s2p(tx_time_bits_s)
        tx_time_bits[i,:,:] = tx_time_bits_p

    z = tx_time_bits
    c = z