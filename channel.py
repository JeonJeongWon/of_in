import tensorflow as tf
from ofdm import *
from variables import *

def channel():
    ch_pwr_profile = tf.complex(tf.math.exp(-0.5))