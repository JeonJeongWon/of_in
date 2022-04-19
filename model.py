import tensorflow as tf
from variables import *

class autoencoder(tf.keras.Model):
    def __init__(self,**kwargs):
        super().__init__(self,**kwargs)
        self.__dict__.update(kwargs)
        self.encoder = tf.keras.Sequential(self.enc_struc)
        self.decoder = tf.keras.Sequential(self.dec_struc)

    def encode(self,encin):
        return self.encoder(encin)

    def decode(self,decin):
        return self.decoder(decin)

    def log10(self,x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10,dtype = numerator.type))
        return numerator/denominator

    #def compute_loss_1(self,x):


    data_shape = sys.num_sc * sys.bps

    enc_struc = [
     tf.keras.layers.InputLayer(input_shape=(data_shape)),

     tf.keras.layers.Desne(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Desne(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Desne(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Desne(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Desne(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),

     tf.keras.layers.Desne(data_shape,activation='linear')
    ]

    dec_struc = [
        tf.keras.layers.InputLayer(input_shape=(data_shape)),

        tf.keras.layers.Desne(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Desne(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Desne(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Desne(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Desne(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Desne(data_shape, activation='tanh')
    ]
