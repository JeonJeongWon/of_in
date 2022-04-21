import tensorflow as tf
from variables import *
from ofdm import *
from channel import *

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
        denominator = tf.math.log(tf.constant(10,dtype=numerator.dtype))
        return numerator/denominator


    def compute_loss_1(self,x):
        # BER LOSS
        # TX-------------------------------------
        tx_fre_sym_div = self.encode(x)
        tx_fre_sym = flt2com(tx_fre_sym_div)
        tx_time_sym = IFFT(tx_fre_sym)
        # CH-------------------------------------
        ofdm_sym = channel(tx_time_sym)
        # RX-------------------------------------
        rx_fre_sym = FFT(ofdm_sym)
        rx_fre_sym_div = com2flt(rx_fre_sym)
        rx_time_sym = self.decode(rx_fre_sym_div)

        loss_1 = tf.math.reduce_sum(tf.math.square(x.numpy()-rx_time_sym),axis=1)/(sys.num_sc*sys.bps)

        return loss_1


    def compute_loss_2(self, x):
        # BER LOSS
        # TX-------------------------------------
        tx_fre_sym_div = self.encode(x)
        tx_fre_sym = flt2com(tx_fre_sym_div)
        tx_time_sym = IFFT(tx_fre_sym)
        # CH-------------------------------------
        ofdm_sym = channel(tx_time_sym)
        # RX-------------------------------------
        rx_fre_sym = FFT(ofdm_sym)
        rx_fre_sym_div = com2flt(rx_fre_sym)
        rx_time_sym = self.decode(rx_fre_sym_div)

        loss_1 = tf.math.reduce_sum(tf.math.square(x.numpy()-rx_time_sym),axis=1)/(sys.num_sc*sys.bps)

        # PAPR
        pwr = tf.math.pow(tf.abs(tx_time_sym),2)
        max = tf.reduce_max(pwr,axis=1)
        mean = tf.reduce_mean(pwr,axis=1)
        papr = 10*self.log10(max/mean)

        return loss_1, papr

    def compute_gradient_1(self,x):
        with tf.GradientTape() as tape:
            loss_1 = self.compute_loss_1(x)
        cg = tape.gradient(loss_1,self.encoder.trainable_variables + self.decoder.trainable_variables)
        return cg, loss_1


    def compute_gradient_2(self,x):
        with tf.GradientTape() as tape:
            loss_1, papr = self.compute_loss_2(x)
            loss_sum = loss_1 + nn.weight*papr
        cg = tape.gradient(loss_sum,self.encoder.trainable_variables + self.decoder.trainable_variables)
        return cg, loss_1, papr

    def train_1(self,x):
        cg, loss_1 = self.compute_gradient_1(x)
        self.optimizer.apply_gradients(zip(cg,self.encoder.trainable_variables+self.decoder.trainable_variables))
        return loss_1


    def train_2(self,x):
        cg, loss_1,papr = self.compute_gradient_2(x)
        self.optimizer.apply_gradients(zip(cg,self.encoder.trainable_variables+self.decoder.trainable_variables))
        return loss_1, papr

    data_shape = sys.num_sc * sys.bps

    enc_struc = [
     tf.keras.layers.InputLayer(input_shape=(data_shape)),

     tf.keras.layers.Dense(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Dense(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Dense(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Dense(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),
     tf.keras.layers.Dense(nn.num_nodes),
     tf.keras.layers.BatchNormalization(),
     tf.keras.layers.Activation('relu'),

     tf.keras.layers.Dense(data_shape,activation='linear')
    ]

    dec_struc = [
        tf.keras.layers.InputLayer(input_shape=(data_shape)),

        tf.keras.layers.Dense(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dense(nn.num_nodes),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(data_shape, activation='tanh')
    ]

    optimizer= tf.keras.optimizers.Adam(learning_rate=nn.lr)