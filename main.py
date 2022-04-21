import numpy as np
import matplotlib.pyplot as plt
from train import *
from salehAmp import *

# a = np.array([-3-3j,-3-1j,-3+1j,-3+3j,
#               -1-3j,-1-1j,-1+1j,-1+3j,
#               +1-3j,+1-1j,+1+1j,+1+3j,
#               +3-3j,+3-1j,+3+1j,+3+3j])
#
#
# a_n = a/(abs(a)**2)
# out=pa(a_n)
# rx=out*(abs(a)**2)
# #
# in_pw = abs(a)**2
# out_pw = abs(out)**2
# #
# plt.scatter(rx.real,rx.imag)
# plt.scatter(a.real,a.imag)
# plt.show()
# #
# plt.plot(in_pw,out_pw)
# plt.show()
# ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# ------------------

train_data, test_data = data_generate()
model_loss_1 = train(0,train_data,test_data)  # 0 : loss_1 only training, # 1 : loss_1 + papr training

model_loss_1.encoder.save("Encoder_loss1.h5")
model_loss_1.decoder.save("Decoder_loss1.h5")

