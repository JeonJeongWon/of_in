import numpy as np

def pa(x):
    amplitude = np.abs(x)
    angle = np.angle(x)
    A = 2.1587
    B = 1.1517
    alpha = 4.033
    beta = 9.1040

    phase_dis = (alpha*(amplitude**2))/(1+beta*(amplitude**2))
    amp_dis = (A*amplitude)/(1+B*(amplitude**2))

    y = amp_dis*np.exp(1j*(angle+phase_dis))
    return y
