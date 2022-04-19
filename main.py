import numpy as np
import matplotlib.pyplot as plt
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


a = np.arange(0,128,1)
print(a.reshape(64,2).reshape(128,))