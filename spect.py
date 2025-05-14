from load_dataset import *
import scipy.signal as signal
import matplotlib.pyplot as plt

h1 = s5[:, 0]

f, t, Sxx = signal.spectrogram(h1)
plt.pcolormesh(t, f, Sxx)
plt.clim(0, 0.000005)
plt.show()

f_cutoff = [0.1, 0.3]
