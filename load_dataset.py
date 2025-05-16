from loaddata import *

import numpy as np
import scipy.fft as fft


len_s = 5
total_samples = s5.shape[0]
num_cuts =int(total_samples / (f_samp * len_s))

biases = np.mean(s5, axis=0)

s5 -= biases

trim_start_idx = 0

idxs = np.arange(trim_start_idx, s5.shape[0], f_samp * len_s)

times = [idxtot(idx) for idx in idxs]

trim_end_idx = None
s5_trimmed = s5[trim_start_idx:, :]

split = np.array_split(s5, num_cuts)
split = [arr.T for arr in split]

split_fft = [fft.fft((split_window), axis=1) for split_window in split]



ys = ranges["Range(km)"][:-2].to_numpy(dtype=float)
xp = np.arange(0, len(ys) * (60 // len_s), 60 // len_s) # evenly spaced x-values corresponding to the 1-minute range values
x_interp = np.arange(0, len(ys) * (60 // len_s), 1) # evenly spaced x-values corresponding to the starts of each sub-minute window
ys_repeat = np.repeat(ys, 60 // len_s)  # range values that are simply repeated
ys_interp = np.interp(x_interp, xp, ys)  # range values that are linearly interpolated. this is better

# import matplotlib.pyplot as plt
#
# plt.plot(x_interp, ys_interp)
# plt.plot(x_interp, ys_repeat)
# plt.show()


