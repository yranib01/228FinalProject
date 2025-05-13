from loaddata import *

import numpy as np

len_s = 5
total_samples = s5.shape[0]
num_cuts =int(total_samples / (f_samp * len_s))

split = np.array_split(s5, num_cuts)
split = [arr.T for arr in split]

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


