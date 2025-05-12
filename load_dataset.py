from loaddata import *

import numpy as np

len_s = 5
total_samples = s5.shape[0]
num_cuts =int(total_samples / (f_samp * len_s))

split = np.array_split(s5, num_cuts)

ys = range["Range(km)"].to_numpy(dtype=float)
ys = np.repeat(ys, 60 // len_s).tolist()


