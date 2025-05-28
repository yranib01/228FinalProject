"""
load_dataset.py

- Swellex96 VLA(raw data) -> per 5 seconds (X)
- Range(km)               -> per 5 seconds resolution label (y)

X.shape == (N_WINDOWS, N_SEONSORS =21, N_SAMPLES_PER_WINDOW = 7500)
y.shape == (N_WINDOWS, 1)    # float
"""
import numpy as np
import loaddata as ld
import scipy.fft as fft
import matplotlib.pyplot as plt
from preprocessing import *


# 1) parameters
LEN_S    = 5 # length in Seconds => how many seconds per window(processing data per one time)
SAMP_WIN = ld.F_SAMP * LEN_S   # SAMPles per WINdow : 1500 * 5 = 7500

# 2) Read the signal & remove the bias
START_CUT = 0 # if we want to change, then change this value
s5 = ld.s5[START_CUT:].astype(np.float32).copy()   # s5.mat copy
biases = np.mean(s5,axis=0) # axis = 0 : mean of each sensor(sensor -> column)
s5 -= biases

# 3) cut the window
total_samples = s5.shape[0]   # total_samles = 6,750,000
# new_total_samples = ld.s5[:n_windows * SAMP_WIN]   # if the windows are not cut evenly
n_windows     = total_samples // SAMP_WIN   # n_windows = 900 (cuts)
s5_trim       = s5[:n_windows * SAMP_WIN]    # For cutting the last window / # 6,750,000 - 7500 = 6,742,500
# -> for slicing the last window 0 ~ n_windows * SAMP_WIN live, rest is cut
windows = np.array_split(s5_trim, n_windows, axis=0) # list[(7500, 21)]
X_raw = np.stack([w.T for w in windows])  # (n_win(now 900), 21, 7500)
X_raw_fft = np.fft.rfft(X_raw, axis=2)
X_deep = deep_bandpass(X_raw) # (900, 21, 7500) # deep bandpass filter

# # ======================================================================
# # 4) FFT
# win_idx, chan = 0, 1
# # Number of sample points
# N = 7500
# # Sample spacing
# fs = ld.F_SAMP

# # windowing(fft전에 먼저 filter링 해주는것 같은데... 잡음제거?)
# print(sig.shape) # (7500,)
# sig = X[win_idx, chan].copy() # (7500,)
# sig = sig.astype(float)
# sig -= sig.mean() # remove the mean
# sig *= np.hanning(len(sig)) # Hanning window
# print(sig.shape, len(sig)) # (7500,)

# # frequency axis (0 ~ 750Hz, 3751 bin)
# freqs = np.fft.rfftfreq(len(sig), d=1/fs) # (3751,)
# print(freqs.shape) # (3751,)
# # FFT(window, channel, frequency)
# # we already choose a specific window and channel at the windowing step
# X_fft = np.fft.rfft(sig)
# print(X_fft.shape) # (3751,)
# # log scale
# mag = 20 * np.log10(np.abs(X_fft)/len(sig)) # (3751,)
# # X_fft = np.abs([np.fft.rfft(x, axis=1) for x in X]) # (900, 21, 7500)
# # print(X_fft[0][1].shape) # (21, 7500)
# print(mag.shape) # (3751,)
# # specific channel
# # spec = X_fft[win_idx][chan] # (3751,)

# plt.figure(figsize=(5,3))
# deep = [49,64,79,94,112,130,148,166,201,235,283,338,388]
# for f0 in deep:
#     plt.axvline(f0, color='cyan', alpha=0.3, ls='--')
# # plt.semilogy(freqs, 2/N * spec + 1e-12) # log scale
# plt.plot(freqs, mag) # log scale
# # plt.xlim(0, 450)
# plt.title(f'FFT - window , ch ')
# plt.xlabel('Hz'); plt.ylabel('|FFT|')
# plt.tight_layout()
# plt.savefig("fft_ex.png", dpi=300, bbox_inches='tight')
# print("Saved -> 'fft_ex.png'")
# plt.close()
# # ======================================================================



# 5) Make the label
range_vals = ld.range_df["Range(km)"].dropna().to_numpy(np.float32) # (900,)
offset_sec = START_CUT / ld.F_SAMP   # offset_sec(signal's 0s location) = 0
xp_minutes = np.arange(len(range_vals)) * 60  # [0, 60, 120, ..., 900*60] seconds
x_windows = np.arange(n_windows) * LEN_S #[0, 5, 10, ..., 900*5] seconds

X_fft_selected_freqs = fft_bandpass(X_raw_fft, DEEP_TONES)
X_fft_coeffs = complex_to_mag_angle(X_fft_selected_freqs)

y = np.interp(x_windows, xp_minutes, range_vals).reshape(-1,1).astype(np.float32) # (900, 1) -> 1D

# =======================================
# Results
# =======================================
# X.shape == (900, 21, 7500)
# y.shape == (900, 1)    # float

if __name__ == "__main__":
    np.savez("X_deep", X_deep)
    np.savez("X_raw", X_raw)
    np.savez("X_fft_selected", X_fft_coeffs)
    np.savez("y_range", y)
