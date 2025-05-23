import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import load_dataset as ld
import scipy.fft as fft

# 1) Parameters
chan = 0  # sensor number
win_idx = 12 # window index(0 ~899) for FFT
tone_ref = 49.0 # reference tone for tracking doppler(Hz)
band = 2.0 # +- band width(Hz)
fs = 1500 # sampling frequency(1500Hz)

# 2) Windowing
slice_sig = ld.X[win_idx, chan].copy()
slice_sig -= slice_sig.mean() # remove the mean
slice_sig = np.hanning(len(slice_sig)) # Window..(Hanning)
# print(slice_sig.shape) # (7500,)

# 3) FFT
fft_list = [fft.fft(x, axis=1) for x in ld.X] # (900, 21, 3751)
# axis = 1 : FFT along the time axis(7500 samples) -> 3751 frequency bins
print(fft_list[0].shape) # (21, 3751)

fft_win = fft_list[win_idx] # FFT data   (21, 3751)
print(fft_win.shape) # (21, 3751)

# spec_ch = np.abs(np.fft.rfft(slice_sig)) # frequency magnitude spectrum (3751,)
spec_ch = np.abs(fft_win[chan]) # frequency magnitude spectrum in one channel we want to see (3751,)
# It shows 1-D array that shows where the tone peak is
print(spec_ch.shape) # (3751,)
# print(spec_ch[800])
freq_axis = np.fft.fftfreq(ld.X.shape[-1], 1/fs) # frequency(0 ~ 750Hz) (3751,)
print(freq_axis.shape) # (3751,)

# 3) Plot
plt.figure(figsize=(5,3))
plt.plot(freq_axis, spec_ch)
plt.xlim(0, 450)
plt.title(f'FFT - window {win_idx}, ch {chan}')
plt.xlabel('Hz'); plt.ylabel('|FFT|')
plt.tight_layout()
plt.savefig("FFT.png", dpi=300, bbox_inches='tight')
print("Saved -> 'FFT.png'")
plt.close()


# 3) 