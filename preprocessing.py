# preprocessing.py
import numpy as np
from scipy.signal import butter, lfilter

DEEP_TONES = np.array([49,64,79,94,112,130,148,166,201,235,283,338,388])
# DEEP_TONES = np.array([130,148,166,201,235,283,338,388])

def deep_bandpass(sig, fs=1500, bw=1.5):
    """
    sig : ndarray (..., N)  마지막 축이 시간
    반환 : 동일 shape  (Deep 톤 tone ±bw Hz 대역만 통과)
    """
    y = np.zeros_like(sig)
    for f0 in DEEP_TONES:
        wn = [(f0-bw)/(fs/2), (f0+bw)/(fs/2)]   # normalized frequency
        b, a = butter(4, wn, btype='band')
        y += lfilter(b, a, sig, axis=-1)
    return y

def idx_to_f(idx, fs=1500, K=7500):
    if idx < 0 or idx > K:
        raise IndexError("Index is out of range")
    return (idx/K) * fs

def f_to_idx(f, fs=1500, K=7500):
    if f > fs / 2:
        raise Warning(f"Frequency {f} is greater than the Nyquist frequency. Aliasing may occur")
    return round((f/fs) * K)

def fft_bandpass(fft_coeffs, frequencies, fs=1500, K=7500, bw=1.5):
    """
    :param fft_coeffs: array containing (positive) FFT coefficients
    :param frequencies: array containing central frequencies that we want to keep
    :param fs: sampling frequency (Hz)
    :param K: Length of window for DFT
    :param bw: bandwidth in Hz (+/-, not total)
    :return: coefficients
    """
    conversion_factor = fs / K
    idx_range = bw // conversion_factor

    idxs =  np.array([np.arange(f_to_idx(freq) - idx_range, f_to_idx(freq) + idx_range + 1) for freq in frequencies]).flatten().astype(int)

    return fft_coeffs[:, :, idxs]

def complex_to_mag_angle(arr):
    return np.stack([np.abs(arr), np.angle(arr)], axis=-1)