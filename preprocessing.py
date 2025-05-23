# preprocessing.py
import numpy as np
from scipy.signal import butter, lfilter

DEEP_TONES = np.array([49,64,79,94,112,130,148,166,201,235,283,338,388])

def deep_bandpass(sig, fs=1500, bw=1.5):
    """
    sig : ndarray (..., N)  마지막 축이 시간
    반환 : 동일 shape  (Deep 톤 ±bw Hz 대역만 통과)
    """
    y = np.zeros_like(sig)
    for f0 in DEEP_TONES:
        wn = [(f0-bw)/(fs/2), (f0+bw)/(fs/2)]   # normalized frequency
        b, a = butter(4, wn, btype='band')
        y += lfilter(b, a, sig, axis=-1)
    return y
