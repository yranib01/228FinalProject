# For seeing the spectrogram of the signal
# expecially, finding the CW and chirp

import loaddata as ld
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

sig = ld.s5[:, 0] - ld.s5[:,0].mean()          # 채널 0 예시
f, t, S = signal.spectrogram(sig, fs=ld.F_SAMP,
                      nperseg=4096, noverlap=3072,
                      scaling='density')
plt.pcolormesh(t/60, f, 10*np.log10(S+1e-12),
               cmap='inferno', vmin=-120, vmax=-60)
plt.ylim(0, 450); plt.xlabel('time [min]'); plt.ylabel('Hz')

plt.savefig("spectrogram_CW_chirp.png", dpi=300, bbox_inches='tight')
print("Saved -> 'spectrogram_CW_chirp.png'")
plt.close()