import loaddata as ld
import scipy.signal as signal
import matplotlib.pyplot as plt
import numpy as np
# import matplotlib
# print(matplotlib.get_backend())

h1 = ld.s5[:, 0]

f, t, Sxx = signal.spectrogram(h1)

Sxx_db = 10 * np.log10(Sxx + 1e-12)
vmax = Sxx_db.mean()
vmin = vmax - 20
pc = plt.pcolormesh(t, f, Sxx_db,
                    shading = 'auto',
                    vmin = vmin, vmax = vmax)
# plt.clim(0, 5e-9)
plt.colorbar(pc, label='Power [dB]')
plt.xlabel('Time[s]')
plt.ylabel('Frequency [Hz]')
# plt.ylim(0, 300)
plt.tight_layout()
plt.show()

f_cutoff = [0.1, 0.3]




# Save the figure
plt.savefig("spectrogram.png", dpi=300, bbox_inches='tight')
print("Saved -> 'spectrogram.png'")
plt.close()