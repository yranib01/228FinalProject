import matplotlib.pyplot as plt
import loaddata as ld

h1 = ld.s5[:, 0]
k, d = 1000000, 250

plt.figure(figsize=(6,3))
plt.plot(h1[k:k+d])
plt.savefig("segment_plot.png", dpi=300, bbox_inches='tight')
plt.close()