import matplotlib.pyplot as plt
from loaddata import *

h1 = s5[:, 0]
k = 1000000
d = 250
plt.plot(h1[k:k+d])
plt.show()


