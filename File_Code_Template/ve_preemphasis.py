import matplotlib.pyplot as plt
import numpy as np
Fmax = 5000
n = 500
f = np.linspace(0, Fmax, 500)
Fs = 10000
Ts = 1/Fs
a = 0.98
H = np.sqrt(1 + a**2 - 2*a*np.cos(2 * np.pi * f * Ts))
H = 20*np.log10(H)

plt.plot(f, H)
plt.plot([0, 5000], [0, 0], 'r')
plt.xlabel('Hz')
plt.ylabel('|H(f)|\n(dB)')
plt.title('Frequency Response of Pre-emphasis')
plt.show()

