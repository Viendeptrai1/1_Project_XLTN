import matplotlib.pyplot as plt
import numpy as np

def MyFourier(x):
    M = len(x)
    XR = np.zeros(M, np.float32)
    XI = np.zeros(M, np.float32)
    for m in range(0, M):
        XR[m] = 0.0
        XI[m] = 0.0
        for n in range(0, M-1):
            XR[m] = XR[m] + x[n] * np.cos(2*np.pi*m*n/M)
            XI[m] = XI[m] + x[n] * np.sin(2*np.pi*m*n/M)
    spectrum = np.sqrt(XR**2 + XI**2)
    return spectrum

n =  500
x = np.linspace(0, 500, n)
x1 = np.linspace(0, 2*2*np.pi, n)
y1 = np.sin(x1)

x2 = np.linspace(0, 200*2*np.pi, n)
y2 = np.sin(x2)

y = y1 + y2

S = MyFourier(y)
L = len(S)
f = open('Spectrum.txt', 'wt')
s = ''
for i in range(0, L):
    s = s + '%3d --> %10.2f\n' % (i, S[i])
f.write(s)
f.close()

plt.plot(x, y)
plt.show()