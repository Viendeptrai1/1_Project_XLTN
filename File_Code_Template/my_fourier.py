import numpy as np

def DFT(x, N):
    XR = np.zeros(N, dtype = 'float32')
    XI = np.zeros(N, dtype = 'float32')
    for k in range(0, N):
        XR[k] = 0.0
        XI[k] = 0.0
        for n in range(0, N):
            XR[k] = XR[k] + x[n] * np.cos(2*np.pi*k*n/N)
            XI[k] = XI[k] - x[n] * np.sin(2*np.pi*k*n/N)
    return (XR, XI)

def IDFT(XR, XI, N):
    xR = np.zeros(N, dtype = 'float32')
    xI = np.zeros(N, dtype = 'float32')
    for n in range(0, N):
        xR[n] = 0.0
        xI[n] = 0.0
        for k in range(0, N):
            a = XR[k]
            b = XI[k]
            c = np.cos(2*np.pi*k*n/N)
            d = np.sin(2*np.pi*k*n/N)
            xR[n] = xR[n] + a * c - b * d
            xI[n] = xI[n] + a * d + b * c
        xR[n] = xR[n] / N
        xI[n] = xI[n] / N
    return (xR, xI)

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='float32')

XR, XI = DFT(x, 16)

xR, xI = IDFT(XR, XI, 16)

for n in range(0, 16):
    print('%5.0f' % xR[n], end = '')
print() 
for n in range(0, 16):
    print('%5.0f' % xI[n], end = '')   