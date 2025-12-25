import numpy as np
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype='float32')
N = len(x)

# Biến đổi DFT
X = np.fft.fft(x, N)

# Phần thực 
print(X.real)
# Phần ảo
print(X.imag)

print('Spectrum')
S = np.sqrt(X.real**2 + X.imag**2)

for k in range(0, N):
    print('%2d --> %10.2f\n' % (k, S[k]))
    
# Biến đổi ngược IDFT
x_original = np.fft.ifft(X, N)
print(x_original.real)
print(x_original.imag)