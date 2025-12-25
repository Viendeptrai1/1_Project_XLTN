import numpy as np
import librosa

x = np.arange(1, 201)
N = len(x)
y = librosa.autocorrelate(x, max_size=12)
print(y)

R0 = 0
k = 0
for n in range(0, N):
    R0 = R0 + x[n] * x[n+k]
print(R0)

R11 = 0
k = 11
for n in range(0, N):
    chi_so_sau = n + k
    if chi_so_sau > N - 1:
        temp = 0
    else:
        temp = x[n+k]
    R11 = R11 + x[n] * temp
print(R11)  