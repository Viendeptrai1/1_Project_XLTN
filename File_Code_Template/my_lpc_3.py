import numpy as np
import librosa
import soundfile as sf

bat_dau = 95*16
ket_thuc = bat_dau + 400
data, fs = sf.read('mix_03.wav', dtype='int16')
x = data[bat_dau:ket_thuc]
# Chuẩn hóa giữa -1 và 1
x = x.astype(np.float32)
x = x / 32768

# pre-emphasis
N = len(x)
y = np.zeros((N,), np.float32)
a = 0.9
for n in range(0, N):
    if n == 0:
        y[n] = x[n] - a*x[n]
    else:
        y[n] = x[n] - a*x[n-1]
        
# Tạo cửa sổ Hamming
w = np.zeros((N,), np.float32)
for n in range(0, N):
    w[n] = 0.54 - 0.46*np.cos(2*np.pi*n/N)
    
# Nhân với cửa sổ Hamming
z = y*w
 
# Tính các hệ số dự báo tuyến tính
# Return the [1, -a1, ..., -am] coefficients.
a = librosa.lpc(z, order=12)
a = -a
m = 18
p = 12
c = np.zeros((19,), dtype=np.float32)
m = 1
while m <= p:
    c[m] = a[m]
    k = 1
    while k <= m-1:
        c[m] = c[m] + (k/m)*c[k]*a[m-k]
        k = k+1
    m = m+1
m =  p+1
while m <= 18:
    c[m] = 0.0
    k = 1
    while k <= m-1:
        chi_so = m-k
        if m-k > p:
            temp = 0.0
        else:
            temp = a[m-k]
        c[m] = c[m] + (k/m)*c[k]*temp
        k = k+1
    m = m+1
print(c)
    
    