import numpy as np
N = 8
x = np.zeros(N, dtype='float32')
maxS0 = 0.0
maxS1 = 0.0
maxS2 = 0.0
maxS3 = 0.0
maxS4 = 0.0
maxS5 = 0.0
maxS6 = 0.0
maxS7 = 0.0
print('Đang chạy')
x0 = np.zeros(N, dtype='float32')
x1 = np.zeros(N, dtype='float32')
x2 = np.zeros(N, dtype='float32')
x3 = np.zeros(N, dtype='float32')
x4 = np.zeros(N, dtype='float32')
x5 = np.zeros(N, dtype='float32')
x6 = np.zeros(N, dtype='float32')
x7 = np.zeros(N, dtype='float32')

for i0 in range(-4, 4):
    for i1 in range(-4, 4):
        for i2 in range(-4, 4):
            for i3 in range(-4, 4):
                for i4 in range(-4, 4):
                    for i5 in range(-4, 4):
                        for i6 in range(-4, 4):
                            for i7 in range(-4, 4):
                                x[0] = i0
                                x[1] = i1
                                x[2] = i2
                                x[3] = i3
                                x[4] = i4
                                x[5] = i5
                                x[6] = i6
                                x[7] = i7
                                X = np.fft.fft(x, N)
                                S = np.sqrt(X.real**2 + X.imag**2)
                                
                                if S[0] > maxS0:
                                    maxS0 = S[0]
                                    x0 = x.copy()
                                if S[1] > maxS1:
                                    maxS1 = S[1]
                                    x1 = x.copy()
                                if S[2] > maxS2:
                                    maxS2 = S[2]
                                    x2 = x.copy()
                                if S[3] > maxS3:
                                    maxS3 = S[3]
                                    x3 = x.copy()
                                if S[4] > maxS4:
                                    maxS4 = S[4]
                                    x4 = x.copy()
                                if S[5] > maxS5:
                                    maxS5 = S[5]
                                    x5 = x.copy()
                                if S[6] > maxS6:
                                    maxS6 = S[6]
                                    x6 = x.copy()
                                if S[7] > maxS7:
                                    maxS7 = S[7]
                                    x7 = x.copy()
print('maxS0 = %10.2f ' % maxS0, end = '')
for i in range(0, N):
    print('%4.0f' % x0[i], end = '')
print()

print('maxS1 = %10.2f ' % maxS1, end = '')
for i in range(0, N):
    print('%4.0f' % x1[i], end = '')
print()

print('maxS2 = %10.2f ' % maxS2, end = '')
for i in range(0, N):
    print('%4.0f' % x2[i], end = '')
print()

print('maxS3 = %10.2f ' % maxS3, end = '')
for i in range(0, N):
    print('%4.0f' % x3[i], end = '')
print()

print('maxS4 = %10.2f ' % maxS4, end = '')
for i in range(0, N):
    print('%4.0f' % x4[i], end = '')
print()

print('maxS5 = %10.2f ' % maxS5, end = '')
for i in range(0, N):
    print('%4.0f' % x5[i], end = '')
print()

print('maxS6 = %10.2f ' % maxS6, end = '')
for i in range(0, N):
    print('%4.0f' % x6[i], end = '')
print()

print('maxS7 = %10.2f ' % maxS7, end = '')
for i in range(0, N):
    print('%4.0f' % x7[i], end = '')
print()

print('Chạy xong')