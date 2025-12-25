import soundfile as sf
import numpy as np
from sklearn.decomposition import PCA

x1, fs = sf.read('mix_violin_piano_01.wav', dtype='int16')
L = len(x1)
x1 = x1.astype(np.float32)
x1 = x1 / 32768

x2, fs = sf.read('mix_violin_piano_02.wav', dtype='int16')
x2 = x2.astype(np.float32)
x2 = x2 / 32768

x = np.zeros((2, L), np.float32)
x[0, :] = x1
x[1, :] = x2

x = x.T

m = np.mean(x, 0)
x = x - m

# Kiểm tra ma trận convariance
C = np.matmul(x.T, x)
C = C / L
print ('C', C)

pca = PCA(whiten=True)
y = pca.fit_transform(x)

# Kiểm tra ma trận convariance
C = np.matmul(y.T, y)
C = C / L
print ('C sau khi làm trắng ', C)
