import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import FastICA

np.random.seed(100)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal
S = np.zeros((3, n_samples), dtype=np.float64)
S[0, :] = s1
S[1, :] = s2
S[2, :] = s3
S = S.T
S = S + 0.2 * np.random.normal(size=S.shape)  # Add noise

# Chuẩn hóa bằng cách chia cho độ lệch chuẩn (standard deviation)
S /= S.std(axis=0)
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.matmul(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3, whiten="arbitrary-variance")
R = ica.fit_transform(X)  # Reconstruct signals

plt.plot(time, R[:, 2])
plt.show()