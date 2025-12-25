import soundfile as sf
import numpy as np
from sklearn.decomposition import FastICA
import sounddevice as sd

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

# Compute ICA
ica = FastICA(n_components=2, whiten="unit-variance")
s = ica.fit_transform(x)  # Reconstruct signals
s1 = s[:, 0]
s2 = s[:, 1]

# Chuẩn hóa giữa -1 và 1
min_s1 = min(s1)
s1 = s1 / abs(min_s1)
# sd.play(s1, 8000)
# sd.wait()

max_s2 = max(s2)
s2 = s2 / abs(max_s2)
sd.play(s2, 8000)
sd.wait()