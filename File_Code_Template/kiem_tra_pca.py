import numpy as np
from sklearn.decomposition import PCA
rng = np.random.default_rng(100)
n = 72000
x = rng.normal(size=(n, 2))
pca = PCA(whiten=True)
y = pca.fit_transform(x)
print('Done')

# Kiểm tra ma trận convariance
C = np.matmul(x.T, x)
C = C / n
print ('C', C)