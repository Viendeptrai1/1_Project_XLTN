import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(100)
n = 7200
x = rng.normal(size=(n, 2))

_, _, Vh = np.linalg.svd(x, full_matrices=True)
Vt = Vh.T
y = np.matmul(x, Vt)

# Kiểm tra ma trận convariance
Cx = np.matmul(x.T, x)
Cx = Cx / n
print ('Cx', Cx)

# Kiểm tra ma trận convariance
Cy = np.matmul(y.T, y)
Cy = Cy / n
print ('Cy', Cy)


