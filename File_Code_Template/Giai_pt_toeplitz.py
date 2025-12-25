import numpy as np
from scipy.linalg import toeplitz

#Define c and r as before
c = np.array([1, 3, 6, 10])
r = np.array([1, -1, -2, -3])
b = np.array([1, 2, 2, 5])

# Construct the full Toepplit
T = toeplitz()