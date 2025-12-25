import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import FastICA

X, y = load_digits(return_X_y=True)
