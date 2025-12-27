"""
Các hàm tương phản đo độ phi Gaussian - dùng trong FastICA
"""

import numpy as np


def kurtosis(x):
    """Tính kurtosis (moment bậc 4 - 3). Đo khoảng cách phân phối so với Gaussian."""
    x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-10)
    kurt = np.mean(x_normalized ** 4) - 3
    return kurt


def negentropy(x, n_samples=1000):
    """
    Xấp xỉ negentropy: J(x) ≈ [E{G(x)} - E{G(v)}]²
    với v ~ N(0,1)
    """
    x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-10)
    v = np.random.randn(n_samples)
    exp_g_x = np.mean(g_logcosh(x_normalized))
    exp_g_v = np.mean(g_logcosh(v))
    neg_ent = (exp_g_x - exp_g_v) ** 2
    return neg_ent


def g_logcosh(x, alpha=1.0):
    """Hàm contrast: G(x) = log(cosh(αx))"""
    return np.log(np.cosh(alpha * x) + 1e-10)


def dg_logcosh(x, alpha=1.0):
    """Đạo hàm bậc 1: g(x) = tanh(αx)"""
    return alpha * np.tanh(alpha * x)


def ddg_logcosh(x, alpha=1.0):
    """Đạo hàm bậc 2: g'(x) = α²(1 - tanh²(αx))"""
    tanh_val = np.tanh(alpha * x)
    return alpha ** 2 * (1 - tanh_val ** 2)
