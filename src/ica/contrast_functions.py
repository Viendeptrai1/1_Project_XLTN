"""
Contrast functions for measuring non-Gaussianity
Used in FastICA algorithm
"""

import numpy as np


def kurtosis(x):
    """
    Compute kurtosis (4th moment - 3)
    Measures how far distribution is from Gaussian
    
    Parameters:
    -----------
    x : np.ndarray
        Input signal
        
    Returns:
    --------
    kurt : float
        Kurtosis value
    """
    x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-10)
    kurt = np.mean(x_normalized ** 4) - 3
    return kurt


def negentropy(x, n_samples=1000):
    """
    Approximate negentropy using Gaussian reference
    J(x) ≈ [E{G(x)} - E{G(v)}]^2
    where v ~ N(0,1)
    
    Parameters:
    -----------
    x : np.ndarray
        Input signal
    n_samples : int
        Number of samples for Gaussian reference
        
    Returns:
    --------
    neg_ent : float
        Negentropy approximation
    """
    x_normalized = (x - np.mean(x)) / (np.std(x) + 1e-10)
    
    v = np.random.randn(n_samples)
    
    exp_g_x = np.mean(g_logcosh(x_normalized))
    exp_g_v = np.mean(g_logcosh(v))
    
    neg_ent = (exp_g_x - exp_g_v) ** 2
    
    return neg_ent


def g_logcosh(x, alpha=1.0):
    """
    Contrast function G(x) = log(cosh(alpha * x))
    
    Parameters:
    -----------
    x : np.ndarray
        Input
    alpha : float
        Parameter (typically 1.0)
        
    Returns:
    --------
    result : np.ndarray
        G(x)
    """
    return np.log(np.cosh(alpha * x) + 1e-10)


def dg_logcosh(x, alpha=1.0):
    """
    Derivative of G(x) = tanh(alpha * x)
    g(x) = d/dx log(cosh(alpha * x))
    
    Parameters:
    -----------
    x : np.ndarray
        Input
    alpha : float
        Parameter (typically 1.0)
        
    Returns:
    --------
    result : np.ndarray
        g(x)
    """
    return alpha * np.tanh(alpha * x)


def ddg_logcosh(x, alpha=1.0):
    """
    Second derivative of G(x)
    g'(x) = alpha^2 * (1 - tanh^2(alpha * x))
    
    Parameters:
    -----------
    x : np.ndarray
        Input
    alpha : float
        Parameter (typically 1.0)
        
    Returns:
    --------
    result : np.ndarray
        g'(x)
    """
    tanh_val = np.tanh(alpha * x)
    return alpha ** 2 * (1 - tanh_val ** 2)
