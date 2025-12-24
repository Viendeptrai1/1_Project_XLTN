"""
FastICA Algorithm
Based on: Hyvarinen & Oja (2000) - Independent Component Analysis: Algorithms and Applications
"""

import numpy as np
from ..signal_processing.preprocessing import whitening
from .contrast_functions import dg_logcosh, ddg_logcosh


class FastICA:
    """
    FastICA algorithm for blind source separation
    
    Parameters:
    -----------
    n_components : int, optional
        Number of components to extract (default: None = use all)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    alpha : float
        Parameter for logcosh contrast function
    random_state : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, n_components=None, max_iter=200, tol=1e-4, alpha=1.0, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.random_state = random_state
        
        self.whitening_matrix = None
        self.dewhitening_matrix = None
        self.mean = None
        self.unmixing_matrix = None
        self.mixing_matrix = None
        self.n_iter = 0
    
    def _symmetric_decorrelation(self, W):
        """
        Symmetric decorrelation: W = (W * W^T)^(-1/2) * W
        
        Parameters:
        -----------
        W : np.ndarray
            Weight matrix
            
        Returns:
        --------
        W_orth : np.ndarray
            Orthogonalized weight matrix
        """
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        W_orth = np.dot(U, Vt)
        return W_orth
    
    def _ica_parallel(self, X_white):
        """
        Parallel FastICA algorithm
        Extract all components simultaneously
        
        Parameters:
        -----------
        X_white : np.ndarray
            Whitened data (n_components, n_samples)
            
        Returns:
        --------
        W : np.ndarray
            Unmixing matrix
        """
        n_components, n_samples = X_white.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        W = np.random.randn(n_components, n_components)
        W = self._symmetric_decorrelation(W)
        
        for iteration in range(self.max_iter):
            gwtx = dg_logcosh(np.dot(W, X_white), self.alpha)
            g_wtx = ddg_logcosh(np.dot(W, X_white), self.alpha)
            
            W_new = (np.dot(gwtx, X_white.T) / n_samples - 
                     np.dot(np.diag(np.mean(g_wtx, axis=1)), W))
            
            W_new = self._symmetric_decorrelation(W_new)
            
            max_change = np.max(np.abs(np.abs(np.diag(np.dot(W_new, W.T))) - 1))
            
            W = W_new
            
            if max_change < self.tol:
                self.n_iter = iteration + 1
                break
        else:
            self.n_iter = self.max_iter
            print(f"Warning: FastICA did not converge in {self.max_iter} iterations")
        
        return W
    
    def fit(self, X):
        """
        Fit the ICA model
        
        Parameters:
        -----------
        X : np.ndarray
            Input data (n_features, n_samples) or (n_samples, n_features)
            
        Returns:
        --------
        self : FastICA
            Fitted model
        """
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        n_components = self.n_components if self.n_components else X.shape[0]
        
        X_white, self.whitening_matrix, self.dewhitening_matrix, self.mean = whitening(
            X, n_components=n_components
        )
        
        self.unmixing_matrix = self._ica_parallel(X_white)
        
        self.mixing_matrix = np.dot(self.dewhitening_matrix, 
                                     np.linalg.pinv(self.unmixing_matrix))
        
        return self
    
    def transform(self, X):
        """
        Separate sources from mixtures
        
        Parameters:
        -----------
        X : np.ndarray
            Mixed signals (n_features, n_samples)
            
        Returns:
        --------
        S : np.ndarray
            Separated sources (n_components, n_samples)
        """
        if self.unmixing_matrix is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        X_centered = X - self.mean
        
        X_white = np.dot(self.whitening_matrix, X_centered)
        
        S = np.dot(self.unmixing_matrix, X_white)
        
        return S
    
    def fit_transform(self, X):
        """
        Fit model and separate sources
        
        Parameters:
        -----------
        X : np.ndarray
            Mixed signals
            
        Returns:
        --------
        S : np.ndarray
            Separated sources
        """
        self.fit(X)
        return self.transform(X)
