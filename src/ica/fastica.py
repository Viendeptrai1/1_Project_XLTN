"""
Thuật toán FastICA - Phân tích thành phần độc lập
Dựa trên: Hyvarinen & Oja (2000)
"""

import numpy as np
from ..signal_processing.preprocessing import whitening
from .contrast_functions import dg_logcosh, ddg_logcosh


class FastICA:
    """
    FastICA cho phân tách nguồn mù (Blind Source Separation).
    
    Tham số:
        n_components: Số thành phần trích xuất (None = tất cả)
        max_iter: Số vòng lặp tối đa
        tol: Ngưỡng hội tụ
        alpha: Tham số cho hàm logcosh
        random_state: Seed ngẫu nhiên
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
        """Trực giao hóa đối xứng: W = (W * W^T)^(-1/2) * W"""
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        W_orth = np.dot(U, Vt)
        return W_orth
    
    def _ica_parallel(self, X_white):
        """FastICA song song - trích xuất tất cả thành phần cùng lúc"""
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
            print(f"Cảnh báo: FastICA không hội tụ sau {self.max_iter} vòng lặp")
        
        return W
    
    def fit(self, X):
        """Huấn luyện mô hình ICA"""
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
        """Phân tách nguồn từ hỗn hợp"""
        if self.unmixing_matrix is None:
            raise ValueError("Mô hình chưa được huấn luyện. Gọi fit() trước.")
        
        if X.shape[0] > X.shape[1]:
            X = X.T
        
        X_centered = X - self.mean
        X_white = np.dot(self.whitening_matrix, X_centered)
        S = np.dot(self.unmixing_matrix, X_white)
        
        return S
    
    def fit_transform(self, X):
        """Huấn luyện và phân tách nguồn"""
        self.fit(X)
        return self.transform(X)
