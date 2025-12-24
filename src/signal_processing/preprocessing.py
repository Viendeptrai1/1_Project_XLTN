"""
Các hàm tiền xử lý cho ICA
- Centering: Loại bỏ giá trị trung bình
- Whitening: Decorrelation và chuẩn hóa phương sai dựa trên PCA
"""

import numpy as np


def centering(X):
    """
    Trung tâm hóa dữ liệu bằng cách trừ đi giá trị trung bình
    
    Tham số:
    --------
    X : np.ndarray
        Ma trận dữ liệu (n_samples, n_features) hoặc (n_features, n_samples)
        
    Trả về:
    -------
    X_centered : np.ndarray
        Dữ liệu đã được trung tâm hóa
    mean : np.ndarray
        Vector giá trị trung bình
    """
    mean = np.mean(X, axis=1, keepdims=True)
    X_centered = X - mean
    return X_centered, mean


def compute_pca(X, n_components=None):
    """
    Tính PCA sử dụng phương pháp phân rã giá trị riêng (eigenvalue decomposition)
    
    Tham số:
    --------
    X : np.ndarray
        Ma trận dữ liệu đã được centered (n_features, n_samples)
    n_components : int, tùy chọn
        Số thành phần cần giữ lại
        
    Trả về:
    -------
    eigenvectors : np.ndarray
        Các thành phần chính (n_features, n_components)
    eigenvalues : np.ndarray
        Các giá trị riêng (n_components,)
    """
    # Tính ma trận hiệp phương sai
    cov_matrix = np.dot(X, X.T) / (X.shape[1] - 1)
    
    # Phân rã eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sắp xếp theo thứ tự giảm dần
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]
    
    return eigenvectors, eigenvalues


def whitening(X, n_components=None):
    """
    Làm trắng dữ liệu sử dụng PCA
    Phép biến đổi: X_white = D^(-1/2) * E^T * X
    trong đó D là ma trận đường chéo của eigenvalues, E là eigenvectors
    
    Tham số:
    --------
    X : np.ndarray
        Ma trận dữ liệu đầu vào (n_features, n_samples)
    n_components : int, tùy chọn
        Số thành phần cần giữ lại
        
    Trả về:
    -------
    X_white : np.ndarray
        Dữ liệu đã được làm trắng
    whitening_matrix : np.ndarray
        Ma trận biến đổi whitening
    dewhitening_matrix : np.ndarray
        Ma trận biến đổi ngược
    """
    X_centered, mean = centering(X)
    
    eigenvectors, eigenvalues = compute_pca(X_centered, n_components)
    
    # Ma trận D^(-1/2)
    epsilon = 1e-10
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + epsilon))
    
    # Ma trận whitening
    whitening_matrix = np.dot(D_inv_sqrt, eigenvectors.T)
    
    # Áp dụng whitening
    X_white = np.dot(whitening_matrix, X_centered)
    
    # Ma trận dewhitening (để khôi phục)
    D_sqrt = np.diag(np.sqrt(eigenvalues + epsilon))
    dewhitening_matrix = np.dot(eigenvectors, D_sqrt)
    
    return X_white, whitening_matrix, dewhitening_matrix, mean
