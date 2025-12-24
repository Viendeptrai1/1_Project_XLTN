"""
Các tiện ích trộn audio để tạo hỗn hợp tổng hợp
"""

import numpy as np


def generate_mixing_matrix(n_sources, n_mixtures=None, seed=None):
    """
    Tạo ma trận trộn ngẫu nhiên
    
    Tham số:
    --------
    n_sources : int
        Số lượng tín hiệu nguồn
    n_mixtures : int, tùy chọn
        Số lượng tín hiệu hỗn hợp (mặc định: bằng n_sources)
    seed : int, tùy chọn
        Seed ngẫu nhiên để tái tạo kết quả
        
    Trả về:
    -------
    mixing_matrix : np.ndarray
        Ma trận trộn (n_mixtures, n_sources)
    """
    if n_mixtures is None:
        n_mixtures = n_sources
    
    if seed is not None:
        np.random.seed(seed)
    
    # Tạo ma trận ngẫu nhiên
    mixing_matrix = np.random.randn(n_mixtures, n_sources)
    
    # Chuẩn hóa mỗi hàng (L2 norm)
    for i in range(n_mixtures):
        norm = np.linalg.norm(mixing_matrix[i])
        if norm > 0:
            mixing_matrix[i] /= norm
    
    return mixing_matrix


def pad_signals(signals):
    """
    Đệm các tín hiệu để có cùng độ dài
    
    Tham số:
    --------
    signals : list of np.ndarray
        Danh sách các tín hiệu audio có độ dài khác nhau
        
    Trả về:
    -------
    padded_signals : np.ndarray
        Mảng có shape (n_signals, max_length)
    """
    max_length = max(len(s) for s in signals)
    n_signals = len(signals)
    
    padded_signals = np.zeros((n_signals, max_length))
    
    for i, signal in enumerate(signals):
        padded_signals[i, :len(signal)] = signal
    
    return padded_signals


def create_mixtures(sources, mixing_matrix=None, normalize=True):
    """
    Tạo các tín hiệu hỗn hợp từ các tín hiệu nguồn
    X = A * S
    
    Tham số:
    --------
    sources : np.ndarray hoặc list
        Các tín hiệu nguồn (n_sources, n_samples) hoặc list các arrays
    mixing_matrix : np.ndarray, tùy chọn
        Ma trận trộn (n_mixtures, n_sources)
        Nếu None, ma trận ngẫu nhiên sẽ được tạo
    normalize : bool
        Có chuẩn hóa mixtures về [-1, 1] hay không
        
    Trả về:
    -------
    mixtures : np.ndarray
        Các tín hiệu hỗn hợp (n_mixtures, n_samples)
    mixing_matrix : np.ndarray
        Ma trận trộn đã sử dụng
    """
    if isinstance(sources, list):
        sources = pad_signals(sources)
    
    n_sources, n_samples = sources.shape
    
    if mixing_matrix is None:
        mixing_matrix = generate_mixing_matrix(n_sources)
    
    # Trộn: X = A * S
    mixtures = np.dot(mixing_matrix, sources)
    
    # Chuẩn hóa nếu cần
    if normalize:
        for i in range(len(mixtures)):
            max_val = np.max(np.abs(mixtures[i]))
            if max_val > 0:
                mixtures[i] /= max_val
    
    return mixtures, mixing_matrix
