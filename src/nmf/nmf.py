"""
Non-negative Matrix Factorization (NMF) cho tách nguồn âm thanh
Triển khai thuật toán Lee & Seung (2001) sử dụng multiplicative update rules
"""

import numpy as np
from ..features.stft import stft, istft


class NMF:
    """
    Non-negative Matrix Factorization
    
    Phân rã ma trận không âm: V ≈ W * H
    - V: Ma trận magnitude spectrogram (n_freq, n_frames)
    - W: Ma trận cơ sở (basis matrix) (n_freq, n_components)
    - H: Ma trận kích hoạt (activation matrix) (n_components, n_frames)
    
    Tham số:
    --------
    n_components : int
        Số lượng thành phần (nguồn) cần tách
    max_iter : int
        Số vòng lặp tối đa
    tol : float
        Ngưỡng hội tụ (convergence tolerance)
    init : str
        Phương pháp khởi tạo ('random', 'nndsvd')
    random_state : int, tùy chọn
        Seed ngẫu nhiên để tái tạo kết quả
    """
    
    def __init__(self, n_components, max_iter=200, tol=1e-4, init='random', random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.random_state = random_state
        
        self.W = None
        self.H = None
        self.n_iter = 0
        self.reconstruction_error = []
    
    def _initialize_wh(self, V):
        """
        Khởi tạo ma trận W và H
        
        Tham số:
        --------
        V : np.ndarray
            Ma trận magnitude spectrogram
            
        Trả về:
        -------
        W, H : np.ndarray
            Ma trận W và H đã khởi tạo
        """
        F, T = V.shape
        K = self.n_components
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.init == 'random':
            # Khởi tạo ngẫu nhiên với giá trị dương nhỏ
            W = np.random.rand(F, K) * np.sqrt(V.mean() / K) + 1e-4
            H = np.random.rand(K, T) * np.sqrt(V.mean() / K) + 1e-4
        
        elif self.init == 'nndsvd':
            # Non-negative Double SVD (đơn giản hóa)
            U, S, Vt = np.linalg.svd(V, full_matrices=False)
            W = np.abs(U[:, :K]) * np.sqrt(S[:K])[:, np.newaxis].T
            H = np.abs(Vt[:K, :]) * np.sqrt(S[:K])[:, np.newaxis]
            
            # Thêm giá trị nhỏ để tránh zeros
            W += 1e-4
            H += 1e-4
        
        else:
            raise ValueError(f"Phương pháp khởi tạo '{self.init}' không hợp lệ")
        
        return W, H
    
    def _multiplicative_update(self, V, W, H):
        """
        Multiplicative update rules (Lee & Seung 2001)
        
        Cập nhật:
        - H = H * (W^T V) / (W^T W H + epsilon)
        - W = W * (V H^T) / (W H H^T + epsilon)
        
        Tham số:
        --------
        V : np.ndarray
            Ma trận magnitude spectrogram gốc
        W, H : np.ndarray
            Ma trận W và H hiện tại
            
        Trả về:
        -------
        W, H : np.ndarray
            Ma trận W và H đã cập nhật
        """
        epsilon = 1e-10
        
        # Update H
        WtV = W.T @ V
        WtWH = W.T @ W @ H
        H *= WtV / (WtWH + epsilon)
        
        # Update W
        VHt = V @ H.T
        WHHt = W @ H @ H.T
        W *= VHt / (WHHt + epsilon)
        
        return W, H
    
    def _compute_reconstruction_error(self, V, W, H):
        """
        Tính lỗi tái tạo (Frobenius norm)
        
        Error = ||V - WH||_F
        
        Tham số:
        --------
        V : np.ndarray
            Ma trận gốc
        W, H : np.ndarray
            Ma trận phân rã
            
        Trả về:
        -------
        error : float
            Lỗi Frobenius
        """
        reconstruction = W @ H
        error = np.linalg.norm(V - reconstruction, 'fro')
        return error
    
    def fit_transform(self, V):
        """
        Phân rã ma trận V thành W * H
        
        Tham số:
        --------
        V : np.ndarray
            Ma trận magnitude spectrogram (n_freq, n_frames)
            Tất cả giá trị phải >= 0
            
        Trả về:
        -------
        W : np.ndarray
            Ma trận cơ sở (n_freq, n_components)
        H : np.ndarray
            Ma trận kích hoạt (n_components, n_frames)
        """
        # Kiểm tra input
        if np.any(V < 0):
            raise ValueError("Ma trận V phải có tất cả phần tử không âm (non-negative)")
        
        # Khởi tạo W và H
        self.W, self.H = self._initialize_wh(V)
        
        # Iterative optimization
        prev_error = np.inf
        self.reconstruction_error = []
        
        for iteration in range(self.max_iter):
            # Multiplicative updates
            self.W, self.H = self._multiplicative_update(V, self.W, self.H)
            
            # Tính lỗi
            error = self._compute_reconstruction_error(V, self.W, self.H)
            self.reconstruction_error.append(error)
            
            # Kiểm tra hội tụ
            if abs(prev_error - error) < self.tol:
                self.n_iter = iteration + 1
                break
            
            prev_error = error
        else:
            self.n_iter = self.max_iter
        
        return self.W, self.H
    
    def fit(self, V):
        """Fit model (tương tự fit_transform nhưng không return)"""
        self.fit_transform(V)
        return self
    
    def transform(self, V):
        """
        Tính H cho ma trận V mới (với W cố định)
        
        Tham số:
        --------
        V : np.ndarray
            Ma trận magnitude spectrogram mới
            
        Trả về:
        -------
        H : np.ndarray
            Ma trận kích hoạt
        """
        if self.W is None:
            raise ValueError("Model chưa được fit. Gọi fit() hoặc fit_transform() trước.")
        
        # Initialize H
        K, T = self.n_components, V.shape[1]
        H = np.random.rand(K, T) * np.sqrt(V.mean() / K) + 1e-4
        
        epsilon = 1e-10
        
        # Chỉ update H (giữ W cố định)
        for _ in range(100):
            WtV = self.W.T @ V
            WtWH = self.W.T @ self.W @ H
            H *= WtV / (WtWH + epsilon)
        
        return H
    
    def separate_sources(self, mixture_signal, sample_rate, n_fft=512, hop_length=256):
        """
        Tách nguồn từ tín hiệu hỗn hợp sử dụng NMF
        
        Tham số:
        --------
        mixture_signal : np.ndarray
            Tín hiệu hỗn hợp (time domain)
        sample_rate : int
            Tần số lấy mẫu
        n_fft : int
            Kích thước FFT
        hop_length : int
            Hop length cho STFT
            
        Trả về:
        -------
        separated_sources : list of np.ndarray
            Danh sách các nguồn đã tách (time domain)
        """
        # 1. STFT để lấy spectrogram
        mixture_stft = stft(mixture_signal, n_fft=n_fft, hop_length=hop_length)
        mixture_magnitude = np.abs(mixture_stft)
        mixture_phase = np.angle(mixture_stft)
        
        # 2. NMF trên magnitude spectrogram
        W, H = self.fit_transform(mixture_magnitude)
        
        # 3. Tách từng nguồn
        separated_sources = []
        
        for k in range(self.n_components):
            # Tái tạo magnitude spectrogram của nguồn k
            source_magnitude = np.outer(W[:, k], H[k, :])
            
            # Soft masking (normalize)
            total_magnitude = W @ H
            mask = source_magnitude / (total_magnitude + 1e-10)
            
            # Áp dụng mask
            masked_magnitude = mixture_magnitude * mask
            
            # Kết hợp với phase gốc
            source_stft = masked_magnitude * np.exp(1j * mixture_phase)
            
            # Inverse STFT
            source_signal = istft(source_stft, hop_length=hop_length)
            
            # Đảm bảo cùng độ dài với mixture
            if len(source_signal) > len(mixture_signal):
                source_signal = source_signal[:len(mixture_signal)]
            elif len(source_signal) < len(mixture_signal):
                padded = np.zeros(len(mixture_signal))
                padded[:len(source_signal)] = source_signal
                source_signal = padded
            
            separated_sources.append(source_signal)
        
        return separated_sources
    
    def reconstruct(self):
        """
        Tái tạo ma trận V từ W * H
        
        Trả về:
        -------
        V_reconstructed : np.ndarray
            Ma trận tái tạo
        """
        if self.W is None or self.H is None:
            raise ValueError("Model chưa được fit")
        
        return self.W @ self.H
