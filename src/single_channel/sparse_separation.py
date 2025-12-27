"""
Phân tách nguồn đơn kênh (Single-Channel Separation)
Tách nguồn từ 1 mixture duy nhất sử dụng:
- Tính thưa (sparsity) của speech
- Binary masking trong miền thời gian-tần số
- K-means clustering / NMF
"""

import numpy as np
import librosa
from sklearn.cluster import KMeans


class SparseSeparation:
    """
    Phân tách đơn kênh bằng Binary Masking.
    
    Luồng xử lý:
    1. STFT → biểu diễn thời gian-tần số
    2. K-means clustering trên magnitude spectrogram
    3. Binary masking cho từng nguồn
    4. iSTFT → waveform
    """
    
    def __init__(self, n_sources=2, n_fft=1024, hop_length=256):
        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def separate(self, mixture, sr=16000):
        """Tách 1 mixture thành n nguồn. Trả về list các tín hiệu."""
        print(f"\n[Single-Channel] Tách 1 mixture thành {self.n_sources} sources...")
        
        # STFT
        print("[Single-Channel] Step 1: Computing STFT...")
        X = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(X)
        phase = np.angle(X)
        
        print(f"[Single-Channel] Spectrogram shape: {magnitude.shape}")
        
        # Binary masking với K-means
        print("[Single-Channel] Step 2: Computing binary masks (K-means clustering)...")
        masks = self._compute_binary_masks(magnitude)
        
        # Apply masks và iSTFT
        print("[Single-Channel] Step 3: Applying masks and reconstructing...")
        sources = []
        
        for i, mask in enumerate(masks):
            S_magnitude = magnitude * mask
            S_complex = S_magnitude * np.exp(1j * phase)
            s = librosa.istft(S_complex, hop_length=self.hop_length, length=len(mixture))
            sources.append(s)
            
            energy = np.sum(s ** 2)
            print(f"[Single-Channel]   Source {i+1}: energy = {energy:.2e}")
        
        print(f"[Single-Channel] ✓ Done! Separated into {len(sources)} sources")
        return sources
    
    def _compute_binary_masks(self, magnitude):
        """Tính binary masks bằng K-means clustering."""
        n_freq, n_time = magnitude.shape
        
        features = magnitude.T
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        
        print(f"[Single-Channel]   Running K-means (k={self.n_sources})...")
        kmeans = KMeans(
            n_clusters=self.n_sources,
            random_state=42,
            n_init=10,
            max_iter=100
        )
        
        labels = kmeans.fit_predict(features_norm)
        
        masks = []
        for i in range(self.n_sources):
            mask_1d = (labels == i).astype(float)
            mask = np.tile(mask_1d, (n_freq, 1))
            masks.append(mask)
            
            coverage = np.mean(mask) * 100
            print(f"[Single-Channel]   Mask {i+1}: {coverage:.1f}% of TF bins")
        
        return masks


class SparseNMFSeparation:
    """
    Phân tách đơn kênh bằng Sparse NMF.
    
    NMF: V ≈ W × H
    - V: magnitude spectrogram
    - W: basis (mẫu tần số)
    - H: activation (mẫu thời gian)
    """
    
    def __init__(self, n_sources=2, n_fft=1024, hop_length=256, n_components_per_source=2):
        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_components_per_source = n_components_per_source
    
    def separate(self, mixture, sr=16000):
        """Tách bằng Sparse NMF. Trả về list các tín hiệu."""
        print(f"\n[Sparse NMF] Tách 1 mixture thành {self.n_sources} sources...")
        
        # STFT
        print("[Sparse NMF] Step 1: Computing STFT...")
        X = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        V = np.abs(X)
        phase = np.angle(X)
        
        # NMF
        print("[Sparse NMF] Step 2: Running NMF...")
        n_components = self.n_sources * self.n_components_per_source
        
        from sklearn.decomposition import NMF
        
        nmf = NMF(
            n_components=n_components,
            init='random',
            random_state=42,
            alpha_W=0.5,
            alpha_H=0.5,
            l1_ratio=0.5,
            max_iter=200,
            verbose=0
        )
        
        W = nmf.fit_transform(V)
        H = nmf.components_
        
        print(f"[Sparse NMF] W shape: {W.shape}, H shape: {H.shape}")
        
        # Nhóm components thành sources
        print("[Sparse NMF] Step 3: Grouping components...")
        sources = []
        
        for i in range(self.n_sources):
            idx_start = i * self.n_components_per_source
            idx_end = (i + 1) * self.n_components_per_source
            
            V_source = W[:, idx_start:idx_end] @ H[idx_start:idx_end, :]
            S_complex = V_source * np.exp(1j * phase)
            s = librosa.istft(S_complex, hop_length=self.hop_length, length=len(mixture))
            
            sources.append(s)
            
            energy = np.sum(s ** 2)
            print(f"[Sparse NMF]   Source {i+1}: energy = {energy:.2e}")
        
        print(f"[Sparse NMF] ✓ Done!")
        return sources
