"""
Single-Channel Source Separation Module

Tách nguồn từ 1 mixture duy nhất sử dụng:
- Tính thưa (sparsity) của speech
- Binary masking trong time-frequency domain
- K-means clustering
"""

import numpy as np
import librosa
from sklearn.cluster import KMeans


class SparseSeparation:
    """
    Single-channel source separation
    
    Phương pháp:
    1. STFT → time-frequency representation
    2. K-means clustering trên magnitude spectrogram
    3. Binary masking cho từng source
    4. iSTFT → waveform
    
    Assumption:
    - Speech rất thưa (supergaussian)
    - Tại mỗi TF bin, chỉ 1 source dominant
    """
    
    def __init__(self, n_sources=2, n_fft=1024, hop_length=256):
        """
        Parameters:
        -----------
        n_sources : int
            Số lượng nguồn cần tách (2-5)
        n_fft : int
            FFT window size
        hop_length : int
            Hop size cho STFT
        """
        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def separate(self, mixture, sr=16000):
        """
        Tách 1 mixture thành n sources
        
        Parameters:
        -----------
        mixture : ndarray, shape (n_samples,)
            Single mixture signal
        sr : int
            Sample rate
            
        Returns:
        --------
        sources : list of ndarrays
            Separated sources (mỗi source có shape (n_samples,))
        """
        print(f"\n[Single-Channel] Tách 1 mixture thành {self.n_sources} sources...")
        
        # Step 1: STFT
        print("[Single-Channel] Step 1: Computing STFT...")
        X = librosa.stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(X)
        phase = np.angle(X)
        
        print(f"[Single-Channel] Spectrogram shape: {magnitude.shape}")
        
        # Step 2: Binary masking với K-means
        print("[Single-Channel] Step 2: Computing binary masks (K-means clustering)...")
        masks = self._compute_binary_masks(magnitude)
        
        # Step 3: Apply masks và iSTFT
        print("[Single-Channel] Step 3: Applying masks and reconstructing...")
        sources = []
        
        for i, mask in enumerate(masks):
            # Apply mask
            S_magnitude = magnitude * mask
            S_complex = S_magnitude * np.exp(1j * phase)
            
            # iSTFT
            s = librosa.istft(S_complex, hop_length=self.hop_length, length=len(mixture))
            sources.append(s)
            
            # Stats
            energy = np.sum(s ** 2)
            print(f"[Single-Channel]   Source {i+1}: energy = {energy:.2e}")
        
        print(f"[Single-Channel] ✓ Done! Separated into {len(sources)} sources")
        
        return sources
    
    def _compute_binary_masks(self, magnitude):
        """
        Tính binary masks dựa trên K-means clustering
        
        Idea:
        - Cluster các TF bins thành K groups
        - Mỗi group tương ứng 1 source
        - Binary mask: 1 nếu thuộc group, 0 nếu không
        
        Parameters:
        -----------
        magnitude : ndarray, shape (n_freq, n_time)
            Magnitude spectrogram
            
        Returns:
        --------
        masks : list of ndarrays
            Binary masks cho từng source
        """
        n_freq, n_time = magnitude.shape
        
        # Transpose để clustering theo time frames
        # Mỗi time frame là 1 feature vector (chiều n_freq)
        features = magnitude.T  # (n_time, n_freq)
        
        # Normalize features (optional, giúp clustering tốt hơn)
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-10)
        
        # K-means clustering
        print(f"[Single-Channel]   Running K-means (k={self.n_sources})...")
        kmeans = KMeans(
            n_clusters=self.n_sources,
            random_state=42,
            n_init=10,
            max_iter=100
        )
        
        labels = kmeans.fit_predict(features_norm)
        
        # Create binary masks
        masks = []
        for i in range(self.n_sources):
            # Mask: 1 where label == i, 0 elsewhere
            mask_1d = (labels == i).astype(float)
            
            # Broadcast to (n_freq, n_time)
            mask = np.tile(mask_1d, (n_freq, 1))
            
            masks.append(mask)
            
            # Stats
            coverage = np.mean(mask) * 100
            print(f"[Single-Channel]   Mask {i+1}: {coverage:.1f}% of TF bins")
        
        return masks


class SparseNMFSeparation:
    """
    Single-channel separation using Sparse NMF
    
    NMF: V ≈ W × H
    - V: magnitude spectrogram
    - W: basis (frequency patterns)
    - H: activation (time patterns)
    
    Với sparsity constraints để force tách rời
    """
    
    def __init__(self, n_sources=2, n_fft=1024, hop_length=256, n_components_per_source=2):
        """
        Parameters:
        -----------
        n_sources : int
            Số nguồn
        n_fft, hop_length : int
            STFT parameters
        n_components_per_source : int
            Số basis vectors per source (overcomplete)
        """
        self.n_sources = n_sources
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_components_per_source = n_components_per_source
    
    def separate(self, mixture, sr=16000):
        """
        Tách bằng Sparse NMF
        """
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
            alpha_W=0.5,  # Sparsity on W
            alpha_H=0.5,  # Sparsity on H
            l1_ratio=0.5,
            max_iter=200,
            verbose=0
        )
        
        W = nmf.fit_transform(V)  # (n_freq, n_components)
        H = nmf.components_        # (n_components, n_time)
        
        print(f"[Sparse NMF] W shape: {W.shape}, H shape: {H.shape}")
        
        # Group components into sources
        print("[Sparse NMF] Step 3: Grouping components...")
        sources = []
        
        for i in range(self.n_sources):
            # Get components for this source
            idx_start = i * self.n_components_per_source
            idx_end = (i + 1) * self.n_components_per_source
            
            # Reconstruct magnitude
            V_source = W[:, idx_start:idx_end] @ H[idx_start:idx_end, :]
            
            # Add phase and iSTFT
            S_complex = V_source * np.exp(1j * phase)
            s = librosa.istft(S_complex, hop_length=self.hop_length, length=len(mixture))
            
            sources.append(s)
            
            energy = np.sum(s ** 2)
            print(f"[Sparse NMF]   Source {i+1}: energy = {energy:.2e}")
        
        print(f"[Sparse NMF] ✓ Done!")
        
        return sources


# Test
if __name__ == '__main__':
    print("=" * 60)
    print("TEST Single-Channel Separation")
    print("=" * 60)
    
    # Tạo synthetic mixture
    np.random.seed(42)
    sr = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, sr * duration)
    
    # 2 sources: 2 sine waves khác tần số
    s1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz
    s2 = np.sin(2 * np.pi * 880 * t)  # 880 Hz
    
    # Modulate amplitude (simulate speech-like sparsity)
    s1 = s1 * np.abs(np.sin(2 * np.pi * 2 * t))  # Slow modulation
    s2 = s2 * np.abs(np.sin(2 * np.pi * 3 * t))
    
    # Mix
    mixture = 0.6 * s1 + 0.4 * s2
    
    print(f"\nMixture: {mixture.shape}, {duration}s @ {sr}Hz")
    
    # Test Binary Masking
    print("\n" + "=" * 60)
    print("Method 1: Binary Masking")
    print("=" * 60)
    
    separator1 = SparseSeparation(n_sources=2)
    sources1 = separator1.separate(mixture, sr=sr)
    
    print(f"\nSeparated {len(sources1)} sources")
    for i, s in enumerate(sources1):
        print(f"  Source {i+1}: {s.shape}, energy={np.sum(s**2):.2e}")
    
    # Test Sparse NMF
    print("\n" + "=" * 60)
    print("Method 2: Sparse NMF")
    print("=" * 60)
    
    separator2 = SparseNMFSeparation(n_sources=2)
    sources2 = separator2.separate(mixture, sr=sr)
    
    print(f"\nSeparated {len(sources2)} sources")
    for i, s in enumerate(sources2):
        print(f"  Source {i+1}: {s.shape}, energy={np.sum(s**2):.2e}")
    
    print("\n" + "=" * 60)
    print("✓ Test completed!")
    print("=" * 60)
