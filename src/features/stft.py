"""
Short-Time Fourier Transform (STFT) implementation
Using NumPy FFT for frequency analysis
"""

import numpy as np


def create_window(window_length, window_type='hann'):
    """
    Create window function
    
    Parameters:
    -----------
    window_length : int
        Length of the window
    window_type : str
        Type of window ('hann', 'hamming', 'rectangular')
        
    Returns:
    --------
    window : np.ndarray
        Window function
    """
    if window_type == 'hann':
        n = np.arange(window_length)
        window = 0.5 - 0.5 * np.cos(2 * np.pi * n / (window_length - 1))
    elif window_type == 'hamming':
        n = np.arange(window_length)
        window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (window_length - 1))
    elif window_type == 'rectangular':
        window = np.ones(window_length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    return window


def stft(signal, n_fft=512, hop_length=256, window='hann'):
    """
    Compute Short-Time Fourier Transform
    
    Parameters:
    -----------
    signal : np.ndarray
        Input signal (1D array)
    n_fft : int
        FFT size
    hop_length : int
        Number of samples between successive frames
    window : str
        Window type
        
    Returns:
    --------
    stft_matrix : np.ndarray
        Complex STFT matrix (n_fft//2 + 1, n_frames)
    """
    window_func = create_window(n_fft, window)
    
    n_frames = 1 + (len(signal) - n_fft) // hop_length
    
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex128)
    
    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        end = start + n_fft
        
        if end > len(signal):
            frame = np.zeros(n_fft)
            frame[:len(signal) - start] = signal[start:]
        else:
            frame = signal[start:end]
        
        windowed_frame = frame * window_func
        
        fft_result = np.fft.rfft(windowed_frame)
        
        stft_matrix[:, frame_idx] = fft_result
    
    return stft_matrix


def istft(stft_matrix, hop_length=256, window='hann'):
    """
    Inverse Short-Time Fourier Transform
    
    Parameters:
    -----------
    stft_matrix : np.ndarray
        Complex STFT matrix (n_fft//2 + 1, n_frames)
    hop_length : int
        Number of samples between successive frames
    window : str
        Window type
        
    Returns:
    --------
    signal : np.ndarray
        Reconstructed signal
    """
    n_fft = (stft_matrix.shape[0] - 1) * 2
    n_frames = stft_matrix.shape[1]
    
    window_func = create_window(n_fft, window)
    
    signal_length = n_fft + (n_frames - 1) * hop_length
    signal = np.zeros(signal_length)
    window_sum = np.zeros(signal_length)
    
    for frame_idx in range(n_frames):
        ifft_result = np.fft.irfft(stft_matrix[:, frame_idx])
        
        start = frame_idx * hop_length
        end = start + n_fft
        
        signal[start:end] += ifft_result * window_func
        window_sum[start:end] += window_func ** 2
    
    nonzero_indices = window_sum > 1e-10
    signal[nonzero_indices] /= window_sum[nonzero_indices]
    
    return signal
