"""
MFCC (Mel-Frequency Cepstral Coefficients) extraction
Implemented from scratch following speech processing textbooks
"""

import numpy as np
from .stft import stft


def hz_to_mel(hz):
    """Convert Hz to Mel scale"""
    return 2595 * np.log10(1 + hz / 700.0)


def mel_to_hz(mel):
    """Convert Mel scale to Hz"""
    return 700 * (10 ** (mel / 2595.0) - 1)


def mel_filterbank(n_filters, n_fft, sample_rate, fmin=0, fmax=None):
    """
    Create Mel filterbank
    
    Parameters:
    -----------
    n_filters : int
        Number of Mel filters
    n_fft : int
        FFT size
    sample_rate : int
        Sample rate in Hz
    fmin : float
        Minimum frequency
    fmax : float
        Maximum frequency (default: sample_rate/2)
        
    Returns:
    --------
    filterbank : np.ndarray
        Mel filterbank matrix (n_filters, n_fft//2 + 1)
    """
    if fmax is None:
        fmax = sample_rate / 2.0
    
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    
    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    
    for m in range(1, n_filters + 1):
        f_left = bin_points[m - 1]
        f_center = bin_points[m]
        f_right = bin_points[m + 1]
        
        for k in range(f_left, f_center):
            filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
        
        for k in range(f_center, f_right):
            filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
    
    return filterbank


def dct_matrix(n_filters, n_mfcc):
    """
    Create Discrete Cosine Transform matrix
    
    Parameters:
    -----------
    n_filters : int
        Number of input features
    n_mfcc : int
        Number of output MFCC coefficients
        
    Returns:
    --------
    dct_mat : np.ndarray
        DCT matrix (n_mfcc, n_filters)
    """
    dct_mat = np.zeros((n_mfcc, n_filters))
    
    for i in range(n_mfcc):
        for j in range(n_filters):
            dct_mat[i, j] = np.cos(np.pi * i * (j + 0.5) / n_filters)
        
        if i == 0:
            dct_mat[i, :] *= np.sqrt(1 / n_filters)
        else:
            dct_mat[i, :] *= np.sqrt(2 / n_filters)
    
    return dct_mat


def mfcc(signal, sample_rate, n_mfcc=13, n_fft=512, hop_length=256, n_filters=40):
    """
    Extract MFCC features from audio signal
    
    Parameters:
    -----------
    signal : np.ndarray
        Input audio signal
    sample_rate : int
        Sample rate in Hz
    n_mfcc : int
        Number of MFCC coefficients
    n_fft : int
        FFT size
    hop_length : int
        Hop length for STFT
    n_filters : int
        Number of Mel filters
        
    Returns:
    --------
    mfcc_features : np.ndarray
        MFCC features (n_mfcc, n_frames)
    """
    stft_matrix = stft(signal, n_fft=n_fft, hop_length=hop_length)
    
    power_spectrum = np.abs(stft_matrix) ** 2
    
    mel_filters = mel_filterbank(n_filters, n_fft, sample_rate)
    
    mel_spectrum = np.dot(mel_filters, power_spectrum)
    
    log_mel_spectrum = np.log(mel_spectrum + 1e-10)
    
    dct_mat = dct_matrix(n_filters, n_mfcc)
    
    mfcc_features = np.dot(dct_mat, log_mel_spectrum)
    
    return mfcc_features
