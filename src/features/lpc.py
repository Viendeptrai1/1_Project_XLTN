"""
Linear Predictive Coding (LPC) Feature Extraction

LPC models the vocal tract and extracts speech features by predicting
samples as a linear combination of previous samples. Based on professor's
template code with improvements for efficiency and frame-based processing.

Algorithm Flow:
  Audio → Frame → Pre-emphasis → Hamming Window → 
  Autocorrelation → Levinson-Durbin → LPC Coefficients
"""

import numpy as np
import librosa


def preemphasis(signal, alpha=0.97):
    """
    Apply pre-emphasis filter to boost high frequencies.
    
    Formula: y[n] = x[n] - α*x[n-1]
    
    This emphasizes formants in speech and compensates for the
    natural -6dB/octave roll-off in speech production.
    
    Args:
        signal: Input audio signal (1D array)
        alpha: Pre-emphasis coefficient (default 0.97)
               Higher values = more emphasis on high frequencies
    
    Returns:
        Filtered signal (same shape as input)
    """
    emphasized = np.zeros_like(signal, dtype=np.float32)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - alpha * signal[:-1]
    return emphasized


def hamming_window(length):
    """
    Create a Hamming window of given length.
    
    Formula: w[n] = 0.54 - 0.46*cos(2πn/N)
    
    Hamming window reduces spectral leakage and improves
    frequency resolution in spectral analysis.
    
    Args:
        length: Window length in samples
    
    Returns:
        Hamming window array of shape (length,)
    """
    n = np.arange(length, dtype=np.float32)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / length)
    return window


def autocorrelation(signal, max_lag):
    """
    Compute autocorrelation coefficients.
    
    The autocorrelation function measures similarity between
    a signal and a delayed version of itself.
    
    Args:
        signal: Input signal (1D array)
        max_lag: Maximum lag (number of coefficients to compute)
    
    Returns:
        Autocorrelation coefficients R[0], R[1], ..., R[max_lag]
        Shape: (max_lag + 1,)
    """
    return librosa.autocorrelate(signal, max_size=max_lag + 1)


def levinson_durbin(r, order):
    """
    Solve the Toeplitz system using Levinson-Durbin recursion.
    
    This finds LPC coefficients by solving:
        R * a = r
    where R is the autocorrelation Toeplitz matrix.
    
    Levinson-Durbin is O(n²) vs O(n³) for matrix inversion,
    making it much more efficient than the professor's approach.
    
    Args:
        r: Autocorrelation coefficients [R[0], R[1], ..., R[order]]
        order: LPC order (number of coefficients to compute)
    
    Returns:
        LPC coefficients a[1], a[2], ..., a[order]
        Shape: (order,)
        
    Reference:
        Durbin, J. (1960). "The fitting of time-series models"
    """
    # Initialize
    a = np.zeros(order, dtype=np.float32)
    e = r[0]  # Prediction error
    
    for i in range(order):
        # Calculate reflection coefficient
        lambda_i = r[i + 1]
        for j in range(i):
            lambda_i -= a[j] * r[i - j]
        lambda_i /= e
        
        # Update coefficients
        a[i] = lambda_i
        for j in range(i // 2 + 1):
            temp = a[j]
            a[j] -= lambda_i * a[i - j - 1]
            if j != i - j - 1:
                a[i - j - 1] -= lambda_i * temp
        
        # Update prediction error
        e *= (1 - lambda_i * lambda_i)
    
    return a


def lpc(signal, sr, order=12, frame_length=400, hop_length=160):
    """
    Extract LPC features from audio signal using frame-based processing.
    
    This is the main interface for LPC feature extraction, compatible
    with existing DTW classifier input format (similar to MFCC).
    
    Process:
        1. Divide signal into overlapping frames
        2. For each frame:
           - Apply pre-emphasis filter
           - Apply Hamming window
           - Compute autocorrelation
           - Solve for LPC coefficients using Levinson-Durbin
    
    Args:
        signal: Input audio signal (1D array)
        sr: Sample rate (Hz)
        order: LPC order (default 12, standard for 16kHz ~= 1 coeff per kHz)
        frame_length: Frame size in samples (default 400 ~= 25ms at 16kHz)
        hop_length: Hop size between frames (default 160 ~= 10ms at 16kHz)
    
    Returns:
        LPC coefficients array of shape (n_frames, order)
        Each row contains LPC coefficients for one frame
        
    Example:
        >>> signal, sr = load_wav('audio.wav')
        >>> lpc_features = lpc(signal, sr, order=12)
        >>> print(lpc_features.shape)  # (n_frames, 12)
    
    Notes:
        - Standard LPC order = sample_rate / 1000
          (e.g., 16 for 16kHz, 8 for 8kHz)
        - Frame overlap helps capture transitions between phonemes
        - Output format matches MFCC for easy swapping in recognition pipeline
    """
    # Normalize signal to [-1, 1] if needed
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    elif np.abs(signal).max() > 1.0:
        signal = signal / np.abs(signal).max()
    
    # Calculate number of frames
    n_samples = len(signal)
    n_frames = 1 + (n_samples - frame_length) // hop_length
    
    # Initialize output
    lpc_coeffs = np.zeros((n_frames, order), dtype=np.float32)
    
    # Create Hamming window once
    window = hamming_window(frame_length)
    
    # Process each frame
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        
        # Extract frame
        frame = signal[start:end]
        
        # Apply pre-emphasis
        emphasized = preemphasis(frame, alpha=0.97)
        
        # Apply Hamming window
        windowed = emphasized * window
        
        # Compute autocorrelation
        r = autocorrelation(windowed, max_lag=order)
        
        # Solve for LPC coefficients using Levinson-Durbin
        # Handle edge case where autocorrelation is zero
        if r[0] < 1e-10:
            coeffs = np.zeros(order, dtype=np.float32)
        else:
            coeffs = levinson_durbin(r, order)
        
        lpc_coeffs[i] = coeffs
    
    return lpc_coeffs


def lpc_to_cepstrum(lpc_coeffs, n_cepstral=18):
    """
    Convert LPC coefficients to cepstral coefficients.
    
    This implements the recursion from professor's my_lpc_3.py.
    Cepstral coefficients can be used for distance measures.
    
    Args:
        lpc_coeffs: LPC coefficients a[1], ..., a[p] (1D array)
        n_cepstral: Number of cepstral coefficients to compute
    
    Returns:
        Cepstral coefficients c[1], ..., c[n_cepstral]
        Shape: (n_cepstral,)
    """
    p = len(lpc_coeffs)
    c = np.zeros(n_cepstral + 1, dtype=np.float32)
    
    # First p coefficients
    for m in range(1, min(p + 1, n_cepstral + 1)):
        c[m] = lpc_coeffs[m - 1]
        for k in range(1, m):
            c[m] += (k / m) * c[k] * lpc_coeffs[m - k - 1]
    
    # Remaining coefficients (if n_cepstral > p)
    for m in range(p + 1, n_cepstral + 1):
        for k in range(1, m):
            if m - k <= p:
                c[m] += (k / m) * c[k] * lpc_coeffs[m - k - 1]
    
    return c[1:]  # Return c[1] to c[n_cepstral]


if __name__ == '__main__':
    # Quick test
    print("LPC Feature Extraction Module")
    print("=" * 50)
    print("Available functions:")
    print("  - preemphasis(signal, alpha)")
    print("  - hamming_window(length)")
    print("  - autocorrelation(signal, max_lag)")
    print("  - levinson_durbin(r, order)")
    print("  - lpc(signal, sr, order, frame_length, hop_length)")
    print("  - lpc_to_cepstrum(lpc_coeffs, n_cepstral)")
    print("\nExample usage:")
    print("  from src.features.lpc import lpc")
    print("  from src.signal_processing import load_wav")
    print("  signal, sr = load_wav('audio.wav')")
    print("  features = lpc(signal, sr, order=12)")
    print("  print(features.shape)  # (n_frames, 12)")
