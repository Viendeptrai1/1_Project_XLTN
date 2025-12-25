"""
Evaluation metrics for source separation quality
"""

import numpy as np


def snr(original, separated):
    """
    Calculate Signal-to-Noise Ratio with amplitude normalization
    
    SNR = 10 * log10(||s||^2 / ||s - s_hat||^2)
    
    Now includes:
    - Amplitude normalization (fixes scaling issue)
    - Phase inversion handling (fixes phase flip)
    
    Parameters:
    -----------
    original : np.ndarray
        Original source signal
    separated : np.ndarray
        Separated signal
        
    Returns:
    --------
    snr_db : float
       SNR in decibels (more accurate, reflects perceptual quality)
    """
    # Ensure same length
    min_len = min(len(original), len(separated))
    original = original[:min_len]
    separated = separated[:min_len]
    
    # IMPORTANT: Normalize amplitude to fix scaling issue!
    # Scale separated to match original's RMS energy
    original_rms = np.sqrt(np.mean(original ** 2))
    separated_rms = np.sqrt(np.mean(separated ** 2))
    
    if separated_rms > 1e-10:  # Avoid division by zero
        separated = separated * (original_rms / separated_rms)
    
    # Try both normal and inverted phase, use better one
    noise_normal = original - separated
    noise_inverted = original - (-separated)
    
    power_noise_normal = np.sum(noise_normal ** 2)
    power_noise_inverted = np.sum(noise_inverted ** 2)
    
    # Use the phase that gives less noise
    noise = noise_normal if power_noise_normal < power_noise_inverted else noise_inverted
    
    power_signal = np.sum(original ** 2)
    power_noise = np.sum(noise ** 2)
    
    if power_noise < 1e-10:
        return 100.0  # Perfect separation
    
    snr_value = 10 * np.log10(power_signal / power_noise)
    return snr_value


def sdr(reference, estimated):
    """
    Calculate Signal-to-Distortion Ratio (Simplified version)
    With amplitude normalization for better accuracy
    
    SDR = 10 * log10(||s_target||^2 / ||e_interf + e_artif||^2)
    
    Parameters:
    -----------
    reference : np.ndarray
        Reference signal
    estimated : np.ndarray
        Estimated signal
        
    Returns:
    --------
    sdr_db : float
        SDR in decibels
    """
    # Ensure same length
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]
    
    # Normalize amplitude
    ref_rms = np.sqrt(np.mean(reference ** 2))
    est_rms = np.sqrt(np.mean(estimated ** 2))
    
    if est_rms > 1e-10:
        estimated = estimated * (ref_rms / est_rms)
    
    # Try both phases
    distortion_normal = reference - estimated
    distortion_inverted = reference - (-estimated)
    
    power_dist_normal = np.sum(distortion_normal ** 2)
    power_dist_inverted = np.sum(distortion_inverted ** 2)
    
    distortion = distortion_normal if power_dist_normal < power_dist_inverted else distortion_inverted
    
    power_reference = np.sum(reference ** 2)
    power_distortion = np.sum(distortion ** 2)
    
    if power_distortion < 1e-10:
        return 100.0
    
    sdr_value = 10 * np.log10(power_reference / power_distortion)
    return sdr_value


def permutation_solver(sources, separated_sources):
    """
    Solve permutation problem by finding best alignment
    ICA doesn't guarantee order of separated sources
    
    Parameters:
    -----------
    sources : np.ndarray
        Original sources (n_sources, n_samples)
    separated_sources : np.ndarray
        Separated sources (n_sources, n_samples)
        
    Returns:
    --------
    aligned_sources : np.ndarray
        Reordered separated sources
    permutation : list
        Permutation indices
    correlations : np.ndarray
        Correlation matrix
    """
    n_sources = sources.shape[0]
    
    correlations = np.zeros((n_sources, n_sources))
    
    for i in range(n_sources):
        for j in range(n_sources):
            corr = np.abs(np.corrcoef(sources[i], separated_sources[j])[0, 1])
            correlations[i, j] = corr
    
    permutation = []
    available_indices = list(range(n_sources))
    
    for i in range(n_sources):
        best_idx = -1
        best_corr = -1
        
        for j in available_indices:
            if correlations[i, j] > best_corr:
                best_corr = correlations[i, j]
                best_idx = j
        
        permutation.append(best_idx)
        available_indices.remove(best_idx)
    
    aligned_sources = separated_sources[permutation]
    
    for i in range(n_sources):
        if np.corrcoef(sources[i], aligned_sources[i])[0, 1] < 0:
            aligned_sources[i] *= -1
    
    return aligned_sources, permutation, correlations
