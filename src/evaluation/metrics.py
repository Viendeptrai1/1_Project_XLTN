"""
Evaluation metrics for source separation quality
"""

import numpy as np


def snr(original, separated):
    """
    Compute Signal-to-Noise Ratio
    
    SNR = 10 * log10(||s||^2 / ||s - s_hat||^2)
    
    Parameters:
    -----------
    original : np.ndarray
        Original source signal
    separated : np.ndarray
        Separated signal
        
    Returns:
    --------
    snr_db : float
        SNR in decibels
    """
    signal_power = np.sum(original ** 2)
    noise = original - separated
    noise_power = np.sum(noise ** 2)
    
    if noise_power < 1e-10:
        return np.inf
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    
    return snr_db


def sdr(original, separated):
    """
    Compute Signal-to-Distortion Ratio
    
    SDR = 10 * log10(||s_target||^2 / ||e_interf + e_artif||^2)
    
    Parameters:
    -----------
    original : np.ndarray
        Original source signal
    separated : np.ndarray
        Separated signal
        
    Returns:
    --------
    sdr_db : float
        SDR in decibels
    """
    target = original
    target_power = np.sum(target ** 2)
    
    distortion = separated - original
    distortion_power = np.sum(distortion ** 2)
    
    if distortion_power < 1e-10:
        return np.inf
    
    sdr_db = 10 * np.log10(target_power / distortion_power)
    
    return sdr_db


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
