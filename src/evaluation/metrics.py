"""
Các độ đo đánh giá chất lượng phân tách nguồn
"""

import numpy as np


def snr(original, separated):
    """
    Tính SNR (Signal-to-Noise Ratio) với chuẩn hóa biên độ.
    Công thức: SNR = 10 * log10(||s||² / ||s - ŝ||²)
    Tự động xử lý đảo pha và chuẩn hóa biên độ.
    """
    min_len = min(len(original), len(separated))
    original = original[:min_len]
    separated = separated[:min_len]
    
    # Chuẩn hóa biên độ
    original_rms = np.sqrt(np.mean(original ** 2))
    separated_rms = np.sqrt(np.mean(separated ** 2))
    
    if separated_rms > 1e-10:
        separated = separated * (original_rms / separated_rms)
    
    # Thử cả pha thường và đảo pha
    noise_normal = original - separated
    noise_inverted = original - (-separated)
    
    power_noise_normal = np.sum(noise_normal ** 2)
    power_noise_inverted = np.sum(noise_inverted ** 2)
    
    noise = noise_normal if power_noise_normal < power_noise_inverted else noise_inverted
    
    power_signal = np.sum(original ** 2)
    power_noise = np.sum(noise ** 2)
    
    if power_noise < 1e-10:
        return 100.0
    
    snr_value = 10 * np.log10(power_signal / power_noise)
    return snr_value


def sdr(reference, estimated):
    """
    Tính SDR (Signal-to-Distortion Ratio).
    Công thức: SDR = 10 * log10(||s_target||² / ||e||²)
    """
    min_len = min(len(reference), len(estimated))
    reference = reference[:min_len]
    estimated = estimated[:min_len]
    
    # Chuẩn hóa biên độ
    ref_rms = np.sqrt(np.mean(reference ** 2))
    est_rms = np.sqrt(np.mean(estimated ** 2))
    
    if est_rms > 1e-10:
        estimated = estimated * (ref_rms / est_rms)
    
    # Thử cả 2 pha
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
    Giải quyết bài toán hoán vị - căn chỉnh thứ tự nguồn đã tách.
    ICA không đảm bảo thứ tự nguồn đầu ra.
    Trả về: (aligned_sources, permutation, correlations)
    """
    n_sources = sources.shape[0]
    
    # Tính ma trận tương quan
    correlations = np.zeros((n_sources, n_sources))
    for i in range(n_sources):
        for j in range(n_sources):
            corr = np.abs(np.corrcoef(sources[i], separated_sources[j])[0, 1])
            correlations[i, j] = corr
    
    # Tìm hoán vị tối ưu (greedy)
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
    
    # Sắp xếp lại và sửa pha
    aligned_sources = separated_sources[permutation]
    
    for i in range(n_sources):
        if np.corrcoef(sources[i], aligned_sources[i])[0, 1] < 0:
            aligned_sources[i] *= -1
    
    return aligned_sources, permutation, correlations
