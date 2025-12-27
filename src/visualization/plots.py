"""
Các tiện ích trực quan hóa tín hiệu âm thanh và spectrogram
"""

import numpy as np
import matplotlib.pyplot as plt
from ..features.stft import stft


def plot_waveform(signal, sample_rate, title="Waveform", ax=None):
    """Vẽ dạng sóng âm thanh."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    
    time = np.arange(len(signal)) / sample_rate
    ax.plot(time, signal, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    if ax is None:
        plt.tight_layout()
        return fig
    return ax


def plot_spectrogram(signal, sample_rate, title="Spectrogram", ax=None, n_fft=512, hop_length=256):
    """Vẽ spectrogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    stft_matrix = stft(signal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft_matrix)
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    time = np.arange(magnitude_db.shape[1]) * hop_length / sample_rate
    freq = np.arange(magnitude_db.shape[0]) * sample_rate / n_fft
    
    im = ax.imshow(magnitude_db, aspect='auto', origin='lower', 
                   extent=[time[0], time[-1], freq[0], freq[-1]],
                   cmap='viridis')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Magnitude (dB)')
    
    if ax is None:
        plt.tight_layout()
        return fig
    return ax


def plot_mfcc(mfcc_features, sample_rate, hop_length=256, title="MFCC", ax=None):
    """Vẽ MFCC dạng heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    time = np.arange(mfcc_features.shape[1]) * hop_length / sample_rate
    
    im = ax.imshow(mfcc_features, aspect='auto', origin='lower',
                   extent=[time[0], time[-1], 0, mfcc_features.shape[0]],
                   cmap='coolwarm')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MFCC Coefficient')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Value')
    
    if ax is None:
        plt.tight_layout()
        return fig
    return ax


def plot_comparison(originals, mixtures, separated, sample_rate, titles=None):
    """So sánh tín hiệu gốc, hỗn hợp và đã tách."""
    n_sources = len(originals)
    
    fig, axes = plt.subplots(n_sources, 3, figsize=(15, 3 * n_sources))
    
    if n_sources == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_sources):
        title_prefix = titles[i] if titles else f"Source {i+1}"
        
        plot_waveform(originals[i], sample_rate, 
                     f"{title_prefix} - Original", axes[i, 0])
        
        if i < len(mixtures):
            plot_waveform(mixtures[i], sample_rate, 
                         f"Mixture {i+1}", axes[i, 1])
        
        plot_waveform(separated[i], sample_rate, 
                     f"{title_prefix} - Separated", axes[i, 2])
    
    plt.tight_layout()
    return fig


def plot_mixing_matrix(matrix, ax=None):
    """Trực quan hóa ma trận trộn."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto')
    ax.set_xlabel('Source')
    ax.set_ylabel('Mixture')
    ax.set_title('Mixing Matrix')
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.colorbar(im, ax=ax, label='Weight')
    
    if ax is None:
        plt.tight_layout()
        return fig
    return ax
