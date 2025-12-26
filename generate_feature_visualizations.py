"""
Script để tạo các hình ảnh minh họa chi tiết cho Feature Extraction
Phân tích file digit_0.wav qua từng bước của STFT, MFCC, và LPC
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from scipy.io import wavfile
import os

# Import các module từ src
import sys
sys.path.append('src')
from features.stft import stft, create_window
from features.mfcc import mfcc, hz_to_mel, mel_to_hz, mel_filterbank, dct_matrix
from features.lpc import lpc, preemphasis, autocorrelation, levinson_durbin

# Tạo thư mục output
OUTPUT_DIR = '.gemini/antigravity/brain/2bab40a8-3000-4eeb-b1ef-a49a90ffd607'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load audio file
AUDIO_FILE = 'tts_dataset_vi/digit_0.wav'
signal, sr = librosa.load(AUDIO_FILE, sr=16000)

print(f"Loaded {AUDIO_FILE}")
print(f"Sample rate: {sr} Hz")
print(f"Duration: {len(signal)/sr:.2f} seconds")
print(f"Number of samples: {len(signal)}")
print()

# ============================================================================
# PHẦN 1: STFT - Short-Time Fourier Transform
# ============================================================================
print("=" * 70)
print("PHẦN 1: STFT (Short-Time Fourier Transform)")
print("=" * 70)

# 1.1: Raw Waveform
plt.figure(figsize=(14, 4))
time_axis = np.arange(len(signal)) / sr
plt.plot(time_axis, signal, linewidth=0.5, color='#2E86AB')
plt.xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
plt.ylabel('Biên độ', fontsize=12, fontweight='bold')
plt.title('1.1 - Tín hiệu âm thanh thô từ file digit_0.wav', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_raw_waveform.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 01_raw_waveform.png")

# 1.2: Windowing example
n_fft = 512
hop_length = 256
window = create_window(n_fft, 'hamming')

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Show a single frame
frame_start = 1000
frame_end = frame_start + n_fft
frame = signal[frame_start:frame_end]

# Original frame
axes[0].plot(frame, linewidth=1.5, color='#2E86AB', label='Khung gốc')
axes[0].set_xlabel('Số mẫu', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Biên độ', fontsize=11, fontweight='bold')
axes[0].set_title('Khung tín hiệu gốc (25ms, 512 mẫu)', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Windowed frame
windowed = frame * window
axes[1].plot(frame, linewidth=1, color='#A8DADC', alpha=0.5, label='Khung gốc')
axes[1].plot(window, linewidth=1.5, color='#F77F00', label='Hàm cửa sổ Hamming', linestyle='--')
axes[1].plot(windowed, linewidth=1.5, color='#E63946', label='Khung sau khi nhân cửa sổ')
axes[1].set_xlabel('Số mẫu', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Biên độ', fontsize=11, fontweight='bold')
axes[1].set_title('Áp dụng hàm cửa sổ Hamming (giảm rò rỉ phổ)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.suptitle('1.2 - Framing & Windowing (Chia khung và Nhân hàm cửa sổ)', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_windowing.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 02_windowing.png")

# 1.3: STFT Spectrogram
stft_matrix = stft(signal, n_fft=n_fft, hop_length=hop_length, window='hamming')
magnitude = np.abs(stft_matrix)
magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

plt.figure(figsize=(14, 6))
librosa.display.specshow(magnitude_db, sr=sr, hop_length=hop_length, 
                         x_axis='time', y_axis='hz', cmap='viridis')
plt.colorbar(format='%+2.0f dB', label='Cường độ (dB)')
plt.xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
plt.ylabel('Tần số (Hz)', fontsize=12, fontweight='bold')
plt.title('1.3 - STFT Spectrogram: Biểu diễn thời gian-tần số', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_stft_spectrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 03_stft_spectrogram.png")

# ============================================================================
# PHẦN 2: MFCC - Mel-Frequency Cepstral Coefficients
# ============================================================================
print("\n" + "=" * 70)
print("PHẦN 2: MFCC (Mel-Frequency Cepstral Coefficients)")
print("=" * 70)

# 2.1: Pre-emphasis
alpha = 0.97
emphasized = preemphasis(signal, alpha=alpha)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))
axes[0].plot(time_axis[:2000], signal[:2000], linewidth=1, color='#2E86AB')
axes[0].set_ylabel('Biên độ', fontsize=11, fontweight='bold')
axes[0].set_title('Tín hiệu gốc', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(time_axis[:2000], emphasized[:2000], linewidth=1, color='#E63946')
axes[1].set_xlabel('Thời gian (s)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Biên độ', fontsize=11, fontweight='bold')
axes[1].set_title(f'Sau Pre-emphasis (α={alpha}) - Tăng cường tần số cao', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('2.1 - Pre-emphasis Filter: y[n] = x[n] - α·x[n-1]', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_preemphasis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 04_preemphasis.png")

# 2.2: Power Spectrum
power_spectrum = magnitude ** 2

plt.figure(figsize=(14, 6))
librosa.display.specshow(librosa.amplitude_to_db(power_spectrum, ref=np.max), 
                         sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='hz', cmap='magma')
plt.colorbar(format='%+2.0f dB', label='Năng lượng (dB)')
plt.xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
plt.ylabel('Tần số (Hz)', fontsize=12, fontweight='bold')
plt.title('2.2 - Power Spectrum: |FFT|² (Phổ năng lượng)', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_power_spectrum.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 05_power_spectrum.png")

# 2.3: Mel Filterbank
n_filters = 40
mel_filters = mel_filterbank(n_filters, n_fft, sr)

# Tạo visualization cho Mel filterbank
plt.figure(figsize=(14, 6))
freq_bins = np.linspace(0, sr/2, n_fft//2 + 1)
for i in range(0, n_filters, 2):  # Plot every 2nd filter để tránh quá lộn xộn
    plt.plot(freq_bins, mel_filters[i], linewidth=1.5, alpha=0.7)

plt.xlabel('Tần số (Hz)', fontsize=12, fontweight='bold')
plt.ylabel('Độ lợi', fontsize=12, fontweight='bold')
plt.title('2.3 - Mel Filterbank: Bộ lọc tam giác trên thang Mel (40 filters)', 
          fontsize=14, fontweight='bold', pad=15)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_mel_filterbank.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 06_mel_filterbank.png")

# 2.4: Mel spectrum conversion visualization
# Vẽ so sánh thang Hz vs thang Mel
hz_range = np.linspace(0, 8000, 1000)
mel_range = hz_to_mel(hz_range)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear Hz scale
axes[0].plot(hz_range, hz_range, linewidth=2, color='#2E86AB')
axes[0].set_xlabel('Tần số (Hz)', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Tần số (Hz)', fontsize=11, fontweight='bold')
axes[0].set_title('Thang tuyến tính Hz (không phù hợp với tai người)', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Mel scale
axes[1].plot(hz_range, mel_range, linewidth=2, color='#E63946')
axes[1].set_xlabel('Tần số (Hz)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Tần số (Mel)', fontsize=11, fontweight='bold')
axes[1].set_title('Thang Mel (phi tuyến tính, mô phỏng tai người)', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('2.4 - Chuyển đổi thang Hz → Mel: Mel = 2595·log₁₀(1 + Hz/700)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_mel_scale_conversion.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 07_mel_scale_conversion.png")

# 2.5: Mel Spectrum
mel_spectrum = np.dot(mel_filters, power_spectrum)

plt.figure(figsize=(14, 6))
librosa.display.specshow(librosa.amplitude_to_db(mel_spectrum, ref=np.max), 
                         sr=sr, hop_length=hop_length,
                         x_axis='time', cmap='inferno')
plt.colorbar(format='%+2.0f dB', label='Năng lượng (dB)')
plt.xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
plt.ylabel('Mel band', fontsize=12, fontweight='bold')
plt.title('2.5 - Mel Spectrum: Năng lượng qua các bộ lọc Mel', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_mel_spectrum.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 08_mel_spectrum.png")

# 2.6: Log Mel Spectrum
log_mel_spectrum = np.log(mel_spectrum + 1e-10)

plt.figure(figsize=(14, 6))
librosa.display.specshow(log_mel_spectrum, sr=sr, hop_length=hop_length,
                         x_axis='time', cmap='RdYlBu_r')
plt.colorbar(label='log(Năng lượng)')
plt.xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
plt.ylabel('Mel band', fontsize=12, fontweight='bold')
plt.title('2.6 - Log Mel Spectrum: log(Năng lượng) - Mô phỏng độ động tai người', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/09_log_mel_spectrum.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 09_log_mel_spectrum.png")

# 2.7: DCT Matrix visualization
n_mfcc = 13
dct_mat = dct_matrix(n_filters, n_mfcc)

plt.figure(figsize=(10, 8))
plt.imshow(dct_mat, aspect='auto', cmap='seismic', interpolation='nearest')
plt.colorbar(label='Trọng số DCT')
plt.xlabel('Mel band (40 filters)', fontsize=12, fontweight='bold')
plt.ylabel('MFCC coefficient', fontsize=12, fontweight='bold')
plt.title('2.7 - DCT Matrix: Biến đổi Cosine rời rạc (phi tương quan hóa)', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/10_dct_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 10_dct_matrix.png")

# 2.8: Final MFCC features
mfcc_features = mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft, 
                     hop_length=hop_length, n_filters=n_filters)

plt.figure(figsize=(14, 6))
librosa.display.specshow(mfcc_features, sr=sr, hop_length=hop_length,
                         x_axis='time', cmap='coolwarm')
plt.colorbar(label='Giá trị MFCC')
plt.xlabel('Thời gian (s)', fontsize=12, fontweight='bold')
plt.ylabel('MFCC coefficient', fontsize=12, fontweight='bold')
plt.title('2.8 - MFCC Features: Vector đặc trưng cuối cùng (13 coefficients)', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/11_mfcc_features.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 11_mfcc_features.png")

# ============================================================================
# PHẦN 3: LPC - Linear Predictive Coding
# ============================================================================
print("\n" + "=" * 70)
print("PHẦN 3: LPC (Linear Predictive Coding)")
print("=" * 70)

# 3.1: LPC coefficients
lpc_order = 12
lpc_features = lpc(signal, sr, order=lpc_order, frame_length=400, hop_length=160)

plt.figure(figsize=(14, 6))
plt.imshow(lpc_features.T, aspect='auto', origin='lower', cmap='plasma', 
           interpolation='nearest')
plt.colorbar(label='Giá trị hệ số LPC')
plt.xlabel('Frame index', fontsize=12, fontweight='bold')
plt.ylabel('LPC coefficient', fontsize=12, fontweight='bold')
plt.title('3.1 - LPC Coefficients: Mô hình hóa đường phát âm (12 coefficients)', 
          fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/12_lpc_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 12_lpc_coefficients.png")

# 3.2: Autocorrelation example cho một frame
frame_length = 400
test_frame = signal[1000:1000+frame_length]
emphasized_frame = preemphasis(test_frame, alpha=0.97)
window_func = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_length) / frame_length)
windowed_frame = emphasized_frame * window_func
autocorr = autocorrelation(windowed_frame, max_lag=lpc_order)

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# Windowed frame
axes[0].plot(windowed_frame, linewidth=1, color='#2E86AB')
axes[0].set_ylabel('Biên độ', fontsize=11, fontweight='bold')
axes[0].set_title('Khung tín hiệu sau Pre-emphasis và Windowing', fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# Autocorrelation
axes[1].stem(range(len(autocorr)), autocorr, linefmt='#E63946', 
             markerfmt='o', basefmt=' ')
axes[1].set_xlabel('Lag (k)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('R[k]', fontsize=11, fontweight='bold')
axes[1].set_title('Hàm tự tương quan R[k] (dùng để tìm LPC coefficients)', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.suptitle('3.2 - Autocorrelation: Đo độ tương đồng giữa tín hiệu và phiên bản trễ', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/13_autocorrelation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: 13_autocorrelation.png")

print("\n" + "=" * 70)
print("✅ Hoàn thành! Đã tạo 13 hình ảnh minh họa chi tiết")
print("=" * 70)
print(f"\nTất cả hình ảnh đã được lưu trong: {OUTPUT_DIR}/")
print("\nDanh sách file:")
for i in range(1, 14):
    print(f"  {i:02d}_*.png")
