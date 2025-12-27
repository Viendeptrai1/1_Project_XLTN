"""
Trích xuất đặc trưng LPC (Linear Predictive Coding)
Mô hình hóa bộ máy phát âm, dự đoán mẫu hiện tại từ các mẫu trước đó.

Luồng xử lý:
  Audio → Chia frame → Tiền nhấn → Cửa sổ Hamming → 
  Tự tương quan → Levinson-Durbin → Hệ số LPC
"""

import numpy as np
import librosa


def preemphasis(signal, alpha=0.97):
    """
    Bộ lọc tiền nhấn - tăng cường tần số cao.
    Công thức: y[n] = x[n] - α*x[n-1]
    """
    emphasized = np.zeros_like(signal, dtype=np.float32)
    emphasized[0] = signal[0]
    emphasized[1:] = signal[1:] - alpha * signal[:-1]
    return emphasized


def hamming_window(length):
    """
    Tạo cửa sổ Hamming - giảm rò rỉ phổ.
    Công thức: w[n] = 0.54 - 0.46*cos(2πn/N)
    """
    n = np.arange(length, dtype=np.float32)
    window = 0.54 - 0.46 * np.cos(2 * np.pi * n / length)
    return window


def autocorrelation(signal, max_lag):
    """
    Tính hệ số tự tương quan.
    Đo độ tương đồng giữa tín hiệu và phiên bản trễ của nó.
    """
    return librosa.autocorrelate(signal, max_size=max_lag + 1)


def levinson_durbin(r, order):
    """
    Giải hệ Toeplitz bằng đệ quy Levinson-Durbin.
    Tìm hệ số LPC từ hệ số tự tương quan.
    Độ phức tạp O(n²) thay vì O(n³) của nghịch đảo ma trận.
    """
    a = np.zeros(order, dtype=np.float32)
    e = r[0]  # Sai số dự đoán
    
    for i in range(order):
        # Tính hệ số phản xạ
        lambda_i = r[i + 1]
        for j in range(i):
            lambda_i -= a[j] * r[i - j]
        lambda_i /= e
        
        # Cập nhật hệ số
        a[i] = lambda_i
        for j in range(i // 2 + 1):
            temp = a[j]
            a[j] -= lambda_i * a[i - j - 1]
            if j != i - j - 1:
                a[i - j - 1] -= lambda_i * temp
        
        # Cập nhật sai số
        e *= (1 - lambda_i * lambda_i)
    
    return a


def lpc(signal, sr, order=12, frame_length=400, hop_length=160):
    """
    Trích xuất đặc trưng LPC theo từng frame.
    
    Tham số:
        signal: Tín hiệu âm thanh
        sr: Tần số lấy mẫu (Hz)
        order: Bậc LPC (mặc định 12, ~1 hệ số/kHz)
        frame_length: Độ dài frame (mặc định 400 ~25ms ở 16kHz)
        hop_length: Bước nhảy frame (mặc định 160 ~10ms ở 16kHz)
    
    Trả về:
        Ma trận hệ số LPC có shape (n_frames, order)
    """
    # Chuẩn hóa về [-1, 1]
    if signal.dtype == np.int16:
        signal = signal.astype(np.float32) / 32768.0
    elif np.abs(signal).max() > 1.0:
        signal = signal / np.abs(signal).max()
    
    # Tính số frame
    n_samples = len(signal)
    n_frames = 1 + (n_samples - frame_length) // hop_length
    
    # Khởi tạo output
    lpc_coeffs = np.zeros((n_frames, order), dtype=np.float32)
    
    # Tạo cửa sổ Hamming
    window = hamming_window(frame_length)
    
    # Xử lý từng frame
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        
        frame = signal[start:end]
        emphasized = preemphasis(frame, alpha=0.97)
        windowed = emphasized * window
        r = autocorrelation(windowed, max_lag=order)
        
        if r[0] < 1e-10:
            coeffs = np.zeros(order, dtype=np.float32)
        else:
            coeffs = levinson_durbin(r, order)
        
        lpc_coeffs[i] = coeffs
    
    return lpc_coeffs


def lpc_to_cepstrum(lpc_coeffs, n_cepstral=18):
    """
    Chuyển hệ số LPC sang hệ số cepstral.
    Dùng cho đo khoảng cách trong nhận dạng.
    """
    p = len(lpc_coeffs)
    c = np.zeros(n_cepstral + 1, dtype=np.float32)
    
    for m in range(1, min(p + 1, n_cepstral + 1)):
        c[m] = lpc_coeffs[m - 1]
        for k in range(1, m):
            c[m] += (k / m) * c[k] * lpc_coeffs[m - k - 1]
    
    for m in range(p + 1, n_cepstral + 1):
        for k in range(1, m):
            if m - k <= p:
                c[m] += (k / m) * c[k] * lpc_coeffs[m - k - 1]
    
    return c[1:]
