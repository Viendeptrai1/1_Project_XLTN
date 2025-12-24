"""
Các tiện ích Audio I/O để đọc và ghi file WAV
Chỉ sử dụng thư viện chuẩn của Python (wave module) và NumPy
"""

import wave
import numpy as np


def load_wav(filepath):
    """
    Đọc file WAV và trả về dữ liệu audio dạng NumPy array
    
    Tham số:
    --------
    filepath : str
        Đường dẫn tới file WAV
        
    Trả về:
    -------
    data : np.ndarray
        Tín hiệu audio dạng mảng 1D (mono) hoặc 2D (stereo)
    sample_rate : int
        Tần số lấy mẫu (Hz)
    """
    with wave.open(filepath, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        audio_bytes = wav_file.readframes(n_frames)
        
        if sample_width == 1:
            dtype = np.uint8
            offset = 128
        elif sample_width == 2:
            dtype = np.int16
            offset = 0
        else:
            raise ValueError(f"Sample width không được hỗ trợ: {sample_width}")
        
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        
        if offset:
            audio_data = audio_data.astype(np.float32) - offset
        else:
            audio_data = audio_data.astype(np.float32)
        
        # Chuẩn hóa về khoảng [-1, 1]
        max_val = 2 ** (8 * sample_width - 1)
        audio_data = audio_data / max_val
        
        # Chuyển stereo sang mono nếu cần
        if n_channels > 1:
            audio_data = audio_data.reshape(-1, n_channels)
            audio_data = audio_data[:, 0]
        
    return audio_data, sample_rate


def save_wav(filepath, data, sample_rate):
    """
    Lưu dữ liệu audio thành file WAV
    
    Tham số:
    --------
    filepath : str
        Đường dẫn file đầu ra
    data : np.ndarray
        Tín hiệu audio (mảng 1D)
    sample_rate : int
        Tần số lấy mẫu (Hz)
    """
    # Giới hạn giá trị trong khoảng [-1, 1]
    data = np.clip(data, -1.0, 1.0)
    
    # Chuyển sang int16
    audio_int16 = (data * 32767).astype(np.int16)
    
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())


def normalize_audio(signal):
    """
    Chuẩn hóa tín hiệu audio về khoảng [-1, 1]
    
    Tham số:
    --------
    signal : np.ndarray
        Tín hiệu audio đầu vào
        
    Trả về:
    -------
    normalized : np.ndarray
        Tín hiệu đã chuẩn hóa
    """
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        return signal / max_val
    return signal
