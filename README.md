# Audio Source Separation - FastICA

Hệ thống tách nguồn âm thanh (Cocktail Party Problem) sử dụng thuật toán **FastICA**, được xây dựng hoàn toàn từ đầu (from scratch) với NumPy.

## 🎯 Mục tiêu

Giải quyết bài toán "Cocktail Party" - tách 4-5 nguồn âm thanh từ tín hiệu hỗn hợp, áp dụng trên dữ liệu tiếng Việt (chữ số 0-9 và chữ cái a-z).

## 🏗️ Kiến trúc hệ thống

### 1. Signal Processing
- **Audio I/O**: Đọc/ghi file WAV sử dụng `wave` module
- **Centering**: Trừ mean để có kỳ vọng = 0
- **Whitening**: PCA decorrelation và chuẩn hóa phương sai
- **Mixing**: Tạo ma trận trộn và tín hiệu hỗn hợp

### 2. Feature Extraction
- **STFT**: Short-Time Fourier Transform với window functions
- **MFCC**: Mel-Frequency Cepstral Coefficients (13 hệ số)
  - Mel filterbank
  - DCT (Discrete Cosine Transform)

### 3. FastICA Algorithm
- Thuật toán parallel FastICA
- Contrast functions: logcosh, kurtosis, negentropy
- Symmetric decorrelation
- Permutation solver

### 4. Evaluation
- **SNR**: Signal-to-Noise Ratio
- **SDR**: Signal-to-Distortion Ratio
- Permutation alignment

### 5. Recognition
- **DTW**: Dynamic Time Warping
- Template-based classification

### 6. Visualization
- Waveform plots
- Spectrograms
- MFCC heatmaps
- Mixing matrix visualization

### 7. GUI (Tkinter)
- Tab 1: Mixing - Chọn files và tạo mixtures
- Tab 2: Separation - Chạy FastICA
- Tab 3: Recognition - DTW nhận dạng
- Tab 4: Evaluation - Hiển thị metrics

## 📦 Cài đặt

```bash
# Clone hoặc download project
cd 1_Project_XLTN

# Cài đặt dependencies
pip install -r requirements.txt
```

## 🚀 Sử dụng

### Chạy ứng dụng GUI

```bash
python main.py
```

### Quy trình sử dụng

1. **Tab Mixing**:
   - Click "Select Audio Files" → Chọn 4-5 file WAV
   - Click "Generate Mixtures" → Tạo tín hiệu hỗn hợp
   - (Optional) "Save Mixtures" → Lưu mixtures

2. **Tab Separation**:
   - Điều chỉnh parameters (Max Iterations, Tolerance)
   - Click "Run FastICA" → Tách nguồn
   - Xem kết quả so sánh Original vs Separated
   - (Optional) "Save Separated Sources"

3. **Tab Recognition**:
   - Click "Load Template Dataset" → Chọn thư mục `tts_dataset_vi`
   - Click "Recognize Separated Sources" → Nhận dạng bằng DTW
   - Xem kết quả nhận dạng

4. **Tab Evaluation**:
   - Click "Compute Metrics"
   - Xem SNR/SDR cho từng source và average

## 📊 Cấu trúc thư mục

```
1_Project_XLTN/
├── tts_dataset_vi/          # Dataset gốc (36 files)
│   ├── digit_0.wav
│   ├── digit_1.wav
│   ├── ...
│   ├── letter_A.wav
│   └── ...
├── src/
│   ├── signal_processing/   # Xử lý tín hiệu
│   ├── features/            # Trích xuất đặc trưng
│   ├── ica/                 # FastICA algorithm
│   ├── evaluation/          # Metrics
│   ├── recognition/         # DTW
│   ├── visualization/       # Plots
│   └── gui/                 # Tkinter GUI
├── main.py                  # Entry point
├── requirements.txt
└── README.md
```

## 🧪 Ví dụ sử dụng từ code

```python
from src.signal_processing import load_wav, create_mixtures
from src.ica import FastICA
from src.evaluation import snr, permutation_solver
from src.features import mfcc
from src.recognition import DTWClassifier

# 1. Load audio files
sources = []
for i in range(5):
    data, sr = load_wav(f"tts_dataset_vi/digit_{i}.wav")
    sources.append(data)

# 2. Create mixtures
mixtures, mixing_matrix = create_mixtures(sources)

# 3. Run FastICA
ica = FastICA(n_components=5, max_iter=200)
separated = ica.fit_transform(mixtures)

# 4. Solve permutation
aligned_sources, perm, corr = permutation_solver(sources, separated)

# 5. Evaluate
for i in range(5):
    snr_val = snr(sources[i], aligned_sources[i])
    print(f"Source {i+1} SNR: {snr_val:.2f} dB")

# 6. Recognition with DTW
templates = []
labels = []
for i in range(10):
    data, sr = load_wav(f"tts_dataset_vi/digit_{i}.wav")
    mfcc_feat = mfcc(data, sr)
    templates.append(mfcc_feat.T)
    labels.append(str(i))

classifier = DTWClassifier()
classifier.fit(templates, labels)

# Recognize separated sources
for i, source in enumerate(aligned_sources):
    mfcc_feat = mfcc(source, sr)
    label, distance = classifier.predict_single(mfcc_feat.T)
    print(f"Source {i+1}: {label} (distance: {distance:.2f})")
```

## 📚 Lý thuyết căn bản

### FastICA Algorithm

FastICA tìm các thành phần độc lập thống kê bằng cách:

1. **Preprocessing**:
   - Centering: \`X_c = X - mean(X)\`
   - Whitening: \`X_w = D^{-1/2} E^T X_c\`

2. **ICA Optimization**:
   - Tối đa hóa non-Gaussianity
   - Sử dụng contrast function: \`G(x) = log(cosh(x))\`
   - Update rule: \`w+ = E{x g(w^T x)} - E{g'(w^T x)} w\`

3. **Symmetric Decorrelation**:
   - \`W = (W W^T)^{-1/2} W\`

### MFCC Extraction

1. STFT → Power Spectrum
2. Mel Filterbank (40 filters)
3. Log compression
4. DCT → 13 MFCC coefficients

### DTW Distance

```
DTW(i,j) = cost(i,j) + min(DTW(i-1,j), DTW(i,j-1), DTW(i-1,j-1))
```

## 📖 Tài liệu tham khảo

1. **Hyvärinen & Oja (2000)** - Independent Component Analysis: Algorithms and Applications
2. **F.J. Owens** - Signal Processing of Speech
3. **Jurafsky & Martin** - Speech and Language Processing (Chapter 14)
4. **Rabiner (1993)** - Fundamentals of Speech Recognition

## ⚙️ Requirements

- Python 3.7+
- NumPy >= 1.21.0
- Matplotlib >= 3.4.0
- sounddevice >= 0.4.4

## 🔬 Kết quả mong đợi

- **SNR**: > 10 dB
- **SDR**: > 8 dB
- **Recognition Accuracy**: > 80% (với dataset đơn giản)

## 📝 Ghi chú

- Tất cả thuật toán được code from scratch (không dùng sklearn, librosa, scipy.signal)
- Chỉ sử dụng NumPy FFT và các hàm cơ bản
- Phù hợp cho đồ án học thuật về xử lý tín hiệu và ICA

## 🎓 Điểm nổi bật cho đồ án

✅ Code from scratch hoàn toàn  
✅ Kiến trúc module rõ ràng, dễ mở rộng  
✅ Giao diện trực quan với Tkinter  
✅ End-to-end pipeline: Mixing → Separation → Recognition → Evaluation  
✅ Đánh giá định lượng (SNR/SDR)  
✅ Visualization đầy đủ (waveform, spectrogram, MFCC)  
✅ Nhận dạng kết quả bằng DTW  

---

**Author**: Vien dep trai  
**Version**: 1.0.0  
**License**: MIT
