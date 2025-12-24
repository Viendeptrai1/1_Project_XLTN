"""
Demo script để test NMF và so sánh với ICA
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.signal_processing import load_wav, create_mixtures, pad_signals
from src.ica import FastICA
from src.nmf import NMF
from src.evaluation import snr, sdr, permutation_solver


def test_nmf_vs_ica():
    """So sánh NMF và ICA"""
    print("=" * 60)
    print("So sánh NMF vs ICA")
    print("=" * 60)
    
    # 1. Load audio files
    print("\n[1] Đang load audio files...")
    dataset_dir = "tts_dataset_vi"
    
    sources = []
    sample_rates = []
    labels = []
    
    for i in range(5):
        filepath = os.path.join(dataset_dir, f"digit_{i}.wav")
        if os.path.exists(filepath):
            data, sr = load_wav(filepath)
            sources.append(data)
            sample_rates.append(sr)
            labels.append(str(i))
            print(f"  ✓ Loaded digit_{i}.wav")
    
    # 2. Create mixtures
    print("\n[2] Tạo mixtures...")
    mixtures, mixing_matrix = create_mixtures(sources)
    print(f"  ✓ Đã tạo {len(mixtures)} mixtures")
    
    # Pad sources
    sources_padded = pad_signals(sources)
    
    # 3. ICA Separation
    print("\n[3] Chạy FastICA...")
    ica = FastICA(n_components=5, max_iter=200, random_state=42)
    ica_separated = ica.fit_transform(mixtures)
    
    ica_aligned, ica_perm, ica_corr = permutation_solver(sources_padded, ica_separated)
    
    print(f"  ✓ ICA converged trong {ica.n_iter} iterations")
    print(f"  ✓ ICA permutation: {ica_perm}")
    
    # Tính metrics cho ICA
    ica_snr = []
    ica_sdr = []
    for i in range(5):
        ica_snr.append(snr(sources_padded[i], ica_aligned[i]))
        ica_sdr.append(sdr(sources_padded[i], ica_aligned[i]))
    
    print(f"  ✓ ICA Average SNR: {np.mean(ica_snr):.2f} dB")
    print(f"  ✓ ICA Average SDR: {np.mean(ica_sdr):.2f} dB")
    
    # 4. NMF Separation
    print("\n[4] Chạy NMF...")
    nmf = NMF(n_components=5, max_iter=200, random_state=42)
    
    # Sử dụng mixture đầu tiên
    nmf_separated = nmf.separate_sources(mixtures[0], sample_rates[0])
    nmf_separated = np.array(nmf_separated)
    
    print(f"  ✓ NMF converged trong {nmf.n_iter} iterations")
    print(f"  ✓ NMF reconstruction error: {nmf.reconstruction_error[-1]:.2f}")
    
    # Align NMF results
    nmf_aligned, nmf_perm, nmf_corr = permutation_solver(sources_padded, nmf_separated)
    print(f"  ✓ NMF permutation: {nmf_perm}")
    
    # Tính metrics cho NMF
    nmf_snr = []
    nmf_sdr = []
    for i in range(5):
        nmf_snr.append(snr(sources_padded[i], nmf_aligned[i]))
        nmf_sdr.append(sdr(sources_padded[i], nmf_aligned[i]))
    
    print(f"  ✓ NMF Average SNR: {np.mean(nmf_snr):.2f} dB")
    print(f"  ✓ NMF Average SDR: {np.mean(nmf_sdr):.2f} dB")
    
    # 5. So sánh kết quả
    print("\n[5] So sánh kết quả:")
    print("\n" + "=" * 60)
    print(f"{'Method':<10} {'Avg SNR (dB)':<15} {'Avg SDR (dB)':<15} {'Avg Correlation':<15}")
    print("=" * 60)
    print(f"{'ICA':<10} {np.mean(ica_snr):<15.2f} {np.mean(ica_sdr):<15.2f} {np.mean(np.diag(ica_corr)):<15.3f}")
    print(f"{'NMF':<10} {np.mean(nmf_snr):<15.2f} {np.mean(nmf_sdr):<15.2f} {np.mean(np.diag(nmf_corr)):<15.3f}")
    print("=" * 60)
    
    # 6. Vẽ biểu đồ so sánh
    print("\n[6] Tạo biểu đồ so sánh...")
    fig, axes = plt.subplots(5, 3, figsize=(15, 12))
    
    for i in range(5):
        # Original
        axes[i, 0].plot(sources_padded[i][:5000])
        axes[i, 0].set_title(f'Original Source {i} ({labels[i]})')
        axes[i, 0].set_ylabel('Amplitude')
        
        # ICA separated
        axes[i, 1].plot(ica_aligned[i][:5000])
        axes[i, 1].set_title(f'ICA Separated (SNR: {ica_snr[i]:.1f}dB)')
        
        # NMF separated
        axes[i, 2].plot(nmf_aligned[i][:5000])
        axes[i, 2].set_title(f'NMF Separated (SNR: {nmf_snr[i]:.1f}dB)')
    
    plt.tight_layout()
    plt.savefig('outputs/nmf_vs_ica_comparison.png', dpi=150)
    print("  ✓ Đã lưu biểu đồ: outputs/nmf_vs_ica_comparison.png")
    
    # 7. Lưu kết quả
    print("\n[7] Lưu kết quả...")
    
    # Lưu NMF separated sources
    os.makedirs('outputs/nmf', exist_ok=True)
    for i, source in enumerate(nmf_aligned):
        from src.signal_processing import save_wav
        save_wav(f'outputs/nmf/nmf_separated_{i}.wav', source, sample_rates[0])
    
    print("  ✓ Đã lưu NMF separated sources vào outputs/nmf/")
    
    print("\n" + "=" * 60)
    print("✓ Hoàn thành so sánh!")
    print("=" * 60)
    
    # 8. Kết luận
    print("\n📊 KẾT LUẬN:")
    if np.mean(ica_snr) > np.mean(nmf_snr):
        print(f"  → ICA tốt hơn NMF (+{np.mean(ica_snr) - np.mean(nmf_snr):.2f} dB SNR)")
    else:
        print(f"  → NMF tốt hơn ICA (+{np.mean(nmf_snr) - np.mean(ica_snr):.2f} dB SNR)")
    
    if np.mean(np.diag(ica_corr)) > np.mean(np.diag(nmf_corr)):
        print(f"  → ICA có correlation cao hơn ({np.mean(np.diag(ica_corr)):.3f} vs {np.mean(np.diag(nmf_corr)):.3f})")
    else:
        print(f"  → NMF có correlation cao hơn ({np.mean(np.diag(nmf_corr)):.3f} vs {np.mean(np.diag(ica_corr)):.3f})")


if __name__ == "__main__":
    test_nmf_vs_ica()
