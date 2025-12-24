"""
Generate Test Dataset - Tạo bộ test data đa dạng từ dataset gốc
Includes: noise, speed changes, volume variations, different mixtures
"""

import os
import numpy as np
from src.signal_processing import load_wav, save_wav, create_mixtures, pad_signals


def add_noise(signal, noise_level=0.05):
    """Thêm nhiễu Gaussian"""
    noise = np.random.randn(len(signal)) * noise_level * np.std(signal)
    return signal + noise


def change_speed(signal, speed_factor=1.2):
    """Thay đổi tốc độ bằng resampling"""
    indices = np.round(np.arange(0, len(signal), speed_factor))
    indices = indices[indices < len(signal)].astype(int)
    return signal[indices]


def change_volume(signal, volume_factor=1.5):
    """Thay đổi âm lượng"""
    return np.clip(signal * volume_factor, -1.0, 1.0)


def generate_test_set():
    """Tạo test dataset đầy đủ"""
    print("=" * 80)
    print("GENERATING TEST DATASET")
    print("=" * 80)
    
    dataset_dir = "tts_dataset_vi"
    output_dir = "test_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all original files
    print("\n[1] Loading original dataset...")
    all_files = {}
    for i in range(10):
        filepath = os.path.join(dataset_dir, f"digit_{i}.wav")
        if os.path.exists(filepath):
            data, sr = load_wav(filepath)
            all_files[f"digit_{i}"] = (data, sr)
            print(f"  ✓ Loaded digit_{i}")
    
    for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        filepath = os.path.join(dataset_dir, f"letter_{letter}.wav")
        if os.path.exists(filepath):
            data, sr = load_wav(filepath)
            all_files[f"letter_{letter}"] = (data, sr)
            print(f"  ✓ Loaded letter_{letter}")
    
    print(f"\nTotal files loaded: {len(all_files)}")
    
    # Test Set 1: Clean originals
    print("\n[2] Test Set 1: Clean Originals")
    clean_dir = os.path.join(output_dir, "01_clean")
    os.makedirs(clean_dir, exist_ok=True)
    
    for name, (data, sr) in all_files.items():
        save_wav(os.path.join(clean_dir, f"{name}.wav"), data, sr)
    
    print(f"  ✓ Saved {len(all_files)} clean files to {clean_dir}")
    
    # Test Set 2: With Noise (3 levels)
    print("\n[3] Test Set 2: Noisy Versions")
    noise_levels = [0.05, 0.1, 0.2]
    
    for noise_level in noise_levels:
        noise_dir = os.path.join(output_dir, f"02_noise_{int(noise_level*100)}pct")
        os.makedirs(noise_dir, exist_ok=True)
        
        for name, (data, sr) in all_files.items():
            noisy = add_noise(data, noise_level)
            save_wav(os.path.join(noise_dir, f"{name}_noisy.wav"), noisy, sr)
        
        print(f"  ✓ Created {len(all_files)} files with {noise_level*100}% noise in {noise_dir}")
    
    # Test Set 3: Speed Variations
    print("\n[4] Test Set 3: Speed Variations")
    speed_factors = [0.8, 1.2, 1.5]
    
    for speed in speed_factors:
        speed_dir = os.path.join(output_dir, f"03_speed_{int(speed*100)}")
        os.makedirs(speed_dir, exist_ok=True)
        
        for name, (data, sr) in all_files.items():
            changed = change_speed(data, speed)
            save_wav(os.path.join(speed_dir, f"{name}_speed{int(speed*100)}.wav"), changed, sr)
        
        print(f"  ✓ Created {len(all_files)} files with {speed}x speed in {speed_dir}")
    
    # Test Set 4: Volume Variations
    print("\n[5] Test Set 4: Volume Variations")
    volume_factors = [0.5, 1.5, 2.0]
    
    for volume in volume_factors:
        vol_dir = os.path.join(output_dir, f"04_volume_{int(volume*100)}")
        os.makedirs(vol_dir, exist_ok=True)
        
        for name, (data, sr) in all_files.items():
            changed = change_volume(data, volume)
            save_wav(os.path.join(vol_dir, f"{name}_vol{int(volume*100)}.wav"), changed, sr)
        
        print(f"  ✓ Created {len(all_files)} files with {volume}x volume in {vol_dir}")
    
    # Test Set 5: Mixed test cases (10 random 5-source mixtures)
    print("\n[6] Test Set 5: Mixture Test Cases")
    mixtures_dir = os.path.join(output_dir, "05_mixtures")
    os.makedirs(mixtures_dir, exist_ok=True)
    
    np.random.seed(42)
    digit_keys = [k for k in all_files.keys() if k.startswith('digit_')]
    
    for test_idx in range(10):
        # Random select 5 sources
        selected = np.random.choice(digit_keys, 5, replace=False)
        sources = [all_files[k][0] for k in selected]
        sr = all_files[selected[0]][1]
        
        # Create mixtures
        mixtures, mixing_matrix = create_mixtures(sources)
        
        # Save mixtures
        test_case_dir = os.path.join(mixtures_dir, f"test_case_{test_idx:02d}")
        os.makedirs(test_case_dir, exist_ok=True)
        
        # Save ground truth sources
        sources_dir = os.path.join(test_case_dir, "sources")
        os.makedirs(sources_dir, exist_ok=True)
        for i, (src_name, src_data) in enumerate(zip(selected, sources)):
            save_wav(os.path.join(sources_dir, f"source_{i}_{src_name}.wav"), src_data, sr)
        
        # Save mixtures
        mixtures_subdir = os.path.join(test_case_dir, "mixtures")
        os.makedirs(mixtures_subdir, exist_ok=True)
        for i, mixture in enumerate(mixtures):
            save_wav(os.path.join(mixtures_subdir, f"mixture_{i}.wav"), mixture, sr)
        
        # Save mixing matrix
        np.save(os.path.join(test_case_dir, "mixing_matrix.npy"), mixing_matrix)
        
        # Save metadata
        with open(os.path.join(test_case_dir, "metadata.txt"), 'w') as f:
            f.write(f"Test Case {test_idx}\n")
            f.write(f"Sources: {', '.join(selected)}\n")
            f.write(f"Mixing Matrix Shape: {mixing_matrix.shape}\n")
    
    print(f"  ✓ Created 10 mixture test cases in {mixtures_dir}")
    
    # Generate summary
    print("\n" + "=" * 80)
    print("TEST DATASET SUMMARY")
    print("=" * 80)
    
    total_files = 0
    for root, dirs, files in os.walk(output_dir):
        wav_files = [f for f in files if f.endswith('.wav')]
        if wav_files:
            print(f"{root}: {len(wav_files)} files")
            total_files += len(wav_files)
    
    print(f"\nTotal WAV files generated: {total_files}")
    print("=" * 80)
    
    return output_dir


if __name__ == "__main__":
    output_dir = generate_test_set()
    print(f"\n✓ Test dataset created in: {output_dir}")
    print("\nYou can now evaluate models on:")
    print("  - Clean audio")
    print("  - Noisy audio (3 levels)")
    print("  - Speed variations (3 speeds)")
    print("  - Volume variations (3 volumes)")
    print("  - Real mixture test cases (10 scenarios)")
