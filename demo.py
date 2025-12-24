"""
Demo script to test the audio source separation system
Tests the complete pipeline without GUI
"""

import os
import numpy as np
from src.signal_processing import load_wav, create_mixtures, save_wav
from src.ica import FastICA
from src.evaluation import snr, sdr, permutation_solver
from src.features import mfcc
from src.recognition import DTWClassifier


def test_basic_pipeline():
    """Test basic mixing and separation"""
    print("=" * 60)
    print("Testing Audio Source Separation Pipeline")
    print("=" * 60)
    
    # 1. Load audio files
    print("\n[1] Loading audio files...")
    dataset_dir = "tts_dataset_vi"
    
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found!")
        return
    
    sources = []
    sample_rates = []
    labels = []
    
    # Load 5 digit files
    for i in range(5):
        filepath = os.path.join(dataset_dir, f"digit_{i}.wav")
        if os.path.exists(filepath):
            data, sr = load_wav(filepath)
            sources.append(data)
            sample_rates.append(sr)
            labels.append(str(i))
            print(f"  ✓ Loaded digit_{i}.wav ({len(data)} samples, {sr} Hz)")
        else:
            print(f"  ✗ File not found: {filepath}")
    
    if len(sources) < 5:
        print("\nError: Need at least 5 audio files!")
        return
    
    # 2. Create mixtures
    print("\n[2] Creating mixtures...")
    mixtures, mixing_matrix = create_mixtures(sources)
    print(f"  ✓ Created {len(mixtures)} mixtures")
    print(f"  ✓ Mixing matrix shape: {mixing_matrix.shape}")
    print("\n  Mixing Matrix:")
    print(mixing_matrix)
    
    # 3. Run FastICA
    print("\n[3] Running FastICA...")
    ica = FastICA(n_components=5, max_iter=200, tol=1e-4, random_state=42)
    separated = ica.fit_transform(mixtures)
    print(f"  ✓ FastICA converged in {ica.n_iter} iterations")
    print(f"  ✓ Separated sources shape: {separated.shape}")
    
    # 4. Solve permutation
    print("\n[4] Solving permutation...")
    from src.signal_processing import pad_signals
    sources_padded = pad_signals(sources)
    aligned_sources, permutation, correlations = permutation_solver(
        sources_padded, separated
    )
    print(f"  ✓ Permutation: {permutation}")
    print(f"  ✓ Correlation matrix:")
    print(correlations.round(3))
    
    # 5. Evaluate metrics
    print("\n[5] Computing evaluation metrics...")
    snr_values = []
    sdr_values = []
    
    for i in range(len(sources_padded)):
        snr_val = snr(sources_padded[i], aligned_sources[i])
        sdr_val = sdr(sources_padded[i], aligned_sources[i])
        snr_values.append(snr_val)
        sdr_values.append(sdr_val)
        print(f"  Source {i} ({labels[i]}): SNR = {snr_val:.2f} dB, SDR = {sdr_val:.2f} dB")
    
    avg_snr = np.mean(snr_values)
    avg_sdr = np.mean(sdr_values)
    print(f"\n  Average SNR: {avg_snr:.2f} dB")
    print(f"  Average SDR: {avg_sdr:.2f} dB")
    
    # 6. Test MFCC extraction
    print("\n[6] Testing MFCC extraction...")
    mfcc_features = mfcc(sources[0], sample_rates[0])
    print(f"  ✓ MFCC shape: {mfcc_features.shape}")
    print(f"  ✓ n_mfcc = {mfcc_features.shape[0]}, n_frames = {mfcc_features.shape[1]}")
    
    # 7. Test DTW recognition
    print("\n[7] Testing DTW recognition...")
    templates_mfcc = []
    templates_labels = []
    
    # Load templates for digits 0-9
    for i in range(10):
        filepath = os.path.join(dataset_dir, f"digit_{i}.wav")
        if os.path.exists(filepath):
            data, sr = load_wav(filepath)
            mfcc_feat = mfcc(data, sr)
            templates_mfcc.append(mfcc_feat.T)
            templates_labels.append(str(i))
    
    classifier = DTWClassifier()
    classifier.fit(templates_mfcc, templates_labels)
    print(f"  ✓ Loaded {len(templates_labels)} templates")
    
    # Recognize separated sources
    print("\n  Recognition results:")
    for i, source in enumerate(aligned_sources):
        mfcc_feat = mfcc(source, sample_rates[0])
        predicted_label, distance = classifier.predict_single(mfcc_feat.T)
        actual_label = labels[i]
        match = "✓" if predicted_label == actual_label else "✗"
        print(f"    {match} Source {i}: Predicted = {predicted_label}, "
              f"Actual = {actual_label}, Distance = {distance:.2f}")
    
    # 8. Save outputs (optional)
    print("\n[8] Saving outputs...")
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save mixtures
    for i, mixture in enumerate(mixtures):
        save_wav(os.path.join(output_dir, f"mixture_{i}.wav"), mixture, sample_rates[0])
    
    # Save separated sources
    for i, source in enumerate(aligned_sources):
        save_wav(os.path.join(output_dir, f"separated_{i}.wav"), source, sample_rates[0])
    
    print(f"  ✓ Saved mixtures and separated sources to '{output_dir}/' directory")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_pipeline()
