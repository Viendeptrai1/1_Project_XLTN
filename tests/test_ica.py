"""
Unit Tests cho FastICA
"""

import numpy as np
from src.ica import FastICA, kurtosis, negentropy
from src.signal_processing import generate_mixing_matrix


def test_kurtosis():
    """Test kurtosis calculation"""
    print("Testing kurtosis...")
    
    # Gaussian signal should have kurtosis near 0
    gaussian = np.random.randn(1000)
    kurt = kurtosis(gaussian)
    assert abs(kurt) < 1.0, f"Gaussian kurtosis should be ~0, got {kurt}"
    
    # Uniform signal should have negative kurtosis
    uniform = np.random.uniform(-1, 1, 1000)
    kurt = kurtosis(uniform)
    assert kurt < 0, f"Uniform kurtosis should be negative, got {kurt}"
    
    print("  ✓ Kurtosis test passed")


def test_ica_basic():
    """Test basic ICA functionality"""
    print("\nTesting FastICA basic...")
    
    # Create simple synthetic data
    np.random.seed(42)
    n_samples = 1000
    time = np.linspace(0, 8, n_samples)
    
    # Two independent sources
    s1 = np.sin(2 * time)  # Signal 1
    s2 = np.sign(np.sin(3 * time))  # Signal 2 (square wave)
    S = np.c_[s1, s2].T
    
    # Mix them
    A = np.array([[1, 1], [0.5, 2]])
    X = np.dot(A, S)
    
    # Run ICA
    ica = FastICA(n_components=2, max_iter=200, random_state=42)
    S_hat = ica.fit_transform(X)
    
    # Check shape
    assert S_hat.shape == S.shape, f"Shape mismatch: {S_hat.shape} vs {S.shape}"
    
    # Check convergence
    assert ica.n_iter < 200, f"Should converge before max_iter, got {ica.n_iter}"
    
    print(f"  ✓ ICA converged in {ica.n_iter} iterations")
    print(f"  ✓ Output shape: {S_hat.shape}")


def test_ica_whitening():
    """Test whitening step"""
    print("\nTesting whitening...")
    
    # Create correlated data
    np.random.seed(42)
    X = np.random.randn(3, 1000)
    X[1] = X[0] * 0.5 + np.random.randn(1000) * 0.1
    
    ica = FastICA(n_components=3, random_state=42)
    ica.fit(X)
    
    # Check whitening matrix exists
    assert ica.whitening_matrix is not None
    assert ica.dewhitening_matrix is not None
    
    # Apply whitening
    from src.signal_processing import whitening
    X_white, _, _, _ = whitening(X)
    
    # Check decorrelation (covariance should be identity)
    cov = np.cov(X_white)
    identity = np.eye(3)
    
    diff = np.linalg.norm(cov - identity)
    assert diff < 0.5, f"Whitened data should be decorrelated, got diff={diff}"
    
    print(f"  ✓ Whitening produces near-identity covariance (diff={diff:.4f})")


def test_ica_determinism():
    """Test that same seed gives same result"""
    print("\nTesting determinism...")
    
    np.random.seed(42)
    X = np.random.randn(3, 1000)
    
    ica1 = FastICA(n_components=3, random_state=42)
    S1 = ica1.fit_transform(X)
    
    ica2 = FastICA(n_components=3, random_state=42)
    S2 = ica2.fit_transform(X)
    
    diff = np.linalg.norm(S1 - S2)
    assert diff < 1e-10, f"Same seed should give same result, got diff={diff}"
    
    print(f"  ✓ Deterministic: diff={diff:.2e}")


def run_all_tests():
    """Run all FastICA tests"""
    print("=" * 60)
    print("FastICA Unit Tests")
    print("=" * 60)
    
    test_kurtosis()
    test_ica_basic()
    test_ica_whitening()
    test_ica_determinism()
    
    print("\n" + "=" * 60)
    print("✓ All FastICA tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
