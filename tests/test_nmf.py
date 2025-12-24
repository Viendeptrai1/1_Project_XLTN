"""
Unit Tests cho NMF
"""

import numpy as np
from src.nmf import NMF


def test_nmf_basic():
    """Test basic NMF functionality"""
    print("Testing NMF basic...")
    
    # Create simple non-negative matrix
    np.random.seed(42)
    n, m, k = 10, 20, 3
    
    W_true = np.abs(np.random.randn(n, k))
    H_true = np.abs(np.random.randn(k, m))
    V = W_true @ H_true + 0.01 * np.abs(np.random.randn(n, m))
    
    # Run NMF
    nmf = NMF(n_components=k, max_iter=200, random_state=42)
    W, H = nmf.fit_transform(V)
    
    # Check shapes
    assert W.shape == (n, k), f"W shape mismatch: {W.shape}"
    assert H.shape == (k, m), f"H shape mismatch: {H.shape}"
    
    # Check non-negativity
    assert np.all(W >= 0), "W should be non-negative"
    assert np.all(H >= 0), "H should be non-negative"
    
    # Check reconstruction
    V_recon = W @ H
    error = np.linalg.norm(V - V_recon)
    
    print(f"  ✓ Shapes correct: W={W.shape}, H={H.shape}")
    print(f"  ✓ Non-negative: min(W)={W.min():.4f}, min(H)={H.min():.4f}")
    print(f"  ✓ Reconstruction error: {error:.4f}")


def test_nmf_convergence():
    """Test convergence behavior"""
    print("\nTesting NMF convergence...")
    
    np.random.seed(42)
    V = np.abs(np.random.randn(20, 30))
    
    nmf = NMF(n_components=5, max_iter=300, tol=1e-4, random_state=42)
    nmf.fit(V)
    
    # Check convergence
    assert nmf.n_iter < 300, f"Should converge, got {nmf.n_iter} iterations"
    
    # Check error decreases
    errors = nmf.reconstruction_error
    assert len(errors) > 0, "Should track reconstruction error"
    assert errors[-1] < errors[0], "Error should decrease"
    
    print(f"  ✓ Converged in {nmf.n_iter} iterations")
    print(f"  ✓ Error decreased: {errors[0]:.4f} → {errors[-1]:.4f}")


def test_nmf_init_methods():
    """Test different initialization methods"""
    print("\nTesting NMF initialization methods...")
    
    np.random.seed(42)
    V = np.abs(np.random.randn(15, 25))
    
    # Random init
    nmf_random = NMF(n_components=5, init='random', max_iter=100, random_state=42)
    W_rand, H_rand = nmf_random.fit_transform(V)
    
    # NNDSVD init
    nmf_svd = NMF(n_components=5, init='nndsvd', max_iter=100, random_state=42)
    W_svd, H_svd = nmf_svd.fit_transform(V)
    
    # Both should work
    assert W_rand.shape == W_svd.shape
    assert H_rand.shape == H_svd.shape
    
    print(f"  ✓ Random init: {nmf_random.n_iter} iterations")
    print(f"  ✓ NNDSVD init: {nmf_svd.n_iter} iterations")


def test_nmf_transform():
    """Test transform on new data"""
    print("\nTesting NMF transform...")
    
    np.random.seed(42)
    V_train = np.abs(np.random.randn(20, 30))
    V_test = np.abs(np.random.randn(20, 40))
    
    # Fit on training data
    nmf = NMF(n_components=5, max_iter=100, random_state=42)
    nmf.fit(V_train)
    
    # Transform test data
    H_test = nmf.transform(V_test)
    
    # Check shape
    assert H_test.shape == (5, 40), f"H_test shape mismatch: {H_test.shape}"
    assert np.all(H_test >= 0), "H_test should be non-negative"
    
    print(f"  ✓ Transform works: H_test shape = {H_test.shape}")


def test_nmf_determinism():
    """Test that same seed gives same result"""
    print("\nTesting NMF determinism...")
    
    np.random.seed(42)
    V = np.abs(np.random.randn(15, 25))
    
    nmf1 = NMF(n_components=5, max_iter=100, random_state=42)
    W1, H1 = nmf1.fit_transform(V)
    
    nmf2 = NMF(n_components=5, max_iter=100, random_state=42)
    W2, H2 = nmf2.fit_transform(V)
    
    diff_W = np.linalg.norm(W1 - W2)
    diff_H = np.linalg.norm(H1 - H2)
    
    assert diff_W < 1e-10, f"W should be deterministic, got diff={diff_W}"
    assert diff_H < 1e-10, f"H should be deterministic, got diff={diff_H}"
    
    print(f"  ✓ Deterministic: diff_W={diff_W:.2e}, diff_H={diff_H:.2e}")


def run_all_tests():
    """Run all NMF tests"""
    print("=" * 60)
    print("NMF Unit Tests")
    print("=" * 60)
    
    test_nmf_basic()
    test_nmf_convergence()
    test_nmf_init_methods()
    test_nmf_transform()
    test_nmf_determinism()
    
    print("\n" + "=" * 60)
    print("✓ All NMF tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
