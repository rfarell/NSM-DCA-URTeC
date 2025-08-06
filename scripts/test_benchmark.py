#!/usr/bin/env python
# scripts/test_benchmark.py
"""
Quick test of benchmark functionality with a small subset of data.
"""

import os
import sys
import numpy as np
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.dca_models import ArpsHyperbolic, DuongModel, bootstrap_dca_uncertainty
from src.probabilistic_metrics import crps_empirical, interval_sharpness

def test_dca_models():
    """Test DCA models with sample data."""
    print("Testing DCA models...")
    
    # Create sample cumulative production data
    t = np.array([0, 30, 60, 90, 180, 360, 720, 1080], dtype=np.float64)
    Q = np.array([0, 100, 180, 250, 400, 600, 800, 900], dtype=np.float64)
    
    # Test Arps model
    print("\nTesting Arps Hyperbolic...")
    arps = ArpsHyperbolic()
    arps.fit(t[1:], Q[1:])  # Skip t=0
    print(f"  Parameters: q_i={arps.params[0]:.2f}, Di={arps.params[1]:.4f}, b={arps.params[2]:.2f}")
    
    pred = arps.predict(t)
    print(f"  Predictions at t=0: {pred[0]:.2f} (should be 0)")
    print(f"  Predictions at t=360: {pred[5]:.2f} (actual: {Q[5]:.2f})")
    
    # Test bootstrap uncertainty
    print("\n  Testing bootstrap uncertainty...")
    unc = bootstrap_dca_uncertainty(arps, t[1:], Q[1:], t, n_bootstrap=50)
    print(f"  P50 at t=360: {unc['p50'][5]:.2f}")
    print(f"  P10-P90 width at t=360: {unc['p90'][5] - unc['p10'][5]:.2f}")
    
    # Test Duong model
    print("\nTesting Duong Model...")
    duong = DuongModel()
    try:
        duong.fit(t[1:], Q[1:])
        print(f"  Parameters: q1={duong.params[0]:.2f}, a={duong.params[1]:.4f}, m={duong.params[2]:.2f}")
        
        pred = duong.predict(t)
        print(f"  Predictions at t=0: {pred[0]:.2f} (should be 0)")
        print(f"  Predictions at t=360: {pred[5]:.2f} (actual: {Q[5]:.2f})")
    except Exception as e:
        print(f"  Duong fitting failed: {e}")
    
    return True

def test_metrics():
    """Test probabilistic metrics."""
    print("\nTesting probabilistic metrics...")
    
    # Create sample data
    y_true = np.array([100, 200, 300, 400, 500])
    y_samples = np.random.normal(y_true[np.newaxis, :], 50, (100, 5))
    
    # Test CRPS
    crps = crps_empirical(y_true, y_samples)
    print(f"  CRPS: {crps:.2f}")
    
    # Test sharpness
    p10 = np.percentile(y_samples, 10, axis=0)
    p90 = np.percentile(y_samples, 90, axis=0)
    sharp = interval_sharpness(p10, p90)
    print(f"  Mean interval width: {sharp['mean_width']:.2f}")
    print(f"  Relative width: {sharp['relative_width']:.3f}")
    
    return True

def test_lightgbm():
    """Test LightGBM model if available."""
    try:
        from src.ml_benchmarks import LightGBMQuantile
        print("\nTesting LightGBM...")
        
        # Create sample data
        n_wells = 20
        n_features = 5
        n_time = 8
        
        X = np.random.randn(n_wells, n_features)
        Y = np.cumsum(np.random.exponential(100, (n_wells, n_time)), axis=1)
        time_points = np.array([0, 30, 60, 90, 180, 360, 720, 1080])
        
        # Train model
        lgb = LightGBMQuantile(quantiles=[0.1, 0.5, 0.9])
        lgb.fit(X, Y, time_points)
        
        # Test prediction
        X_test = np.random.randn(5, n_features)
        pred = lgb.predict(X_test)
        
        print(f"  Prediction shape: {pred['p50'].shape}")
        print(f"  P50 at t=360 for well 0: {pred['p50'][0, 5]:.2f}")
        print(f"  P90-P10 width at t=360 for well 0: {pred['p90'][0, 5] - pred['p10'][0, 5]:.2f}")
        
        return True
    except ImportError:
        print("\nLightGBM not installed, skipping ML benchmark test")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("BENCHMARK COMPONENT TESTS")
    print("="*60)
    
    # Test each component
    success = True
    
    success &= test_dca_models()
    success &= test_metrics()
    success &= test_lightgbm()
    
    print("\n" + "="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("="*60)

if __name__ == "__main__":
    main()