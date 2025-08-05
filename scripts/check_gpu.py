#!/usr/bin/env python
"""
Check GPU availability and configuration for optimal performance.
"""

import torch
import os
import sys

def check_gpu_setup():
    """Check and display GPU configuration."""
    print("=== GPU Configuration Check ===\n")
    
    # Check PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # List all GPUs
        print("\nAvailable GPUs:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            print(f"  Multi-Processors: {props.multi_processor_count}")
            
            # Check current memory usage
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  Memory Allocated: {allocated:.1f} GB")
                print(f"  Memory Reserved: {reserved:.1f} GB")
        
        # Check current device
        current = torch.cuda.current_device()
        print(f"\nCurrent Default GPU: {current}")
        
        # Environment variables
        print("\nRelevant Environment Variables:")
        env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_LAUNCH_BLOCKING']
        for var in env_vars:
            value = os.environ.get(var, 'Not set')
            print(f"  {var}: {value}")
        
        # Performance settings
        print("\nPerformance Settings:")
        print(f"  TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"  Cudnn Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"  Cudnn Deterministic: {torch.backends.cudnn.deterministic}")
        
        # Test tensor operations
        print("\nTesting GPU Operations...")
        try:
            # Small test
            device = torch.device('cuda')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print("  Basic operations: ✓ Success")
            
            # Mixed precision test
            if torch.cuda.get_device_capability()[0] >= 7:
                with torch.cuda.amp.autocast():
                    z = torch.matmul(x, y)
                torch.cuda.synchronize()
                print("  Mixed precision: ✓ Success")
            else:
                print("  Mixed precision: ⚠ Not supported (requires compute capability >= 7.0)")
                
        except Exception as e:
            print(f"  GPU operations test failed: {e}")
    else:
        print("\nNo CUDA GPUs detected!")
        print("Please check:")
        print("  1. NVIDIA GPU drivers are installed")
        print("  2. PyTorch is installed with CUDA support")
        print("  3. Your system has a CUDA-capable GPU")
    
    # Recommendations
    print("\n=== Recommendations ===")
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print("Multiple GPUs detected. To use a specific GPU:")
            print("  - Set CUDA_VISIBLE_DEVICES=0 (or 1, 2, etc.) before running")
            print("  - Or use --device cuda:0 when running scripts")
            
            # Check for Intel GPU
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                if 'Intel' in props.name:
                    print(f"\n⚠ Warning: Intel GPU detected at index {i}")
                    print("  For better performance, use the NVIDIA GPU instead")
    
    print("\nTo force a specific GPU, run:")
    print("  Windows: set CUDA_VISIBLE_DEVICES=0")
    print("  Linux/Mac: export CUDA_VISIBLE_DEVICES=0")

if __name__ == "__main__":
    check_gpu_setup()