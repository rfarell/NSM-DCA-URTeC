#!/usr/bin/env python
# scripts/fix_gpu_training.py
"""
Script to diagnose and fix GPU training issues.
Run this before training to set optimal GPU settings.
"""

import torch
import os
import sys

def diagnose_gpu():
    """Diagnose GPU setup and provide recommendations."""
    
    print("=" * 60)
    print("GPU DIAGNOSTIC TOOL")
    print("=" * 60)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("❌ CUDA is not available!")
        print("Running on CPU only.")
        return False
    
    print("✓ CUDA is available")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    
    # Get GPU info
    gpu_count = torch.cuda.device_count()
    print(f"\nNumber of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        
        # Check current memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"  Currently allocated: {allocated:.2f} GB")
            print(f"  Currently reserved: {reserved:.2f} GB")
    
    return True

def set_optimal_settings():
    """Set optimal settings for GPU training."""
    
    print("\n" + "=" * 60)
    print("APPLYING OPTIMAL SETTINGS")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("No GPU available, skipping GPU optimizations")
        return
    
    # 1. Disable torch.compile for faster startup
    os.environ['TORCH_COMPILE_DISABLE'] = '1'
    print("✓ Disabled torch.compile (prevents slow first epoch)")
    
    # 2. Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print("✓ Set expandable memory segments (reduces fragmentation)")
    
    # 3. Enable TF32 for Ampere GPUs (30xx, 40xx, A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✓ Enabled TF32 (faster on Ampere GPUs)")
    
    # 4. Enable cudnn autotuner
    torch.backends.cudnn.benchmark = True
    print("✓ Enabled cuDNN autotuner")
    
    # 5. Set number of threads for data loading
    torch.set_num_threads(4)
    print("✓ Set 4 CPU threads for data loading")
    
    print("\nRecommended command to run training:")
    print("-" * 40)
    
    # Check GPU memory to recommend batch size
    props = torch.cuda.get_device_properties(0)
    mem_gb = props.total_memory / 1e9
    
    if mem_gb < 8:
        batch_size = 4096
        print(f"For {mem_gb:.1f}GB GPU, use smaller batch size:")
    elif mem_gb < 12:
        batch_size = 8192
        print(f"For {mem_gb:.1f}GB GPU, use medium batch size:")
    else:
        batch_size = 16384
        print(f"For {mem_gb:.1f}GB GPU, can use large batch size:")
    
    print(f"\nEXPORT TORCH_COMPILE_DISABLE=1")
    print(f"python scripts/train_model.py --device cuda --experiment exp_gpu")
    print(f"\nOr edit config.yaml:")
    print(f"  training:")
    print(f"    batch_size: {batch_size}")
    print(f"    compile_model: false")

def test_gpu_speed():
    """Quick benchmark of GPU speed."""
    
    if not torch.cuda.is_available():
        print("No GPU available for speed test")
        return
    
    print("\n" + "=" * 60)
    print("GPU SPEED TEST")
    print("=" * 60)
    
    device = torch.device('cuda')
    
    # Test matrix multiplication speed
    size = 4096
    print(f"Testing {size}x{size} matrix multiplication...")
    
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(3):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # Time the operation
    import time
    start = time.time()
    for _ in range(10):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Time for 10 iterations: {elapsed:.3f} seconds")
    print(f"Average time per matmul: {elapsed/10*1000:.1f} ms")
    
    # Calculate TFLOPS
    flops = 2 * size**3 * 10  # 2*n^3 for matmul, 10 iterations
    tflops = flops / elapsed / 1e12
    print(f"Performance: {tflops:.1f} TFLOPS")
    
    # Reference speeds (approximate)
    if tflops > 100:
        print("✓ Excellent performance (A100/H100 level)")
    elif tflops > 50:
        print("✓ Great performance (RTX 4090/3090 level)")
    elif tflops > 20:
        print("✓ Good performance (RTX 3080/3070 level)")
    elif tflops > 10:
        print("✓ Decent performance (RTX 3060 level)")
    else:
        print("⚠ Low performance - check drivers/cooling")
    
    # Clean up
    del a, b, c
    torch.cuda.empty_cache()

def main():
    """Main diagnostic routine."""
    
    # Run diagnostics
    has_gpu = diagnose_gpu()
    
    if has_gpu:
        # Set optimal settings
        set_optimal_settings()
        
        # Run speed test
        test_gpu_speed()
    else:
        print("\n" + "=" * 60)
        print("CPU-ONLY MODE")
        print("=" * 60)
        print("The model will train on CPU.")
        print("This is normal for Mac systems.")
        print("\nRecommended settings for CPU training:")
        print("  training:")
        print("    batch_size: 2048  # Smaller batch for CPU")
        print("    compile_model: false")
        print("    total_steps: 100  # Start with fewer steps to test")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()