# GPU Optimization Guide

## Overview
This guide helps optimize GPU utilization for training the NSM-DCA model, especially on Windows laptops with dual GPUs (Intel + NVIDIA).

## Quick Start

### 1. Check Your GPU Setup
```bash
python scripts/check_gpu.py
```

### 2. Force NVIDIA GPU Usage
On Windows with dual GPUs, force the NVIDIA GPU:
```bash
# Windows Command Prompt
set CUDA_VISIBLE_DEVICES=1  # Use GPU 1 (typically NVIDIA)
python scripts/train_model.py --experiment my_exp

# Windows PowerShell
$env:CUDA_VISIBLE_DEVICES="1"
python scripts/train_model.py --experiment my_exp
```

### 3. Verify GPU Usage During Training
Monitor GPU usage with:
```bash
# Windows
nvidia-smi -l 1  # Updates every 1 second
```

## Optimizations Implemented

### 1. **Smart GPU Selection**
- Automatically selects GPU with most memory (usually NVIDIA over Intel)
- Respects `CUDA_VISIBLE_DEVICES` environment variable
- Shows which GPU is selected at startup

### 2. **Data Pipeline Optimization**
- Keeps training data on GPU throughout training (no CPU-GPU transfers)
- Uses custom DataLoader for GPU-resident data
- Eliminates per-batch data movement overhead

### 3. **Compilation Mode Selection**
- Uses `torch.compile` with appropriate mode based on GPU memory:
  - GPUs with >8GB: `max-autotune` mode for best performance
  - GPUs with <8GB: `default` mode for stability
- Can be disabled via config: `compile_model: false`

### 4. **Mixed Precision Training**
- Automatically uses mixed precision (FP16) on compatible GPUs
- Reduces memory usage and increases speed
- Requires compute capability >= 7.0 (most modern GPUs)

## Performance Tips

### For Windows Laptops with Dual GPUs

1. **Always use the NVIDIA GPU**:
   ```bash
   set CUDA_VISIBLE_DEVICES=1  # Typically NVIDIA is GPU 1
   ```

2. **Close unnecessary applications** to free GPU memory

3. **Adjust batch size** based on GPU memory:
   - 4GB GPU: `batch_size: 2048`
   - 8GB GPU: `batch_size: 4096`
   - 16GB+ GPU: `batch_size: 8192`

### Monitoring Performance

1. **GPU Utilization** should be >90% during training
2. **Memory Usage** should be close to maximum without OOM errors
3. **Training time per step** should be consistent

### Troubleshooting

**Low GPU Utilization (<50%)**:
- Check if using Intel GPU instead of NVIDIA
- Verify data is kept on GPU (no CPU-GPU transfers)
- Increase batch size if memory allows

**Out of Memory Errors**:
- Reduce batch size
- Disable model compilation: `compile_model: false`
- Reduce number of Monte Carlo samples: `num_mc: 2`

**Slow Training on Windows**:
- Ensure Windows GPU scheduling is set to "Hardware-accelerated"
- Update NVIDIA drivers to latest version
- Disable Windows GPU timeout (TDR) for long training runs

## Configuration Options

In `config.yaml`:
```yaml
training:
  batch_size: 8192  # Adjust based on GPU memory
  compile_model: true  # Set to false if having issues
  
model:
  num_mc: 4  # Reduce if running out of memory
```

## Environment Variables

- `CUDA_VISIBLE_DEVICES`: Select which GPU(s) to use (0, 1, etc.)
- `CUDA_DEVICE_ORDER`: Set to "PCI_BUS_ID" for consistent ordering
- `CUDA_LAUNCH_BLOCKING`: Set to 1 for debugging (slower)

## Validation

After implementing optimizations, you should see:
1. GPU selection message showing NVIDIA GPU
2. GPU utilization >90% during training
3. Consistent step times (no sudden slowdowns)
4. Memory usage near maximum without errors

Run `python scripts/check_gpu.py` to verify your setup!