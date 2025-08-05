#!/usr/bin/env python
# scripts/optimize_batch_size.py
"""
Script to test different batch sizes and find the optimal one based on time per step.
"""

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import copy
from datetime import datetime

# Enable TensorFloat32 (TF32) for better performance on compatible GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config, build_model_from_config, get_device
from src.data_processor import DataProcessor
from src.trainer import Trainer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Find optimal batch size for training')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs to run for each batch size')
    parser.add_argument('--start-batch', type=int, default=64,
                        help='Starting batch size')
    parser.add_argument('--end-batch', type=int, default=1024,
                        help='Ending batch size')
    parser.add_argument('--batch-multiplier', type=float, default=2.0,
                        help='Multiplier between batch sizes (e.g., 2.0 means 64, 128, 256, ...)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='batch_size_benchmark',
                        help='Directory to save benchmark results')
    return parser.parse_args()

def run_benchmark(config, batch_size, epochs, device, data_processor=None):
    """
    Run benchmark for a specific batch size.
    
    Args:
        config: Configuration dictionary
        batch_size: Batch size to test
        epochs: Number of epochs to run
        device: Device to run on
        data_processor: Optional DataProcessor to reuse
        
    Returns:
        dict: Dictionary with timing results
    """
    print(f"\n{'-'*60}")
    print(f"Benchmarking batch size: {batch_size}")
    print(f"{'-'*60}")
    
    # Create a copy of the config to modify
    config_copy = copy.deepcopy(config)
    
    # Update the batch size and total steps
    config_copy["training"]["batch_size"] = batch_size
    config_copy["training"]["total_steps"] = epochs
    
    # Prepare data if not provided
    if data_processor is None:
        data_processor = DataProcessor(config_copy)
        x_train, z_train, x_test, z_test, t_vals = data_processor.prepare_data()
    else:
        x_train, z_train, x_test, z_test, t_vals = data_processor.get_data()
    
    # Build model
    model = build_model_from_config(config_copy, device)
    
    # Initialize trainer
    trainer = Trainer(model, config_copy, device)
    
    # Define eval interval to not interfere with timing
    eval_interval = epochs + 1  # No eval during training
    
    # Start training
    start_time = time.time()
    stats = trainer.train(
        x_train, z_train, x_test, z_test, t_vals,
        eval_interval=eval_interval
    )
    total_time = time.time() - start_time
    
    # Calculate metrics
    step_times = [t * 1000 for t in stats['times']]  # Convert to ms
    
    # Skip first 10 steps (warmup)
    if len(step_times) > 10:
        measurement_times = step_times[10:]
    else:
        measurement_times = step_times
    
    results = {
        'batch_size': batch_size,
        'step_times': step_times,
        'mean_time': np.mean(measurement_times),
        'std_time': np.std(measurement_times),
        'min_time': np.min(measurement_times),
        'max_time': np.max(measurement_times),
        'examples_per_sec': batch_size * 1000 / np.mean(measurement_times),
        'total_time': total_time
    }
    
    print(f"Results for batch size {batch_size}:")
    print(f"  Mean step time: {results['mean_time']:.2f} ms")
    print(f"  Examples/sec: {results['examples_per_sec']:.2f}")
    print(f"  Total time: {results['total_time']:.2f} sec")
    
    return results

def plot_results(benchmarks, output_dir):
    """Plot benchmark results."""
    os.makedirs(output_dir, exist_ok=True)
    
    batch_sizes = [b['batch_size'] for b in benchmarks]
    mean_times = [b['mean_time'] for b in benchmarks]
    examples_per_sec = [b['examples_per_sec'] for b in benchmarks]
    
    # Plot 1: Batch Size vs. Step Time
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, mean_times, 'o-', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Step Time (ms)')
    plt.title('Batch Size vs. Step Time')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'batch_size_vs_time.png'))
    plt.close()
    
    # Plot 2: Batch Size vs. Examples per Second
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, examples_per_sec, 'o-', linewidth=2)
    plt.xlabel('Batch Size')
    plt.ylabel('Examples per Second')
    plt.title('Batch Size vs. Throughput')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'batch_size_vs_throughput.png'))
    plt.close()
    
    # Plot 3: Bar Chart of Step Times
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(batch_sizes)), mean_times, tick_label=[str(b) for b in batch_sizes])
    for i, v in enumerate(mean_times):
        plt.text(i, v + 5, f"{v:.1f}", ha='center')
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Step Time (ms)')
    plt.title('Step Time by Batch Size')
    plt.grid(True, axis='y')
    plt.savefig(os.path.join(output_dir, 'step_time_by_batch.png'))
    plt.close()
    
    # Save the results as a CSV file
    with open(os.path.join(output_dir, 'benchmark_results.csv'), 'w') as f:
        f.write('batch_size,mean_time_ms,examples_per_sec\n')
        for b, t, e in zip(batch_sizes, mean_times, examples_per_sec):
            f.write(f'{b},{t:.2f},{e:.2f}\n')
            
    # Find the optimal batch size: highest throughput
    optimal_idx = np.argmax(examples_per_sec)
    optimal_batch = batch_sizes[optimal_idx]
    
    # Create a summary report
    with open(os.path.join(output_dir, 'benchmark_summary.txt'), 'w') as f:
        f.write('Batch Size Optimization Summary\n')
        f.write(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write(f'Optimal batch size: {optimal_batch}\n')
        f.write(f'Fastest examples per second: {examples_per_sec[optimal_idx]:.2f}\n')
        f.write(f'Average step time: {mean_times[optimal_idx]:.2f} ms\n\n')
        
        f.write('All Results:\n')
        f.write('-' * 60 + '\n')
        f.write(f'{"Batch Size":^15} | {"Step Time (ms)":^15} | {"Examples/sec":^15}\n')
        f.write('-' * 60 + '\n')
        
        for b, t, e in zip(batch_sizes, mean_times, examples_per_sec):
            f.write(f'{b:^15} | {t:^15.2f} | {e:^15.2f}\n')
    
    # Print the optimal batch size
    print(f"\n{'-'*60}")
    print(f"Benchmark complete!")
    print(f"Optimal batch size: {optimal_batch}")
    print(f"Results saved to {output_dir}")
    print(f"{'-'*60}")
    
    return optimal_batch

def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device-specific seeds if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize data processor once to reuse across benchmarks
    print("Loading and preparing data...")
    data_processor = DataProcessor(config)
    data_processor.prepare_data()
    
    # Generate batch sizes to test
    batch_sizes = []
    current_batch = args.start_batch
    while current_batch <= args.end_batch:
        batch_sizes.append(int(current_batch))
        current_batch *= args.batch_multiplier
    
    print(f"Benchmarking batch sizes: {batch_sizes}")
    print(f"Running {args.epochs} epochs for each batch size")
    
    # Run benchmarks
    benchmarks = []
    for batch_size in batch_sizes:
        result = run_benchmark(config, batch_size, args.epochs, device, data_processor)
        benchmarks.append(result)
    
    # Plot results and find optimal batch size
    optimal_batch = plot_results(benchmarks, args.output_dir)
    
    # Save optimal batch size to config
    opt_config = copy.deepcopy(config)
    opt_config["training"]["batch_size"] = optimal_batch
    with open(os.path.join(args.output_dir, 'optimal_config.yaml'), 'w') as f:
        yaml.dump(opt_config, f, default_flow_style=False)
    
    print(f"\nTo use the optimal batch size, run:")
    print(f"python scripts/train_model.py --config {args.output_dir}/optimal_config.yaml")

if __name__ == "__main__":
    main()