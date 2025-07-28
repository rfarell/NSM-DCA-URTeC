#!/usr/bin/env python
# scripts/model_info.py

import os
import sys
import torch
import argparse

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.utils import load_config, build_model_from_config, print_parameter_info, print_module_param_counts, print_parameter_types

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Display MixtureGP model information')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint (optional)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cpu or cuda)')
    return parser.parse_args()

def main():
    """Main script to display model information."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        from src.utils import get_device
        device = get_device()
    print(f"Using device: {device}")
    
    # Build model
    model = build_model_from_config(config, device)
    
    # Load model weights if provided
    if args.model:
        # Load model state dict
        state_dict = torch.load(args.model, map_location=device)
        
        # Check if we need to fix the state dict (compiled model issue)
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            # Remove the '_orig_mod.' prefix from keys
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_state_dict[k[10:]] = v  # Remove the '_orig_mod.' prefix
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        print(f"Model loaded from {args.model}")
    
    # Print configuration
    print("\n=== Model Configuration ===")
    print(f"Input dimension:  {config['model']['input_dim']}")
    print(f"Z dimension:      {config['model']['z_dim']}")
    print(f"Number of basis:  {config['model']['num_basis']}")
    print(f"Number of experts: {config['model']['num_experts']}")
    print(f"K (num samples):  {config['model']['K']}")
    print(f"MC samples:       {config['model']['num_mc']}")
    
    # Print parameter information
    print_parameter_info(model)
    print_module_param_counts(model)
    print_parameter_types(model)
    
    # Print detailed expert information
    print("\n=== Expert Information ===")
    for i, expert in enumerate(model.experts):
        print(f"\nExpert {i}:")
        
        # Print noise covariance matrix
        noise_cov = expert.get_noise_cov().detach().cpu().numpy()
        print("  Noise covariance matrix:")
        for row in noise_cov:
            print(f"    [{', '.join([f'{x:.4f}' for x in row])}]")
        
        # Print raw noise parameters
        print("  Raw noise parameters (unconstrained):")
        noise_params = expert.noise_unconstrained.detach().cpu().numpy()
        print(f"    {noise_params}")
        
        # Print length scales
        print("  Length scales (lambda):")
        length_scales = expert.lam.detach().cpu().numpy()
        print(f"    {length_scales}")
        
        # Print global mean vector (b) statistics
        b = expert.global_b.detach().cpu().numpy()
        print("  Global b vector statistics:")
        print(f"    Mean: {b.mean():.6f}, Std: {b.std():.6f}")
        print(f"    Min: {b.min():.6f}, Max: {b.max():.6f}")
        
        # Print global L matrix statistics
        L_flat = expert.global_L_flat.detach().cpu().numpy()
        print("  Global L_flat statistics:")
        print(f"    Mean: {L_flat.mean():.6f}, Std: {L_flat.std():.6f}")
        print(f"    Min: {L_flat.min():.6f}, Max: {L_flat.max():.6f}")
        
        # Print random frequencies statistics (epsilon buffer)
        epsilon = expert.epsilon.detach().cpu().numpy()
        print("  Random frequencies (epsilon) statistics:")
        print(f"    Mean: {epsilon.mean():.6f}, Std: {epsilon.std():.6f}")
        print(f"    Min: {epsilon.min():.6f}, Max: {epsilon.max():.6f}")
        
        # Compute KL divergence for this expert
        kl = expert.KL_divergence().item()
        print(f"  KL divergence: {kl:.6f}")

    # Print gating network information
    print("\n=== Gating Network Parameters ===")
    # Check if the model has the expected structure
    if hasattr(model, 'gating_net') and isinstance(model.gating_net, torch.nn.Sequential):
        for i, layer in enumerate(model.gating_net):
            if isinstance(layer, torch.nn.Linear):
                print(f"Layer {i} (Linear):")
                # Print weight statistics
                weight = layer.weight.detach().cpu().numpy()
                print(f"  Weight shape: {weight.shape}")
                print(f"  Weight stats - Mean: {weight.mean():.6f}, Std: {weight.std():.6f}")
                # Print bias if exists
                if layer.bias is not None:
                    bias = layer.bias.detach().cpu().numpy()
                    print(f"  Bias shape: {bias.shape}")
                    print(f"  Bias stats - Mean: {bias.mean():.6f}, Std: {bias.std():.6f}")

if __name__ == "__main__":
    main()