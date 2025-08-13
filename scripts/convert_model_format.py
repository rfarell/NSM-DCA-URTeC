#!/usr/bin/env python
"""
Convert model checkpoints to cross-platform compatible format.
This script ensures models can be loaded on both CUDA (Windows) and CPU (Mac).
"""

import os
import sys
import argparse
import torch

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

def convert_checkpoint(input_path, output_path=None, verbose=True):
    """
    Convert a model checkpoint to cross-platform compatible format.
    
    Args:
        input_path: Path to the original checkpoint
        output_path: Path to save converted checkpoint (default: adds '_portable' suffix)
        verbose: Print conversion details
    
    Returns:
        Path to the converted checkpoint
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_portable{ext}"
    
    if verbose:
        print(f"Converting: {input_path}")
        print(f"Output: {output_path}")
    
    # Load checkpoint with CPU map_location to ensure compatibility
    checkpoint = torch.load(input_path, map_location='cpu')
    
    # Check if it's already in new format
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        if verbose:
            print("Checkpoint is already in portable format!")
        state_dict = checkpoint['model_state_dict']
    else:
        # Old format - convert it
        if verbose:
            print("Converting from old format...")
        state_dict = checkpoint
    
    # Ensure all tensors are on CPU
    cpu_state_dict = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            cpu_state_dict[k] = v.cpu()
        else:
            cpu_state_dict[k] = v
    
    # Fix compiled model keys if necessary
    if any(k.startswith('_orig_mod.') for k in cpu_state_dict.keys()):
        if verbose:
            print("Removing compiled model prefixes...")
        cpu_state_dict = {k.replace('_orig_mod.', ''): v 
                         for k, v in cpu_state_dict.items()}
    
    # Create new checkpoint with metadata
    new_checkpoint = {
        'model_state_dict': cpu_state_dict,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'converted': True
    }
    
    # Save the converted checkpoint
    torch.save(new_checkpoint, output_path)
    
    if verbose:
        print(f"✓ Conversion complete!")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"  State dict keys: {len(cpu_state_dict)}")
        
        # Test loading
        try:
            test_load = torch.load(output_path, map_location='cpu')
            print("✓ Verification: Checkpoint loads successfully on CPU")
        except Exception as e:
            print(f"✗ Verification failed: {e}")
    
    return output_path


def convert_experiment_models(experiment_dir, recursive=True):
    """
    Convert all model.pth files in an experiment directory.
    
    Args:
        experiment_dir: Path to experiments directory
        recursive: Process subdirectories recursively
    """
    converted_count = 0
    
    if recursive:
        # Find all model.pth files
        for root, dirs, files in os.walk(experiment_dir):
            for file in files:
                if file == 'model.pth':
                    model_path = os.path.join(root, file)
                    backup_path = os.path.join(root, 'model_original.pth')
                    
                    # Check if already converted
                    try:
                        checkpoint = torch.load(model_path, map_location='cpu')
                        if isinstance(checkpoint, dict) and checkpoint.get('converted'):
                            print(f"✓ Already converted: {model_path}")
                            continue
                    except:
                        pass
                    
                    # Backup original
                    if not os.path.exists(backup_path):
                        print(f"\nBacking up: {model_path} -> {backup_path}")
                        import shutil
                        shutil.copy2(model_path, backup_path)
                    
                    # Convert in place
                    print(f"\nConverting: {model_path}")
                    convert_checkpoint(backup_path, model_path, verbose=False)
                    converted_count += 1
    
    print(f"\n{'='*60}")
    print(f"Converted {converted_count} model(s)")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Convert model checkpoints to portable format')
    parser.add_argument('input', help='Input checkpoint path or experiment directory')
    parser.add_argument('-o', '--output', help='Output checkpoint path (for single file)')
    parser.add_argument('-r', '--recursive', action='store_true',
                      help='Convert all models in directory recursively')
    parser.add_argument('-q', '--quiet', action='store_true',
                      help='Suppress verbose output')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.input):
        # Directory mode
        if args.recursive or args.input.endswith('experiments'):
            print(f"Converting all models in: {args.input}")
            convert_experiment_models(args.input, recursive=True)
        else:
            print(f"Error: {args.input} is a directory. Use -r flag for recursive conversion.")
            sys.exit(1)
    else:
        # Single file mode
        convert_checkpoint(args.input, args.output, verbose=not args.quiet)


if __name__ == "__main__":
    main()