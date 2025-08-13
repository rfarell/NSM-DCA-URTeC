# src/utils.py

import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

def load_config(path="config.yaml"):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def evaluate_model(model, x_data, z_data, t_vals, *, scale_tensor=None, num_mc=1, device=torch.device('cpu')):
    """
    Evaluates the model on a dataset and returns metrics.

    Args:
        model (torch.nn.Module): Trained MixtureGP model.
        x_data (torch.Tensor): Production data tensor of shape [N, T, 3] (normalized).
        z_data (torch.Tensor): Static feature tensor of shape [N, z_dim].
        t_vals (torch.Tensor): 1D tensor of time points.
        num_mc (int, optional): Number of Monte Carlo integration samples to average over.
    scale_tensor (torch.Tensor, optional): Per-well scaling factors, shape
        [N, 3].  If provided, both predictions and ground-truth are
        de-normalised before metric/plot calculation, so values are reported
        in original units.
        device (torch.device, optional): The device on which to perform computations.
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    predictions = []
    
    # Load config once (to avoid repeated disk I/O in MC loop)
    config = load_config()

    with torch.no_grad():
        # Use the initial state at the first time point as the starting condition
        x0 = x_data[:, 0, :]  # shape: [N, 3]
        
        # Generate num_mc predictions and average them
        for _ in range(num_mc):
            # Define the ODE function using the model (batch evaluation)
            def ode_func(t_scalar, x_state):
                return model(t_scalar, x_state, z_data)
            
            # Import odeint function here to avoid circular imports
            from torchdiffeq import odeint
            
            # Set up the ODE solver
            ode_options = {
                'method': config["ode"]["method"],
                'atol': float(config["ode"]["atol"]),
                'rtol': float(config["ode"]["rtol"])
            }
            
            # Run the ODE solver
            pred_traj = odeint(ode_func, x0, t_vals, **ode_options)  # [T, N, 3]
            # Rearrange to [N, T, 3] for easier comparison with x_data
            pred_traj = pred_traj.permute(1, 0, 2)
            predictions.append(pred_traj)
        
        # Average the predictions over the MC samples: result shape [N, T, 3]
        pred_mean = torch.stack(predictions, dim=0).mean(dim=0)

        # Optionally de-normalise predictions and actuals for metrics
        if scale_tensor is not None:
            if isinstance(scale_tensor, torch.Tensor):
                scale_torch = scale_tensor.to(pred_mean.device).unsqueeze(1)  # [N,1,3]
                pred_mean = pred_mean * scale_torch
                x_data    = x_data    * scale_torch
            else:
                raise TypeError("scale_tensor must be a torch.Tensor")
        
        # Calculate evaluation metrics
        metrics = {}
        
        # Calculate metrics for all production phases
        phase_names = ['Gas', 'Oil', 'Water']
        
        for phase_idx, phase_name in enumerate(phase_names):
            # Extract predictions and actual values for this phase
            pred_phase = pred_mean[:, :, phase_idx].cpu().numpy()  # [N, T]
            actual_phase = x_data[:, :, phase_idx].cpu().numpy()   # [N, T]
            
            # Calculate last time step metrics (typically most important)
            pred_last = pred_phase[:, -1]           # Last time step predictions
            actual_last = actual_phase[:, -1]       # Last time step actual values
            
            # Calculate R² (handle constant-actual edge-case)
            denom_last = np.sum((actual_last - np.mean(actual_last)) ** 2)
            if denom_last < 1e-12:
                # All actual values identical – R² is undefined.  We return
                # 1.0 if predictions match perfectly, else 0.0 to keep a
                # meaningful scale for training dashboards.
                r2_last = 1.0 if np.allclose(actual_last, pred_last, atol=1e-6) else 0.0
            else:
                r2_last = 1 - np.sum((actual_last - pred_last) ** 2) / denom_last
            
            # Calculate RMSE
            rmse_last = np.sqrt(np.mean((actual_last - pred_last) ** 2))
            
            # Calculate metrics over all time steps
            # R² over all timesteps – constant sequences are rare but handle anyway
            denom_all = np.sum((actual_phase - np.mean(actual_phase)) ** 2)
            if denom_all < 1e-12:
                r2_all = 1.0 if np.allclose(actual_phase, pred_phase, atol=1e-6) else 0.0
            else:
                r2_all = 1 - np.sum((actual_phase - pred_phase) ** 2) / denom_all
            rmse_all = np.sqrt(np.mean((actual_phase - pred_phase) ** 2))
            
            # Store metrics
            metrics[f'{phase_name}_R2_final'] = r2_last
            metrics[f'{phase_name}_RMSE_final'] = rmse_last
            metrics[f'{phase_name}_R2_all'] = r2_all
            metrics[f'{phase_name}_RMSE_all'] = rmse_all
        
        # Calculate overall metrics (across all phases)
        pred_all = pred_mean.cpu().numpy().reshape(-1)    # Flatten to 1D array
        actual_all = x_data.cpu().numpy().reshape(-1)     # Flatten to 1D array
        
        metrics['Overall_R2'] = 1 - np.sum((actual_all - pred_all) ** 2) / max(1e-10, np.sum((actual_all - np.mean(actual_all)) ** 2))
        metrics['Overall_RMSE'] = np.sqrt(np.mean((actual_all - pred_all) ** 2))
        
        return metrics, pred_mean

def plot_pred_vs_actual(model, x_test, z_test, t_vals, *, scale_tensor=None, num_mc=1, device=torch.device('cpu'), 
                        save_path=None, show_plot=False):
    """
    Evaluates the model on the test set and generates a scatter plot comparing the predicted and 
    actual production at the final time point. The plot includes annotations for metrics.

    Args:
        model (torch.nn.Module): Trained MixtureGP model.
        x_test (torch.Tensor): Test production data tensor of shape [N, T, 3] (normalized).
        z_test (torch.Tensor): Test static feature tensor of shape [N, z_dim].
        t_vals (torch.Tensor): 1D tensor of time points.
        num_mc (int, optional): Number of Monte Carlo integration samples to average over.
        device (torch.device, optional): The device on which to perform computations.
        save_path (str, optional): Path to save the plot. If None, uses default path.
        show_plot (bool, optional): Whether to display the plot. Default is False.
        
    Returns:
        tuple: (metrics_dict, predictions)
    """
    # Get evaluation metrics and predictions
    metrics, pred_mean = evaluate_model(model, x_test, z_test, t_vals, scale_tensor=scale_tensor, num_mc=num_mc, device=device)
    
    # Extract the predictions and actual values at the final time point
    # If scale_tensor is given, x_test needs de-normalisation for plot too
    if scale_tensor is not None:
        st = scale_tensor.to(pred_mean.device).unsqueeze(1)  # [N,1,3]
        pred_denorm = pred_mean  # already de-normalised in evaluate_model
        actual_denorm = x_test * st
    else:
        pred_denorm = pred_mean
        actual_denorm = x_test

    pred_final = pred_denorm[:, -1]  # [N, 3]
    actual_final = actual_denorm[:, -1]   # [N, 3]
    
    # Create a 1x3 subplot for all production phases
    phase_names = ['Gas', 'Oil', 'Water']
    phase_colors = ['blue', 'green', 'red']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (phase_name, color) in enumerate(zip(phase_names, phase_colors)):
        ax = axes[i]
        
        # Get data for this phase
        pred_phase = pred_final[:, i].cpu().numpy()
        actual_phase = actual_final[:, i].cpu().numpy()
        
        # Get metrics
        r2 = metrics[f'{phase_name}_R2_final']
        rmse = metrics[f'{phase_name}_RMSE_final']
        
        # Create scatter plot
        ax.scatter(actual_phase, pred_phase, c=color, alpha=0.6, label='Wells')
        
        # Plot the ideal y=x line
        min_val = min(np.min(actual_phase), np.min(pred_phase))
        max_val = max(np.max(actual_phase), np.max(pred_phase))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
        
        # Add labels and title
        if scale_tensor is None:
            unit_lbl = "(normalized)"
        else:
            unit_lbl = "(original units)"

        ax.set_xlabel(f'Actual {phase_name} Production {unit_lbl}')
        ax.set_ylabel(f'Predicted {phase_name} Production {unit_lbl}')
        ax.set_title(f'Predicted vs. Actual {phase_name} Production')
        
        # Add metrics text
        ax.text(0.05 * max_val, 0.9 * max_val, 
                f'$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}',
                fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    # Determine save path if not provided
    if not save_path:
        # Create plots directory if it doesn't exist
        import os
        plots_dir = 'plots'
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, 'pred_vs_actual.png')
    
    # Save plot
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    
    # Show plot if requested (default is False)
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return metrics, pred_mean

def print_metrics(metrics, prefix=""):
    """
    Print evaluation metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary containing evaluation metrics
        prefix (str, optional): Prefix to add to the output (e.g., "Training" or "Test")
    """
    if prefix:
        print(f"\n=== {prefix} Metrics ===")
    else:
        print("\n=== Evaluation Metrics ===")
    
    # Print metrics for each phase
    phase_names = ['Gas', 'Oil', 'Water']
    for phase_name in phase_names:
        print(f"\n{phase_name} Production:")
        r2_f   = metrics[f'{phase_name}_R2_final']
        r2_all = metrics[f'{phase_name}_R2_all']
        # Graceful formatting even if nan (string 'nan')
        r2_f_str   = 'N/A' if r2_f is None or np.isnan(r2_f) else f"{r2_f:.4f}"
        r2_all_str = 'N/A' if r2_all is None or np.isnan(r2_all) else f"{r2_all:.4f}"

        print(f"  Final timestep R²:   {r2_f_str}")
        print(f"  Final timestep RMSE: {metrics[f'{phase_name}_RMSE_final']:.4f}")
        print(f"  All timesteps R²:    {r2_all_str}")
        print(f"  All timesteps RMSE:  {metrics[f'{phase_name}_RMSE_all']:.4f}")
    
    # Print overall metrics
    print("\nOverall (all phases):")
    print(f"  R²:   {metrics['Overall_R2']:.4f}")
    print(f"  RMSE: {metrics['Overall_RMSE']:.4f}")

def get_device():
    """Get the best available device (CUDA or CPU, skip MPS)."""
    import os
    
    # Check for environment variable to force specific GPU
    cuda_device = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        
        # If CUDA_VISIBLE_DEVICES is set, use it
        if cuda_device is not None and cuda_device.isdigit():
            device_id = int(cuda_device)
            if device_id < device_count:
                props = torch.cuda.get_device_properties(device_id)
                print(f"Using GPU {device_id} (from CUDA_VISIBLE_DEVICES): {props.name} with {props.total_memory / 1024**3:.1f}GB memory")
                return torch.device(f"cuda:{device_id}")
        
        # On systems with multiple GPUs, prefer NVIDIA over Intel
        if device_count > 1:
            print(f"Found {device_count} CUDA devices:")
            best_device = 0
            best_memory = 0
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"  GPU {i}: {props.name} with {memory_gb:.1f}GB memory")
                
                # Prefer devices with more memory (likely dedicated GPUs)
                if props.total_memory > best_memory:
                    best_memory = props.total_memory
                    best_device = i
            
            # Use the GPU with most memory
            props = torch.cuda.get_device_properties(best_device)
            print(f"Selected GPU {best_device}: {props.name} with {props.total_memory / 1024**3:.1f}GB memory")
            return torch.device(f"cuda:{best_device}")
        
        # Single GPU system
        props = torch.cuda.get_device_properties(0)
        print(f"Using GPU: {props.name} with {props.total_memory / 1024**3:.1f}GB memory")
        return torch.device("cuda")
    else:
        print("No CUDA device available, using CPU")
        return torch.device("cpu")

def build_model_from_config(config, device=None):
    """Build model from configuration."""
    # Import here to avoid circular imports
    from src.model import MixtureGP
    
    # Set device if not provided
    if device is None:
        device = get_device()
    
    # Extract model hyperparameters from the config
    input_dim  = config["model"]["input_dim"]
    z_dim      = config["model"]["z_dim"]
    num_basis  = config["model"]["num_basis"]
    num_experts= config["model"]["num_experts"]
    K          = config["model"]["K"]
    
    # Create model
    model = MixtureGP(input_dim=input_dim, z_dim=z_dim,
                      num_basis=num_basis, num_experts=num_experts, K=K)
    
    return model.to(device)

def print_parameter_info(model):
    """Print detailed parameter information for a model."""
    total_params = 0
    print("=== Parameter Details ===")
    for name, param in model.named_parameters():
        count = param.numel()
        total_params += count
        print(f"{name:40s} shape={str(param.shape):20s} count={count}")
    print(f"\nTotal Parameters: {total_params:,}")

def print_module_param_counts(model):
    """Print parameter counts by module."""
    print("\n=== Parameter Counts by Module (non-recursive) ===")
    module_counts = {}
    for module_name, module in model.named_modules():
        # Count only parameters directly in this module (not from submodules)
        params = list(module.parameters(recurse=False))
        if params:
            count = sum(p.numel() for p in params)
            key = f"{module_name} ({module.__class__.__name__})" if module_name else f"(root) {module.__class__.__name__}"
            module_counts[key] = count
    for mod, count in module_counts.items():
        print(f"{mod:40s} count={count:,}")

def print_parameter_types(model):
    """Print parameter counts by type (weights, biases, others)."""
    weight_count = 0
    bias_count = 0
    other_count = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            weight_count += param.numel()
        elif "bias" in name:
            bias_count += param.numel()
        else:
            other_count += param.numel()
    print("\n=== Parameter Counts by Type ===")
    print(f"Weights: {weight_count:,}")
    print(f"Biases : {bias_count:,}")
    print(f"Others : {other_count:,}")


def load_model_checkpoint(model, checkpoint_path, device, strict=True):
    """
    Load model checkpoint with cross-platform compatibility.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to
        strict: Whether to strictly enforce that the keys match
    
    Returns:
        The model with loaded weights
    """
    import torch
    
    # Load checkpoint with map_location for cross-platform compatibility
    # Handle PyTorch 2.6+ weights_only parameter
    try:
        # Try with weights_only=False for compatibility with our checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Older PyTorch versions don't have weights_only parameter
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both old and new checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New format with metadata
        state_dict = checkpoint['model_state_dict']
        if 'pytorch_version' in checkpoint:
            print(f"  Model saved with PyTorch {checkpoint['pytorch_version']}")
        if 'cuda_available' in checkpoint:
            print(f"  Model saved on {'GPU' if checkpoint['cuda_available'] else 'CPU'}")
    else:
        # Old format - just the state dict
        state_dict = checkpoint
    
    # Fix compiled model keys if necessary
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        # Remove the '_orig_mod.' prefix from keys
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    
    # Load the state dict
    model.load_state_dict(state_dict, strict=strict)
    
    # Move model to the correct device
    model = model.to(device)
    
    return model
