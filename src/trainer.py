# src/trainer.py

import os
import torch
import torch.nn.functional as F
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchdiffeq import odeint
from torch.amp import GradScaler, autocast

class Trainer:
    """
    Handles model training, evaluation, and saving.
    """
    def __init__(self, model, config, device=None):
        """
        Initialize the trainer.
        
        Args:
            model: MixtureGP model to train
            config: Configuration dictionary with training parameters
            device: Device to use for training (cpu or cuda)
        """
        self.model = model
        self.config = config
        if device is None:
            # Import get_device locally to avoid circular imports
            from src.utils import get_device
            self.device = get_device()
        else:
            self.device = device
            
        # Move model to device if not already
        self.model = self.model.to(self.device)
    
        # Enable torch.compile for CUDA devices with safe fallback
        compile_model = config.get("training", {}).get("compile_model", True)
        
        # Check if user explicitly disabled compilation
        import os
        compile_disabled = os.environ.get('TORCH_COMPILE_DISABLE', '0') == '1'
        
        if compile_model and hasattr(torch, 'compile') and self.device.type == 'cuda' and not compile_disabled:
            print("Attempting to compile the model with torch.compile...")
            try:
                # Check GPU memory to decide compilation mode
                props = torch.cuda.get_device_properties(self.device)
                print(f"GPU Memory: {props.total_memory / 1e9:.1f} GB")
                
                # Be more conservative with compilation to avoid slowdowns
                if props.total_memory > 24 * 1024**3:  # More than 24GB (e.g., RTX 3090, A100)
                    # Only use max-autotune for very powerful GPUs
                    self.model = torch.compile(self.model, mode="max-autotune")
                    print("Model compiled with max-autotune mode!")
                elif props.total_memory > 12 * 1024**3:  # 12-24GB
                    # Use reduce-overhead for medium GPUs
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    print("Model compiled with reduce-overhead mode!")
                else:
                    # Skip compilation for smaller GPUs as it often slows things down
                    print(f"Skipping torch.compile for GPU with {props.total_memory / 1e9:.1f}GB memory (often slower)")
            except Exception as e:
                print(f"Could not compile model: {e}. Running un-compiled.")
        elif compile_disabled:
            print("Torch compilation disabled via TORCH_COMPILE_DISABLE=1")
        elif self.device.type == 'cpu':
            print("Running on CPU - torch.compile not used")
                
        # Training hyperparameters
        self.learning_rate = config["training"]["learning_rate"]
        self.batch_size = config["training"]["batch_size"]
        self.total_steps = config["training"]["total_steps"]
        self.num_mc = config["model"]["num_mc"]

        # KL annealing schedule (linear 0→1 over kl_anneal_steps)
        self.kl_anneal_steps = config["training"].get("kl_anneal_steps", 1000)
        
        # Cache constants
        self.time_scale = 100.0  # Inverse of 0.01 scaling
        self.kl_weights = config.get('training', {}).get('kl_weights', None)
        
        # Loss hyperparameters from manuscript
        self.lambda_ent = 1e-2  # Entropy bonus weight
        self.lambda_1 = 1.0     # Monotone decline penalty weight (reduced from 1e3)
        self.lambda_2 = 1.0     # Concavity penalty weight (reduced from 1e3)
        
        # Apply weight decay only to the gating network parameters
        gating_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'gating_net' in name:
                gating_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different weight decay
        param_groups = [
            {'params': gating_params, 'weight_decay': 0.01},  # Apply weight decay to gating network
            {'params': other_params, 'weight_decay': 0.0}     # No weight decay for other parameters
        ]
        
        # Optimizer with parameter groups
        self.optimizer = torch.optim.AdamW(param_groups, lr=self.learning_rate)
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler('cuda', enabled=(self.device.type == 'cuda'))
        
    def train(self, x_train, z_train, x_test=None, z_test=None, t_vals=None,
              scale_train=None, scale_test=None,
              eval_interval=100, checkpoint_path=None, plots_dir=None, initial_stats=None):
        """
        Train the model.
        
        Args:
            x_train: Training production data tensor [N, T, 3]
            z_train: Training static features tensor [N, z_dim]
            x_test: Test production data tensor [N, T, 3] (optional)
            z_test: Test static features tensor [N, z_dim] (optional)
            t_vals: Time values tensor [T] (optional)
            eval_interval: Interval for evaluation and plotting
            checkpoint_path: Path to save model checkpoints
            plots_dir: Directory to save plots
            initial_stats: Initial training statistics to continue from a previous run
            
        Returns:
            dict: Training statistics
        """
        # Total steps to run in this session
        total_steps = self.total_steps
        # Create plots directory if needed
        if plots_dir:
            os.makedirs(plots_dir, exist_ok=True)
        
        # For GPU training, we can keep data on GPU and use a custom collate function
        # This avoids CPU-GPU transfers during training
        if self.device.type == 'cuda':
            # Keep data on GPU
            x_train_gpu = x_train.to(self.device) if not x_train.is_cuda else x_train
            z_train_gpu = z_train.to(self.device) if not z_train.is_cuda else z_train
            # For DataLoader, we'll use indices instead of actual data
            train_indices = torch.arange(len(x_train), device='cpu')
        else:
            # For CPU training, use original approach
            x_train_cpu = x_train.cpu() if x_train.is_cuda else x_train
            z_train_cpu = z_train.cpu() if z_train.is_cuda else z_train
        
        # Move non-batched data to device
        if t_vals is not None:
            t_vals = t_vals.to(self.device)
        
        if x_test is not None and z_test is not None:
            x_test = x_test.to(self.device)
            z_test = z_test.to(self.device)
            
        # Create data loader with optimizations
        if self.device.type == 'cuda':
            # Custom dataset that returns indices for GPU data
            class GPUIndexDataset(torch.utils.data.Dataset):
                def __init__(self, length):
                    self.length = length
                
                def __len__(self):
                    return self.length
                
                def __getitem__(self, idx):
                    return idx
            
            train_ds = GPUIndexDataset(len(x_train_gpu))
            
            # Custom collate function to fetch GPU data
            def gpu_collate_fn(indices):
                indices = torch.tensor(indices, device=self.device, dtype=torch.long)
                return x_train_gpu[indices], z_train_gpu[indices]
            
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=gpu_collate_fn,
                persistent_workers=False
            )
        else:
            # CPU training path
            train_ds = TensorDataset(x_train_cpu, z_train_cpu)
            train_loader = DataLoader(
                train_ds,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
        
        # Training statistics - initialize with previous stats if available
        if initial_stats:
            # Continue from existing stats
            stats = initial_stats
            # Calculate the starting step (1-indexed)
            start_step = stats['steps'][-1] + 1 if stats['steps'] else 1
            print(f"Continuing training from step {start_step}")
        else:
            # Start fresh
            stats = {
                'steps': [],
                'losses': [],
                'nll_values': [],
                'kl_values': [],
                'entropy_values': [],
                'pen1_values': [],
                'pen2_values': [],
                'times': [],
                'train_r2': [],
                'test_r2': [],
                'train_rmse': [],
                'test_rmse': [],
                # Phase-specific metrics
                'train_r2_gas': [],
                'train_r2_oil': [],
                'train_r2_water': [],
                'test_r2_gas': [],
                'test_r2_oil': [],
                'test_r2_water': [],
                'train_rmse_gas': [],
                'train_rmse_oil': [],
                'train_rmse_water': [],
                'test_rmse_gas': [],
                'test_rmse_oil': [],
                'test_rmse_water': []
            }
            start_step = 1
        
        # Import evaluation utilities outside the loop
        from src.utils import evaluate_model, plot_pred_vs_actual, print_metrics

        # Function to perform evaluation
        def perform_evaluation(step, is_final=False):
            prefix = "Final" if is_final else f"Step {step}"
            print(f"\n--- {prefix} Evaluation ---")
            
            # Evaluate on training data (use GPU data if available)
            if self.device.type == 'cuda':
                x_train_device = x_train_gpu
                z_train_device = z_train_gpu
            else:
                x_train_device = x_train_cpu.to(self.device)
                z_train_device = z_train_cpu.to(self.device)
            train_metrics, _ = evaluate_model(
                self.model, x_train_device, z_train_device, t_vals,
                scale_tensor=scale_train,
                num_mc=self.num_mc, device=self.device
            )
            print_metrics(train_metrics, prefix="Training")
            
            # Track the training metrics
            stats['train_r2'].append(train_metrics['Overall_R2'])
            stats['train_rmse'].append(train_metrics['Overall_RMSE'])
            
            # Track phase-specific training metrics
            for phase in ['Gas', 'Oil', 'Water']:
                phase_lower = phase.lower()
                stats[f'train_r2_{phase_lower}'].append(train_metrics[f'{phase}_R2_final'])
                stats[f'train_rmse_{phase_lower}'].append(train_metrics[f'{phase}_RMSE_final'])
                print(f"{phase}: R²={train_metrics[f'{phase}_R2_final']:.4f}, RMSE={train_metrics[f'{phase}_RMSE_final']:.4f}")
            
            # Evaluate and plot on test data if available
            if x_test is not None and z_test is not None:
                # Generate plot path if needed
                plot_path = None
                if plots_dir:
                    # Create eval_steps subfolder
                    eval_dir = os.path.join(plots_dir, 'eval_steps')
                    os.makedirs(eval_dir, exist_ok=True)
                    plot_path = os.path.join(eval_dir, f'eval_{prefix.lower()}.png')
                
                # Evaluate and plot (save to file, don't display)
                test_metrics, _ = plot_pred_vs_actual(
                    self.model, x_test, z_test, t_vals,
                    scale_tensor=scale_test,
                    num_mc=self.num_mc, device=self.device,
                    save_path=plot_path,
                    show_plot=False
                )
                print_metrics(test_metrics, prefix="Test")
                
                # Track the test metrics
                stats['test_r2'].append(test_metrics['Overall_R2'])
                stats['test_rmse'].append(test_metrics['Overall_RMSE'])
                
                # Track phase-specific test metrics
                for phase in ['Gas', 'Oil', 'Water']:
                    phase_lower = phase.lower()
                    stats[f'test_r2_{phase_lower}'].append(test_metrics[f'{phase}_R2_final'])
                    stats[f'test_rmse_{phase_lower}'].append(test_metrics[f'{phase}_RMSE_final'])
                    print(f"{phase}: R²={test_metrics[f'{phase}_R2_final']:.4f}, RMSE={test_metrics[f'{phase}_RMSE_final']:.4f}")
            
            # Save checkpoint
            if checkpoint_path:
                self.save_model(checkpoint_path)
                print(f"Model checkpoint saved to {checkpoint_path}")
            
            # Return a row of metrics for progress table
            if x_test is not None and z_test is not None:
                return {
                    'step': step,
                    'loss': loss.item(),
                    'train_r2': train_metrics['Overall_R2'],
                    'test_r2': test_metrics['Overall_R2'],
                    'train_rmse': train_metrics['Overall_RMSE'],
                    'test_rmse': test_metrics['Overall_RMSE']
                }
            return {
                'step': step,
                'loss': loss.item(),
                'train_r2': train_metrics['Overall_R2'],
                'train_rmse': train_metrics['Overall_RMSE']
            }

        # Training loop
        progress_rows = []
        for step in range(start_step, start_step + self.total_steps):
            # Initialize timing trackers for each step
            step_times = {
                'data_prep': [], 'optim_clear': [], 'forward': [],
                'loss_compute': [], 'backward': [], 'optim_step': [], 'total': []
            }
            
            for batch_x, batch_z in train_loader:
                # Data is already on device if using GPU
                if self.device.type != 'cuda':
                    batch_x = batch_x.to(self.device)
                    batch_z = batch_z.to(self.device)
                
                B, T, D = batch_x.shape
                step_start = time.time()
                
                # Random start and end indices for sub-series approach
                start_idx = np.random.randint(0, max(1, T-4))  # Ensure we can get longer sequences
                min_length = min(4, T - start_idx)  # Prefer at least 4 time points for better penalty computation
                max_length = T - start_idx  # Maximum possible length
                length = np.random.randint(min_length, max_length + 1)  # Random length
                end_idx = start_idx + length  # Calculate end index
                
                # Create sub-sequence - Use actual time values (no resetting to zero)
                t_sub = t_vals[start_idx:end_idx]  # Use absolute time values
                
                x0 = batch_x[:, start_idx, :]          # [B, 3]
                x_target = batch_x[:, start_idx:end_idx, :]  # [B, length, 3]
                data_prep_time = time.time() - step_start
                step_times['data_prep'].append(data_prep_time)
                
                self.optimizer.zero_grad()
                opt_clear_time = time.time() - (step_start + data_prep_time)
                step_times['optim_clear'].append(opt_clear_time)
                
                def ode_func(t_scalar, x_state):
                    # x_state: [B, 3]
                    return self.model(t_scalar, x_state, batch_z)
                
                nll_list = []
                entropy_list = []
                pen1_list = []  # Monotone decline penalty
                pen2_list = []  # Concavity penalty
                forward_start = time.time()
                
                # Enable gradient tracking for time to compute derivatives
                t_sub_grad = t_sub.clone().detach().requires_grad_(True)
                
                # Use autocast for mixed precision on CUDA
                with autocast('cuda', enabled=(self.device.type == 'cuda')):
                    for mc_idx in range(self.num_mc):
                        # Set up the solver with appropriate tolerances for the device
                        ode_options = {
                            'method': self.config["ode"]["method"],
                            'atol': float(self.config["ode"]["atol"]),
                            'rtol': float(self.config["ode"]["rtol"])
                        }
                        
                        try:
                            # Modified ODE function to capture mixture weights
                            mix_weights_list = []
                            def ode_func_with_weights(t_scalar, x_state):
                                f_out, mix_w = self.model(t_scalar, x_state, batch_z, return_mixture_weights=True)
                                mix_weights_list.append(mix_w)
                                return f_out
                            
                            # Run the ODE solver with error handling
                            pred_traj = odeint(ode_func_with_weights, x0, t_sub_grad, **ode_options)
                            pred_traj = pred_traj.permute(1, 0, 2)  # => [B, T_sub, 3]
                            
                            # Check for NaN values in prediction
                            if torch.isnan(pred_traj).any() or torch.isinf(pred_traj).any():
                                print("Warning: ODE solver produced NaN/Inf values, using fallback prediction")
                                # Use a simple linear interpolation as fallback
                                pred_traj = x0.unsqueeze(1).expand(-1, t_sub.shape[0], -1)
                            
                            # Compute NLL
                            nll = self.model.neg_loglike(pred_traj, x_target, use_merged_cov=True)
                            nll_list.append(nll)
                            
                            # Compute entropy from mixture weights
                            if mix_weights_list:
                                mix_weights_avg = torch.stack(mix_weights_list).mean(dim=0)  # Average over time
                                # Entropy H = -sum(pi * log(pi)) - we want to maximize this
                                entropy = -(mix_weights_avg * mix_weights_avg.clamp(min=1e-8).log()).sum(dim=1).mean()
                                entropy_list.append(entropy)
                            
                            # Compute rates from cumulative production
                            if t_sub.shape[0] > 1:
                                dt = t_sub_grad[1:] - t_sub_grad[:-1]  # Time differences
                                dt = dt.unsqueeze(0).unsqueeze(-1)  # [1, T-1, 1]
                                
                                # Production rates: q = dQ/dt
                                dQ = pred_traj[:, 1:, :] - pred_traj[:, :-1, :]  # [B, T-1, 3]
                                q = dQ / (dt + 1e-8)  # [B, T-1, 3]
                                
                                # Compute first derivative dq/dt using finite differences
                                # Since q is already the rate (dQ/dt), dq/dt is the second derivative of Q
                                if q.shape[1] > 1:
                                    # Time differences for rate calculation
                                    dt_q = dt[:, :-1, :]  # [1, T-2, 1]
                                    dq = q[:, 1:, :] - q[:, :-1, :]  # [B, T-2, 3]
                                    dq_dt = dq / (dt_q + 1e-8)  # [B, T-2, 3] - this is d²Q/dt²
                                    
                                    # Find peak production rate (smooth argmax)
                                    gamma = 20.0  # Temperature for soft max
                                    q_detach = q.detach()
                                    soft_peak_weights = F.softmax(gamma * q_detach, dim=1)  # [B, T-1, 3]
                                    q_peak = (soft_peak_weights * q_detach).sum(dim=1, keepdim=True)  # [B, 1, 3]
                                    q_peak = q_peak.clamp(min=1e-6)  # Avoid division by zero
                                    
                                    # Penalty 1: Post-peak monotone decline
                                    # For decline curves, q (production rate) should be positive but decreasing
                                    # So dq/dt should be negative. We penalize positive dq/dt
                                    # Only apply to gas and oil (indices 0 and 1), not water (index 2)
                                    # Scale penalties by time_scale to account for normalized time
                                    pen1_gas_oil = F.relu(dq_dt[:, :, :2]) / (q_peak[:, :, :2] * self.time_scale)  # [B, T-2, 2]
                                    pen1 = pen1_gas_oil.mean()  # Exclude water from penalty
                                    pen1_list.append(pen1)
                                    
                                    # Penalty 2: Concavity
                                    # For typical DCA curves, we want concave behavior (not convex)
                                    # Concave means d²q/dt² < 0 (rate decline is slowing)
                                    # We need the third derivative of Q, which is d(dq/dt)/dt
                                    if dq_dt.shape[1] > 1:
                                        dt_dq = dt[:, :-2, :]  # [1, T-3, 1]
                                        d2q_dt2 = (dq_dt[:, 1:, :] - dq_dt[:, :-1, :]) / (dt_dq + 1e-8)  # [B, T-3, 3]
                                        # Penalize positive d²q/dt² (convex behavior)
                                        # Only apply to gas and oil (indices 0 and 1), not water (index 2)
                                        # Scale penalties by time_scale squared for second derivative
                                        pen2_gas_oil = F.relu(d2q_dt2[:, :, :2]) / (q_peak[:, :, :2] * self.time_scale * self.time_scale)  # [B, T-3, 2]
                                        pen2 = pen2_gas_oil.mean()  # Exclude water from penalty
                                        pen2_list.append(pen2)
                        
                        except Exception as e:
                            print(f"Error in ODE integration: {e}")
                            # Return a default high loss value to allow training to continue
                            nll = torch.tensor(1e3, device=x0.device)
                            nll_list.append(nll)
                    forward_time = time.time() - forward_start
                    step_times['forward'].append(forward_time)
                    
                    loss_start = time.time()
                    # Safely average NLL values
                    avg_nll = torch.stack(nll_list).mean() if nll_list else torch.tensor(1e3, device=self.device)
                
                    # Get KL divergence
                    kl = self.model.KL_divergence(self.kl_weights)

                    # Linear KL-annealing from 0 → 1 over kl_anneal_steps
                    kl_weight = min(1.0, step / float(self.kl_anneal_steps))
                    
                    # Check for numerical issues
                    if torch.isnan(avg_nll) or torch.isinf(avg_nll):
                        print("Warning: NLL is NaN/Inf, using fallback value")
                        avg_nll = torch.tensor(1e3, device=self.device)
                    
                    # Average entropy bonus (we want to maximize entropy, so subtract from loss)
                    avg_entropy = torch.stack(entropy_list).mean() if entropy_list else torch.tensor(0.0, device=self.device)
                    
                    # Average penalties
                    avg_pen1 = torch.stack(pen1_list).mean() if pen1_list else torch.tensor(0.0, device=self.device)
                    avg_pen2 = torch.stack(pen2_list).mean() if pen2_list else torch.tensor(0.0, device=self.device)
                    
                    # Compute full loss with all terms
                    # Note: subtract entropy bonus since we want to maximize entropy
                    loss = avg_nll + kl_weight * kl - self.lambda_ent * avg_entropy + self.lambda_1 * avg_pen1 + self.lambda_2 * avg_pen2
                    
                    # Final safety check
                    if torch.isnan(loss) or torch.isinf(loss):
                        print("Warning: Loss is NaN/Inf, using fallback value")
                        loss = torch.tensor(1e3, device=self.device)
                loss_compute_time = time.time() - loss_start
                step_times['loss_compute'].append(loss_compute_time)
                
                backward_start = time.time()
                # Scale loss and perform backward pass for mixed precision
                self.scaler.scale(loss).backward()
                backward_time = time.time() - backward_start
                step_times['backward'].append(backward_time)
                
                opt_start = time.time()
                # Unscale gradients before clipping
                self.scaler.unscale_(self.optimizer)
                # Add gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                # Optimizer step with scaler
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_time = time.time() - opt_start
                step_times['optim_step'].append(optimizer_time)
                
                total_time = time.time() - step_start
                step_times['total'].append(total_time)
                
                # Save statistics
                stats['steps'].append(step)
                stats['losses'].append(loss.item())
                stats['nll_values'].append(avg_nll.item())
                stats['kl_values'].append(kl.item())
                stats['entropy_values'].append(avg_entropy.item())
                stats['pen1_values'].append(avg_pen1.item())
                stats['pen2_values'].append(avg_pen2.item())
                stats['times'].append(total_time)
            
            # Calculate and print average times for the step
            end_step = start_step + total_steps - 1
            print(f"\nStep {step}/{end_step} timing averages:")
            print(f"  Data prep:     {np.mean(step_times['data_prep'])*1000:.2f}ms")
            print(f"  Optim clear:   {np.mean(step_times['optim_clear'])*1000:.2f}ms")
            print(f"  Forward pass:  {np.mean(step_times['forward'])*1000:.2f}ms")
            print(f"  Loss compute:  {np.mean(step_times['loss_compute'])*1000:.2f}ms")
            print(f"  Backward pass: {np.mean(step_times['backward'])*1000:.2f}ms")
            print(f"  Optim step:    {np.mean(step_times['optim_step'])*1000:.2f}ms")
            print(f"  Total time:    {np.mean(step_times['total'])*1000:.2f}ms")
            print(f"Loss={loss.item():.3f}  NLL={avg_nll.item():.3f}  KL={kl.item():.3f}")
            print(f"  Entropy={avg_entropy.item():.4f}  Pen1={avg_pen1.item():.4f}  Pen2={avg_pen2.item():.4f}")
            
            # Debug info for penalty computation
            if step % 50 == 0 and (avg_pen1.item() > 0 or avg_pen2.item() > 0):
                print(f"  [Debug] Sequence length: {length}, Lambda1: {self.lambda_1}, Lambda2: {self.lambda_2}")
            
            # Evaluate and plot at intervals
            if step % eval_interval == 0:
                row = perform_evaluation(step)
                progress_rows.append(row)
        
        # Always perform final evaluation if it wasn't just done
        if self.total_steps % eval_interval != 0:
            perform_evaluation(self.total_steps, is_final=True)
            
        # Print progress table
        if progress_rows:
            print("\n=== Training Progress Summary ===")
            header = "| Step | Loss | Train R² | Test R² | Train RMSE | Test RMSE |"
            separator = "|" + "-" * (len(header) - 2) + "|"
            print(separator)
            print(header)
            print(separator)
            for row in progress_rows:
                if 'test_r2' in row:
                    print(f"| {row['step']:4d} | {row['loss']:.4f} | {row['train_r2']:.4f} | {row['test_r2']:.4f} | {row['train_rmse']:.4f} | {row['test_rmse']:.4f} |")
                else:
                    print(f"| {row['step']:4d} | {row['loss']:.4f} | {row['train_r2']:.4f} | --- | {row['train_rmse']:.4f} | --- |")
            print(separator)
                    
        return stats
    
    def save_model(self, path):
        """Save model state dict to file (CPU-compatible for cross-platform use)."""
        # Get the original model if it's compiled
        if hasattr(self.model, '_orig_mod'):
            # This is a compiled model, save the original module
            state_dict = self.model._orig_mod.state_dict()
        else:
            # Regular model
            state_dict = self.model.state_dict()
        
        # Move all tensors to CPU for cross-platform compatibility
        cpu_state_dict = {k: v.cpu() if torch.is_tensor(v) else v 
                         for k, v in state_dict.items()}
        
        # Save with additional metadata for better compatibility
        checkpoint = {
            'model_state_dict': cpu_state_dict,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
        torch.save(checkpoint, path)
    
    def load_model(self, path):
        """Load model state dict from file (handles cross-platform compatibility)."""
        # Always use map_location to handle CPU/GPU differences
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle both old and new save formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            # Old format - just the state dict
            state_dict = checkpoint
        
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
        
        self.model.load_state_dict(state_dict)
        return self.model