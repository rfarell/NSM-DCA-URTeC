 # Field-Scale Bayesian Production Forecasting via Spectral Gaussian-Process Mixtures

This repository contains the official PyTorch implementation of the paper "Field-Scale Bayesian Production Forecasting via Spectral Gaussian‑Process Mixtures" presented at URTeC 2025.

The model implements a Neural State-Space Model (NSM) for Decline Curve Analysis (DCA) using a probabilistic framework that combines:
- Spectral Mixture Kernels for capturing multi-scale temporal patterns
- Gaussian Process Mixtures with variational inference
- Neural ODEs for continuous-time dynamics
- Full covariance modeling for multi-phase production (gas, oil, water)

## Table of Contents

- [Model Architecture](#model-architecture)
- [Mathematical Formulation](#mathematical-formulation)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Experiments](#experiments)
  - [Additional Scripts](#additional-scripts)
- [Project Structure](#project-structure)
- [Experimental Framework](#experimental-framework)
- [Evaluation Framework](#evaluation-framework)
- [Technical Details](#technical-details)
- [Citation](#citation)
- [License](#license)

## Model Architecture

The model implements the methodology described in Farell et al. (2025) with three main components:

1. **Spectral Gaussian Process Experts**: Each expert uses a Spectral Mixture Kernel with Random Fourier Features (RFF) approximation. The spectral mixture allows capturing multi-scale temporal patterns in production data.

2. **Mixture-of-Experts Framework**: Multiple GP experts are combined via an input-dependent gating network, allowing the model to adapt to different production regimes and geological conditions.

3. **Neural ODE Integration**: The entire system is formulated as a continuous-time state-space model, enabling predictions at arbitrary time points.

Key features:
- **Spectral Mixture Kernels**: Q=3 mixture components per expert to capture different frequency scales
- **Variational Inference**: Global variational posterior over RFF weights for uncertainty quantification
- **Full Covariance Modeling**: 3×3 noise covariance matrix for gas, oil, and water phases
- **Algebraic Decline Taper**: Physical constraints for realistic long-term decline behavior
- **Scalable Implementation**: Supports thousands of wells with efficient batched computation

## Mathematical Formulation

The model formulates production forecasting as a continuous-time neural state-space model:

$$\frac{d\mathbf{x}(t)}{dt} = f_\theta(\mathbf{x}(t), t, \mathbf{z}) \cdot \text{taper}(t)$$

Where:
- $\mathbf{x}(t) \in \mathbb{R}^3$ represents the production rates (gas, oil, water)
- $t$ is time in days (scaled by 0.01 for numerical stability)
- $\mathbf{z} \in \mathbb{R}^{19}$ contains static well features (geological and completion parameters)
- $\text{taper}(t)$ enforces algebraic decline: $(1 + t/t_c)^{-\alpha}$

The mixture model with spectral kernels:

$$f_\theta(\mathbf{x}, t, \mathbf{z}) = \sum_{m=1}^M \pi_m(\mathbf{x}, t, \mathbf{z}) \cdot f_m(\mathbf{x}, t, \mathbf{z})$$

Each expert $f_m$ uses a spectral mixture kernel:

$$k(\mathbf{u}, \mathbf{u}') = \sum_{q=1}^Q w_q \prod_{d=1}^D \exp\left(-2\pi^2(u_d - u'_d)^2 v_{q,d}\right) \cos\left(2\pi(u_d - u'_d)\mu_{q,d}\right)$$

Where:
- $Q$ = 3 spectral components
- $w_q$ are mixture weights
- $\mu_{q,d}$ are frequency means
- $v_{q,d}$ are frequency variances

The RFF approximation samples frequencies from this spectral mixture distribution.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/NSM-DCA-URTeC.git
cd NSM-DCA-URTeC

# Run the setup script (recommended)
./setup.sh

# Or manually install dependencies
pip install -r requirements.txt
```

### Dependencies
- PyTorch (automatically detects CUDA/CPU)
- torchdiffeq (>=0.2.3): Neural ODE solver
- pandas, numpy, matplotlib, seaborn: Data processing and visualization
- scikit-learn: Data preprocessing
- scipy: Statistical functions
- pyyaml: Configuration management

### Platform Support

The codebase automatically detects and uses the best available device based on your hardware:

- **Windows/Linux with NVIDIA GPUs**: CUDA acceleration is used if available
- **Other systems**: CPU is used as fallback

To manually specify a device, use the `--device` argument with any script:

```bash
# Force CPU usage
python scripts/train_model.py --device cpu

# Use CUDA (NVIDIA GPUs on Windows/Linux)
python scripts/train_model.py --device cuda
```

#### Platform-Specific Notes

- **CUDA Systems**: GPU acceleration is used for faster training.
- **CPU Only Systems**: The code automatically uses `torch.compile` for performance optimization when available, which can significantly speed up training.
- **All platforms**: The model will automatically adjust ODE solver parameters based on your hardware to ensure stability.

## Usage

### Data Preparation

Place your production data in CSV format in the `data/` directory. The file should include:
- Time-series production data for gas, oil, and water
- Static well features (completion parameters, geological features, etc.)

### Configuration

Edit the `config.yaml` file to adjust model hyperparameters, training settings, and file paths.

### Training

Train the model using the training script:

```bash
# Basic training
python scripts/train_model.py --config config.yaml

# Create and run a new experiment
python scripts/train_model.py --experiment exp1

# Resume an existing experiment
python scripts/train_model.py --experiment exp1

# Start a new run with an existing experiment configuration
python scripts/train_model.py --experiment exp1 --no-resume
```

Additional options:
- `--device`: Specify 'cpu' or 'cuda' (default: auto-detect)
- `--seed`: Set random seed for reproducibility
- `--eval-interval`: Set evaluation interval during training
- `--experiment`: Name of experiment to create or resume
- `--no-resume`: Start a new training run even if experiment exists

### Evaluation

The evaluation framework provides comprehensive model analysis:

```bash
# Run all evaluation components
python scripts/evaluate_model.py --experiment exp1

# Run specific evaluation components
python scripts/evaluate_model.py --experiment exp1 --eval-type scatter
python scripts/evaluate_model.py --experiment exp1 --eval-type trajectories
python scripts/evaluate_model.py --experiment exp1 --eval-type metrics
python scripts/evaluate_model.py --experiment exp1 --eval-type gradients
python scripts/evaluate_model.py --experiment exp1 --eval-type tables
```

Options:
- `--device`: Device to use (cpu or cuda)
- `--num-mc`: Number of Monte Carlo samples for prediction (default: 10)
- `--num-traj`: Number of trajectory samples for trajectory plots (default: 100)
- `--num-wells`: Number of random wells to sample for trajectory plots (default: 10)
- `--eval-type`: Type of evaluation to perform (scatter, trajectories, metrics, gradients, tables, or all)
- `--seed`: Random seed for reproducibility (default: 42)
- `--specific-wells`: Comma-separated list of specific well indices to plot

### Experiments

The repository includes specialized experiments for systematic analysis:

#### Data Size Impact Experiment
Evaluates model performance across different training dataset sizes:

```bash
python scripts/experiment_data_size.py --train-sizes 0.1,0.2,0.3,0.5,0.7,1.0
```

#### Architecture Sensitivity Experiment
Explores the impact of model architecture parameters:

```bash
python scripts/experiment_architecture.py --experts 1,2,4,8,16 --basis 4,8,16,32,64,128
```

### Additional Scripts

#### Experiment Management
List and manage your experiments:

```bash
# List all experiments with basic info
python scripts/list_experiments.py

# Get detailed information about experiments
python scripts/list_experiments.py --detail
```

#### Batch Size Optimization
Find the optimal batch size for your hardware:

```bash
# Run batch size optimization
python scripts/optimize_batch_size.py

# Customize the batch size range
python scripts/optimize_batch_size.py --start-batch 128 --end-batch 2048 --epochs 30
```

#### Model Information
View detailed information about the model:

```bash
python scripts/model_info.py --config config.yaml --model models/model.pth
```

#### Prediction
Generate predictions with a trained model:

```bash
python scripts/predict.py --config config.yaml --output predictions.csv
```

Additional options:
- `--well-id`: ID of specific well to predict and plot
- `--num-mc`: Number of Monte Carlo samples for prediction

## Project Structure

```
NSM-DCA/
├── config.yaml              # Main configuration file
├── data/                    # Data directory
│   └── bakken.csv           # Bakken formation dataset
├── experiments/             # Experiment directories for different training runs
│   └── {experiment_id}/     # Individual experiment folders
│       ├── config.yaml      # Experiment-specific configuration
│       ├── model.pth        # Saved model checkpoint
│       ├── plots/           # Experiment-specific plots
│       ├── scaler.pkl       # Saved feature scaler
│       ├── training_stats.pt # Saved training statistics
│       └── evaluation/      # All evaluation outputs
│           ├── metrics.csv
│           ├── metrics_summary.txt
│           ├── pred_vs_actual_*.png
│           ├── trajectories_*.png
│           ├── gradients/   # Production gradient visualizations
│           └── tables/      # LaTeX tables for publication
├── models/                  # Legacy folder for saved model checkpoints and scalers
├── notebooks/               # Jupyter notebooks for analysis
├── plots/                   # Legacy folder for generated plots and visualizations
├── scripts/                 # Utility scripts
│   ├── train_model.py       # Train the model
│   ├── evaluate_model.py    # Main evaluation script
│   ├── eval_*.py            # Specialized evaluation scripts
│   ├── experiment_*.py      # Experiment scripts
│   ├── list_experiments.py  # List and manage training experiments
│   ├── model_info.py        # Print model information
│   ├── optimize_batch_size.py # Find optimal batch size
│   └── predict.py           # Generate predictions
└── src/                     # Source code
    ├── data_processor.py    # Data loading and preprocessing
    ├── evaluation.py        # Model evaluation utilities
    ├── model.py             # MixtureGP model implementation
    ├── trainer.py           # Model training and evaluation
    └── utils.py             # Utility functions
```

## Experimental Framework

The repository includes three distinct types of experiments:

### 1. Main Model Evaluation
Trains and evaluates the NSM-DCA model with default hyperparameters, forming the baseline for all other experiments.

### 2. Data Size Impact Experiment
Evaluates model performance across different training dataset sizes to understand data efficiency and learning curves.

**Methodology:**
- Creates a fixed test set (default 30% of data)
- Varies the training set size from a small fraction to the full available training data
- Maintains the same model architecture and hyperparameters across all runs
- Evaluates performance metrics on the fixed test set

**Key Outputs:**
- Learning curves showing R² and RMSE as functions of training data size
- Phase-specific performance across dataset sizes
- Analysis of the minimum data required for adequate performance

### 3. Architecture Sensitivity Experiment
Explores the impact of model architecture parameters (number of experts and basis functions) on performance.

**Methodology:**
- Performs a grid search across number of experts (M) and basis functions (R)
- Each configuration is trained with the same dataset and hyperparameters
- Number of parameters scales approximately as: `Parameters ≈ M × (2R + overhead) + gating network`

**Key Outputs:**
- Heatmaps of R² and RMSE across the architecture grid
- Performance vs. parameter count analysis
- Training time measurements across architectures
- Identification of optimal architecture configurations

## Evaluation Framework

The evaluation framework is organized around:

### Core Components
- **Core Utilities**: Common functions in `src/evaluation.py`
- **Specialized Scripts**: Individual scripts for specific evaluation tasks
- **Master Script**: Main script that orchestrates the evaluation process

### Available Evaluation Scripts

1. **eval_scatter_plots.py**: Generate scatter plots comparing predicted vs. actual production
2. **eval_trajectories.py**: Generate trajectory plots with uncertainty bounds
3. **eval_metrics.py**: Calculate and report model evaluation metrics (R², RMSE, MAE, MAPE)
4. **eval_production_gradients.py**: Visualize production gradients to analyze model dynamics
   - Phase-Phase Gradients: Vector fields showing gradients between two production phases
   - Time-Phase Gradients: Vector fields showing how production rates change over time
5. **eval_latex_tables.py**: Generate publication-ready LaTeX tables of model performance metrics

### Extending the Framework

To add new evaluation capabilities:
1. Add common utilities to `src/evaluation.py` if they will be used by multiple scripts
2. Create a new script in the `scripts/` directory following the naming convention `eval_*.py`
3. Use the `load_experiment()` function to access model, data, and configuration
4. Update `evaluate_model.py` if the new script should be included in the orchestration

## Technical Details

### Key Implementation Features

- **Spectral Mixture Kernels**: Each GP expert uses Q=3 spectral components to capture multi-scale temporal patterns
- **Variational Inference**: Global variational posterior with KL divergence regularization for uncertainty quantification
- **Hardware Optimization**: Automatic CUDA/CPU detection with torch.compile optimization
- **Numerical Stability**: Adaptive ODE solver (midpoint method) with configurable tolerances
- **Scalable Architecture**: 
  - Default: M=16 experts, R=8 basis functions per expert
  - Supports batches of 8192+ samples for efficient training
- **Physical Constraints**: Algebraic taper function ensures realistic long-term decline behavior
- **Multi-phase Modeling**: Full 3×3 covariance matrix captures correlations between gas, oil, and water production

### Dataset

The implementation is demonstrated on the Bakken formation dataset containing:
- Production data from multiple wells in the Bakken shale play
- 19 static features including geological and completion parameters
- Time-series production rates for gas, oil, and water phases
- Suitable for field-scale production forecasting tasks

## Citation

If you use this code for your research, please cite:

```bibtex
@inproceedings{farell2025field,
  author       = {Farell, Ryan and Bickel, J.\,Eric and Bajaj, Chandrajit},
  title        = {Field-Scale Bayesian Production Forecasting via Spectral Gaussian‑Process Mixtures},
  booktitle    = {Proceedings of the SPE/AAPG/SEG Unconventional Resources Technology Conference},
  address      = {Houston, Texas, USA},
  month        = jun,
  year         = {2025},
  doi          = {10.15530/urtec-2025-4265618},
  paper_number = {URTEC-4265618-MS},
  publisher    = {Unconventional Resources Technology Conference (URTeC)}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.