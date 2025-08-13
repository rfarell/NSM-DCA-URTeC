 # NSM‑DCA: Field‑Scale Bayesian Production Forecasting via Spectral GP Mixtures

Neural Spectral Mixture (NSM) decline‑curve analysis for field‑scale multiphase production forecasting. This repository contains the official PyTorch implementation accompanying the URTeC 2025 paper “Field‑Scale Bayesian Production Forecasting via Spectral Gaussian‑Process Mixtures.”

## Overview

NSM‑DCA is a covariate‑aware mixture‑of‑experts Gaussian‑process ODE that provides calibrated, continuous‑time forecasts with principled uncertainty at field scale. It combines:
- Spectral mixture GP experts with Random Fourier Features (near‑linear complexity)
- Covariate‑conditioned gating for cross‑well knowledge sharing
- Physics‑aware algebraic taper for realistic long‑term decline
- Full 3×3 noise covariance for gas, oil, and water
- Variational inference for scalable, calibrated posteriors

Applied in the paper to 8,761 Middle Bakken and Three Forks wells, the approach reduced the P90–P10 interval width by 32% and CRPS by 18% versus a state‑of‑the‑art Bayesian dual‑regime DCA, while remaining well‑calibrated even for wells with little or no history.

## Quick Start

1) Install dependencies
```bash
./setup.sh           # recommended
# or
pip install -r requirements.txt
```

2) Train a model
```bash
python scripts/train_model.py --config config.yaml --experiment exp1
```

3) Evaluate and generate figures/tables
```bash
python scripts/evaluate_model.py --experiment exp1
```

Options commonly used:
- `--device {cpu,cuda}`: select device (auto‑detect by default)
- `--seed INT`: reproducibility control
- `--components {metrics,figures,tables,gradients}` in `evaluate_model.py`

## Data and Configuration

- Input data: place CSVs under `data/` with time‑series (gas, oil, water) and static well covariates.
- Configuration: edit `config.yaml` for paths, hyperparameters, training/evaluation settings.
- Experiments: each run creates `experiments/<name>/` with checkpoints and evaluation outputs.

## Repository Structure

- `src/`: core implementation (`model.py`, `trainer.py`, `evaluation.py`, `data_processor.py`, `utils.py`)
- `scripts/`: entry points (`train_model.py`, `evaluate_model.py`, `evaluate_comprehensive.py`, `eval_production_gradients.py`, utilities)
- `experiments/`: per‑run artifacts (configs, checkpoints, metrics, figures, LaTeX tables)
- `data/`: input datasets (user‑provided)
- `config.yaml`: default configuration template

## Evaluation Outputs

Running `evaluate_model.py` organizes results under `experiments/<name>/evaluation/`:
- `figures/`: publication‑ready scatter and trajectory plots (300 DPI)
- `metrics/`: CSV summaries (R², RMSE, MAE, CRPS)
- `tables/`: LaTeX tables ready for inclusion in manuscripts
- `gradients/`: optional production‑gradient visualizations

See `EVALUATION_README.md` for details and examples.

## Notes on Methodology

- Continuous‑time dynamics via a GP‑driven ODE enable irregular sampling.
- Spectral mixture kernels (Q=3 by default) capture multi‑scale temporal behavior.
- Covariate‑aware gating adapts experts across heterogeneous reservoirs.
- Physics‑aware tapering ensures finite EUR and smooth decline.

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

This project is licensed under the MIT License — see the `LICENSE` file for details.

