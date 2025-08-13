#!/usr/bin/env bash
# Generic project-local Conda bootstrap for NSM-DCA
# Compatible with macOS (M1/Intel) and Linux/WSL
# ------------------------------------
set -euo pipefail

# Check if conda is available
if ! command -v conda &> /dev/null; then
  echo "❌  conda not found! Please install Anaconda or Miniconda first."
  echo "    Visit: https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# Load Conda's shell helpers into *this* subshell
eval "$(conda shell.bash hook)"

PROJECT_NAME=$(basename "$PWD")
echo "🚀  setting up $PROJECT_NAME …"

# ------------------------------------------------------------------
# 1. ensure we start from the base env (avoids nesting inside dev/…)
# ------------------------------------------------------------------
while [[ ${CONDA_DEFAULT_ENV:-base} != "base" ]]; do
  conda deactivate
done

# ------------------------------------------------------------------
# 2. wipe any existing env with the same name (interactive confirm)
# ------------------------------------------------------------------
if conda env list | grep -E "^\\*?\\s*$PROJECT_NAME\\s" >/dev/null; then
  read -rp "⚠️  env '$PROJECT_NAME' exists – delete? [y/N] " ans
  [[ $ans =~ ^[Yy]$ ]] || { echo "abort"; exit 1; }
  conda env remove -n "$PROJECT_NAME" -y
  echo "🗑️  removed old env"
fi

# ------------------------------------------------------------------
# 3. create & activate a *named* env (name ⇒ portable activation)
# ------------------------------------------------------------------
conda create -n "$PROJECT_NAME" python=3.11 -y
conda activate "$PROJECT_NAME"
echo "✅  created and activated $PROJECT_NAME"

# ------------------------------------------------------------------
# 4. install dependencies
#    • Create requirements.txt if not present
# ------------------------------------------------------------------
if [[ ! -f requirements.txt ]]; then
  echo "📝  Creating requirements.txt …"
  cat > requirements.txt << 'EOF'
torchdiffeq>=0.2.3
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.2.0
pyyaml>=6.0
tabulate>=0.9.0
scipy>=1.10.0
EOF
fi

# ------------------------------------------------------------------
# 5. install PyTorch based on platform
# ------------------------------------------------------------------
echo "🔍  Detecting platform for PyTorch installation …"

# Detect if we're on macOS with Apple Silicon
if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
  echo "🍎  Detected Apple Silicon Mac"
  pip install torch>=2.0.0
# For Linux (including WSL) and Intel Macs
else
  echo "🐧  Detected Linux/WSL/Intel platform"
  # Check if CUDA is available
  if command -v nvidia-smi &> /dev/null; then
    echo "🎮  NVIDIA GPU detected, installing PyTorch with CUDA support"
    pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
  else
    echo "💻  No NVIDIA GPU detected, installing CPU-only PyTorch"
    pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
  fi
fi

echo "📦  installing remaining dependencies from requirements.txt …"
pip install -r requirements.txt

echo -e "\n✨  done!  next steps:\n  conda activate $PROJECT_NAME\n"