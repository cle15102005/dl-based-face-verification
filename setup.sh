#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Creating virtual environment..."
python3.11 -m venv .venv

echo "[INFO] Activating virtual environment..."
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[INFO] Upgrading pip..."
pip install --upgrade pip

echo "[INFO] Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo "[INFO] Installing project dependencies..."
pip install -r requirements.txt

echo "[INFO] Verifying GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

echo "[INFO] Setup complete. Activate with: source .venv/bin/activate"
