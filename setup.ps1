$ErrorActionPreference = "Stop"

Write-Host "[INFO] Creating virtual environment..."
python -m venv .venv

Write-Host "[INFO] Activating virtual environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "[INFO] Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "[INFO] Installing PyTorch (CUDA 11.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Write-Host "[INFO] Installing project dependencies..."
pip install -r requirements.txt

Write-Host "[INFO] Verifying GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

Write-Host "[INFO] Setup complete. Activate with: .venv\Scripts\Activate.ps1"
