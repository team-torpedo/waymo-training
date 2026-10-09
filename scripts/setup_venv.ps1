# Create the "venv" virtual environment on Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\setup_venv.ps1 [-Dev]
# Native Windows TensorFlow has no GPU support after 2.10; use WSL2 + setup_venv.sh --gpu for GPU.
param([switch]$Dev)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path "venv")) { python -m venv venv }
& .\venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
& .\venv\Scripts\python.exe -m pip install -r requirements.txt
if ($Dev) { & .\venv\Scripts\python.exe -m pip install -r requirements-dev.txt }
& .\venv\Scripts\python.exe -m pip install -e . --no-deps
Write-Host "Done. Activate with:  .\venv\Scripts\Activate.ps1"
