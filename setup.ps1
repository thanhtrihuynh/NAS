$ErrorActionPreference = "Stop"

if (-not (Test-Path ".\datasets")) {
    throw "Khong tim thay .\datasets"
}

if (-not (Test-Path ".\.venv")) {
    py -3.12 -m venv .venv
}

& .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r .\requirements.txt

Write-Host ""
Write-Host "Setup completed."
Write-Host "Run: python .\scripts\check_project.py"
