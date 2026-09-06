$ErrorActionPreference = "Stop"

Write-Host "[ScamShield] Qoder IDE setup starting..." -ForegroundColor Cyan

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found. Install Python 3.11-3.13 and reopen Qoder IDE."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    python -m venv .venv
}

$python = Join-Path $PWD ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment was not created correctly."
}

Write-Host "Upgrading pip..."
& $python -m pip install --upgrade pip

Write-Host "Installing project requirements..."
& $python -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Add GOOGLE_API_KEY before Gemini tests." -ForegroundColor Yellow
} else {
    Write-Host ".env already exists; leaving it unchanged."
}

Write-Host "Running environment doctor..."
& $python scripts\doctor.py

Write-Host "Running unit tests..."
& $python -m pytest -q

Write-Host ""
Write-Host "Qoder setup complete." -ForegroundColor Green
Write-Host "Next: add GOOGLE_API_KEY to .env, then run:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\build_index.py"
Write-Host "  python scripts\evaluate_retrieval.py"
Write-Host "  python scripts\smoke_test.py"
Write-Host "  python app.py"
