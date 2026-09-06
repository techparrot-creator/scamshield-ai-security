$ErrorActionPreference = "Stop"
if (-not (Test-Path ".venv")) {
  python -m venv .venv
}
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
if (-not (Test-Path ".env")) {
  Copy-Item .env.example .env
}
Write-Host "Setup complete. Add GOOGLE_API_KEY to .env, then run .\run_gradio.ps1" -ForegroundColor Green
