# Launch LocalBiz locally for testing. Usage: ./scripts/dev.ps1
# Opens http://localhost:8501 . Needs a trained model first:
#   python -I train.py data/yelp_academic_dataset_business.json
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$py = Join-Path $root ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $py)) {
  Write-Host "No venv found. Creating one (Python 3.13)..."
  py -3.13 -m venv (Join-Path $root ".venv")
  & (Join-Path $root ".venv\Scripts\python.exe") -m pip install -q streamlit pandas scikit-learn joblib
}
& $py run (Join-Path $root "app.py")
