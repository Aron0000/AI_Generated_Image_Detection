$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe train.py `
  --train-generator stable_diffusion_v_1_5 `
  --epochs 5 `
  --freeze-backbone `
  --learning-rate 0.001 `
  --max-train-per-class 1000 `
  --max-eval-per-class 200 `
  --workers 0 `
  --output outputs\smoke_test
