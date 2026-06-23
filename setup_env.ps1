# 전용 가상환경 구축 (재현용). venv는 프로젝트 내 work\.venv 에 둔다.
$ErrorActionPreference = "Stop"
$sys  = "C:\Users\eg287\AppData\Local\Programs\Python\Python312\python.exe"
$venv = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "work\.venv"

if (-not (Test-Path "$venv\Scripts\python.exe")) { & $sys -m venv $venv }
$vpy = "$venv\Scripts\python.exe"

& $vpy -m pip install --upgrade pip
# RTX 4060(Ada) -> CUDA 12.4 휠
& $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& $vpy -m pip install ultralytics

& $vpy -c "import torch,ultralytics; print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('ultralytics',ultralytics.__version__)"
Write-Host "[완료] venv: $vpy"
