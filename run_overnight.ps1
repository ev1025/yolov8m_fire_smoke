# 오버나이트 일괄 실행: 03 추출 -> 04 변환 -> 05 학습(시간상한) -> 06 평가
# 컴퓨터 켜두고 자면 아침에 best.pt + eval_report.md 완성.
# 원본(D:)은 읽기 전용. 모든 산출물은 C:\mro_fire_yolo.
$ErrorActionPreference = "Continue"
$PROJ = Split-Path -Parent $MyInvocation.MyCommand.Path
$WORK = Join-Path $PROJ "work"
$PY   = Join-Path $WORK ".venv\Scripts\python.exe"
$SZ   = "C:\Program Files\7-Zip\7z.exe"
$LOG  = Join-Path $WORK "overnight.log"
$WEIGHTS = Join-Path $PROJ "yolov8m.pt"      # 미리 받아둔 가중치
$TRAIN_HOURS = 10                            # 학습 시간 상한
$IMGSZ = 960

function Log($m){ $t = Get-Date -Format "yyyy-MM-dd HH:mm:ss"; "$t  $m" | Tee-Object -FilePath $LOG -Append }

Set-Location $PROJ
Log "===== 오버나이트 시작 ====="

# 1) 추출
Log "[1/4] 03 선택추출 시작"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PROJ "03_extract_images.ps1") *>> $LOG
if ($LASTEXITCODE -ne 0) { Log "[중단] 03 실패 (code $LASTEXITCODE)"; "FAILED:extract" | Set-Content "$WORK\STATUS.txt"; exit 1 }
Log "[1/4] 추출 완료"

# 2) 변환
Log "[2/4] 04 YOLO 변환 시작"
& $PY (Join-Path $PROJ "04_convert_to_yolo.py") *>> $LOG
if ($LASTEXITCODE -ne 0) { Log "[중단] 04 실패"; "FAILED:convert" | Set-Content "$WORK\STATUS.txt"; exit 1 }
Log "[2/4] 변환 완료"

# 2.5) 환경 자가점검 (torch+cuda). 미준비면 setup_env 자동 실행
& $PY -c "import torch,ultralytics; assert torch.cuda.is_available()" 2>$null
if ($LASTEXITCODE -ne 0) {
  Log "[env] torch/cuda 미준비 -> setup_env 자동 설치"
  & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PROJ "setup_env.ps1") *>> $LOG
  & $PY -c "import torch; assert torch.cuda.is_available()" 2>$null
  if ($LASTEXITCODE -ne 0) { Log "[중단] 환경 준비 실패"; "FAILED:env" | Set-Content "$WORK\STATUS.txt"; exit 1 }
}
Log "[env] torch/cuda 준비됨"

# 3) 학습 (시간 상한 -> best.pt 저장 후 자동 종료)
Log "[3/4] 05 학습 시작 (imgsz=$IMGSZ, time=${TRAIN_HOURS}h, yolov8m)"
& $PY (Join-Path $PROJ "05_train.py") --imgsz $IMGSZ --time $TRAIN_HOURS --batch -1 --weights $WEIGHTS *>> $LOG
if ($LASTEXITCODE -ne 0) { Log "[경고] 05 종료코드 $LASTEXITCODE (시간상한 종료일 수 있음)" }
Log "[3/4] 학습 종료"

# best.pt 찾기 (가장 최근)
$best = Get-ChildItem (Join-Path $WORK "runs") -Recurse -Filter best.pt -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime | Select-Object -Last 1
if (-not $best) { Log "[중단] best.pt 없음 - 학습 실패"; "FAILED:train" | Set-Content "$WORK\STATUS.txt"; exit 1 }
Log "best.pt: $($best.FullName)"

# 4) 평가
Log "[4/4] 06 평가 시작"
& $PY (Join-Path $PROJ "06_eval.py") --weights $best.FullName --imgsz $IMGSZ --conf 0.25 *>> $LOG
if ($LASTEXITCODE -ne 0) { Log "[경고] 06 종료코드 $LASTEXITCODE" }
Log "[4/4] 평가 완료"

Log "===== 전체 완료 ====="
"DONE  best=$($best.FullName)" | Set-Content "$WORK\STATUS.txt"
