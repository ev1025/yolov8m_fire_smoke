# 03. 선택한 jpg만 분할압축에서 추출 (원본 읽기 전용)
# - 7z 'x'(추출)만 사용. 아카이브/원본 폴더에는 절대 쓰지/지우지 않음.
# - 분할압축(.z01~.z08 + .zip)은 tar로 못 풀므로 7-Zip 필요.
$ErrorActionPreference = "Stop"

# ----- 설정 (config.py와 동일하게 유지) -----
$ZIP      = "D:\엠아르오_학습데이터\01.원천데이터\TS.zip"   # 원본(읽기전용)
$WORK     = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "work"  # 프로젝트 내 work/
$LIST     = Join-Path $WORK "extract_list.txt"        # 클래스폴더 기준 상대경로
$STAGING  = Join-Path $WORK "staging"
$PREFIXOUT= Join-Path $WORK "archive_prefix.txt"
$CANDS    = @("화재현상\이미지", "TS\화재현상\이미지")

# ----- 7-Zip 찾기 -----
$sz = @("C:\Program Files\7-Zip\7z.exe","C:\Program Files (x86)\7-Zip\7z.exe") |
      Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $sz) { $g = Get-Command 7z -ErrorAction SilentlyContinue; if ($g) { $sz = $g.Source } }
if (-not $sz) {
  Write-Host "[에러] 7-Zip 없음. 설치 후 재실행:" -ForegroundColor Red
  Write-Host "  winget install --id 7zip.7zip -e"
  exit 1
}
Write-Host "7-Zip: $sz"

if (-not (Test-Path $LIST)) { Write-Host "[에러] $LIST 없음. 02 먼저 실행." -ForegroundColor Red; exit 1 }
New-Item -ItemType Directory -Force -Path $STAGING | Out-Null

# ----- prefix probe: 첫 항목으로 후보 prefix 검증 -----
$firstRel = (Get-Content -LiteralPath $LIST -TotalCount 1)
$prefix = $null
foreach ($c in $CANDS) {
  $test = "$c\$firstRel"
  $out = & $sz l $ZIP -i!"$test" 2>&1 | Out-String
  if ($out -match [regex]::Escape([IO.Path]::GetFileName($firstRel))) { $prefix = $c; break }
}
if (-not $prefix) {
  Write-Host "[에러] 아카이브 내부 prefix 자동탐지 실패. '$sz l `"$ZIP`" | more' 로 실제 경로 확인 후 CANDS 수정." -ForegroundColor Red
  exit 1
}
Write-Host "탐지된 prefix: $prefix"
$noBom = New-Object System.Text.UTF8Encoding($false)   # BOM 없이 (Python utf-8 호환)
[System.IO.File]::WriteAllText($PREFIXOUT, $prefix, $noBom)

# ----- prefix 붙인 추출목록 작성 -----
$full = Join-Path $WORK "extract_list_full.txt"
$lines = Get-Content -LiteralPath $LIST | ForEach-Object { "$prefix\$_" }
[System.IO.File]::WriteAllLines($full, $lines, $noBom)
$cnt = (Get-Content -LiteralPath $full | Measure-Object -Line).Lines
Write-Host "추출 대상 $cnt 개 -> $STAGING"

# ----- 선택 추출 (x = 읽기/추출, 아카이브 불변) -----
& $sz x $ZIP "-o$STAGING" "@$full" -y -bsp1
if ($LASTEXITCODE -ne 0) { Write-Host "[경고] 7z 종료코드 $LASTEXITCODE" -ForegroundColor Yellow }
Write-Host "[완료] 추출 끝. 다음: python 04_convert_to_yolo.py"
