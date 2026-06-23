# 화재/연기 YOLO 학습 파이프라인

## 1. 데이터 파악 결과

| 항목 | 내용 |
|---|---|
| 정체 | AIHub 계열 화재 발생 영상 (불꽃·연기·정상 + 화재현장 객체) |
| 원천 압축 | `TS.z01~z08` 각 100GiB + `TS.zip` 3.9GB = **약 804GB 분할압축** |
| 이미 풀린 분량 | `TS\` 50.7GB, jpg 143,273장 (불꽃만, 8장소 중 2장소 = 부분추출) |
| 라벨(풀려있음) | JSON 약 191만개 (zip 안 열어도 접근 가능) |
| 이미지 스펙 | 1920×1080, 30fps, **클립당 360프레임(=12초)** |
| 압축도구 | 7-Zip 없음(설치 필요), tar는 분할zip 불가 |

라벨 분포 (화재현상 = 학습 타깃):

| 클래스 | 코드 | JSON | 클립(÷360) |
|---|---|---|---|
| 불꽃 | fl | 610,920 | 1,697 |
| 연기 | sm | 610,920 | 1,697 |
| 정상 | none | 305,280 | 848 |
| 합 | | 1,527,120 | 4,242 |

별도 `화재현장 주요객체` 약 38.5만장(소화기/소화전/표지판 등 15객체)은 화재 자체가 아니라 현장 사물 검출용 -> 이 파이프라인 미사용.

라벨 포맷: COCO 스타일 JSON. `bbox = [x_min, y_min, w, h]` 픽셀절대. categories_id 1=fl, 2=sm, 3=none.

## 2. 핵심 방침

- **압축 다 풀지 않는다.** 800GB + 30fps 중복 프레임. 라벨이 이미 색인이므로 zip은 필요한 jpg만 선택추출.
- **원본 읽기 전용.** 모든 단계가 원본을 읽기만 하고, 출력은 `D:\mro_fire_yolo`로 분리. 원본 폴더·압축파일에 쓰기/이동/삭제 없음.
- **2클래스(fire, smoke).** 정상은 빈 라벨 네거티브(오탐 억제). 평가지표는 06이 별도 산출.
- **클립 단위 분할.** 같은 클립 프레임이 train/val 양쪽에 들어가면 누수 -> 성능 거짓 부풀림.

## 3. 실행 순서

```
pip install -r requirements.txt
winget install --id 7zip.7zip -e        # 분할zip 추출용(1회)

python 01_build_manifest.py             # 라벨 색인 + 통계 (zip 미접근)
python 02_subsample_split.py            # 클립당 8장 서브샘플 + 클립단위 train/val
powershell -File 03_extract_images.ps1  # 선택한 jpg만 7z로 추출(원본 불변)
python 04_convert_to_yolo.py            # JSON->YOLO txt + 이미지 복사
python 05_train.py                      # YOLOv8m 학습 (imgsz 1280)
python 06_eval.py --weights D:\mro_fire_yolo\runs\yolov8m_fire_smoke\weights\best.pt
```

빠른 프로토타입(추출 생략): 03 건너뛰고 04 실행하면 이미 풀린 `TS\`(불꽃)만으로 변환·학습 흐름 검증 가능. 단 연기 0장이라 본학습 전 03 필수.

## 4. 산출물

| 파일 | 내용 |
|---|---|
| `D:\mro_fire_yolo\manifest.csv` | 전체 라벨 색인(152만행) |
| `selected.csv` | 서브샘플 + split 결과 |
| `extract_list.txt` | 7z 선택추출 목록 |
| `dataset/` | YOLO 레이아웃(images/labels × train/val) |
| `data.yaml` | 학습 설정 |
| `runs/.../best.pt` | 학습 가중치 |
| `eval_report.md` | 검출 mAP + 경보 FAR 리포트 |

## 5. 평가지표 (2클래스로도 가능)

- 검출: mAP@50, mAP@50-95, 클래스별 AP/P/R
- 경보: 정상 이미지를 네거티브로 써서 오탐율(FAR), image-level P/R/F1, confusion
- 화재 특성상 recall(놓침 최소) 우선, conf 임계값으로 FAR과 trade-off 튜닝

## 6. 파라미터 (config.py)

- `FRAMES_PER_CLIP=8`, `NORMAL_RATIO=0.15`, `VAL_RATIO=0.20`
- 데이터 더 필요하면 `FRAMES_PER_CLIP` 상향, 장소 편향 보이면 층화 확인
