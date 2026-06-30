# 화재·연기 검출 YOLO 학습 파이프라인

CCTV 영상에서 **불꽃·연기를 검출**하는 YOLO 모델을 학습하는 파이프라인. 800GB 분할압축 원천에서 필요한 프레임만 선택추출해 6종 모델을 동일 조건으로 비교하고, 배포 환경(CPU/GPU)에 맞는 모델을 추론용 ONNX로 변환한다.

**배포 환경별 추천 모델** — 벤치마크 종합 결과

| 환경 | 추천 모델 | mAP@50-95 | 크기(.pt) | CPU 640px |
|---|---|---|---|---|
| CPU 균형 (현재 백엔드) | **YOLOv8s** | 0.622 | 23MB | 37fps |
| CPU 엣지·실시간 | **YOLOv8n** | 0.612 | 6MB | 86fps |
| GPU 최고정확도 | **YOLOv8m** | 0.638 | 52MB | 15fps |

---

## 파이프라인 개요

`01 라벨 색인 → 02 서브샘플·분할 → 03 서브샘플 추출 → 04 YOLO 입력 형태로 변환 → 05 YOLO 학습·HPO → 06 YOLO 비교 → 07 ONNX형식으로 변환`

| # | 단계 | 스크립트 | 하는 일 | 산출 |
|---|---|---|---|---|
| 01 | 라벨 색인 | `data_prep/01_build_manifest.py` | 원본 데이터 804G의 압축을 풀지 않고 JSON 라벨만 스캔하여 메타데이터화 | `manifest.csv` |
| 02 | 서브샘플·분할 | `data_prep/02_subsample_split.py` | 프레임 서브 샘플링을 통해 과적합 방지, train/val에 같은 영상을 넣지 않도록 분할(데이터 누수 방지)  | `selected.csv` |
| 03 | 이미지 추출 | `data_prep/03_extract_images.ps1` | 02에서 샘플링된 이미지만 선별하여 압축해제 | *.jpg |
| 04 | YOLO 변환 | `data_prep/04_convert_to_yolo.py` | COCO JSON → YOLO txt + 이미지 복사 | `dataset_24k` |
| 05 | 학습·HPO | `model.py` | 여러 버전 학습 + Optuna 튜닝 | `best.pt` |
| 06 | 비교·선정 | `model.py` | mAP·속도·크기 → 배포환경별 최적 | `model_comparison.csv` |
| 07 | ONNX 배포 | `model.py` | CPU 추론용 dynamic ONNX | `best.onnx` |

---

## 프로젝트 구조

```
yolo_fire_smoke/
├── config.py               # 중앙 설정 (경로·클래스·파라미터, work/ 경로 앵커)
├── model.py                # 모델 학습·평가·비교·ONNX 변환
├── requirements.txt
├── data_prep/              # 데이터 파이프라인 (순차 실행)
│   ├── 01_build_manifest.py   # 라벨 색인 + 통계 (zip 미접근)
│   ├── 02_subsample_split.py  # 클립당 6장 서브샘플 + 클립단위 train/val 분할
│   ├── 03_extract_images.ps1  # 7z로 선택한 jpg만 추출 (분할압축, 원본 불변)
│   ├── 04_convert_to_yolo.py  # COCO JSON -> YOLO txt + 이미지 복사
│   ├── 05_hn_mine.py          # hard-negative 마이닝 (candidates/score/inject)
│   └── 06_hn_extract.ps1      # 마이닝 후보 jpg 선택추출 (원본 불변)
├── api/                    # 추론 백엔드 (FastAPI, 로컬 데모)
│   ├── app.py              #   라우팅 (/detect, /detect/video, /health)
│   ├── detector.py         #   추론 래퍼 (이미지·영상)
│   ├── static/index.html   #   웹 데모 페이지
│   └── README.md
├── server/                 # GPU 서버 학습 스크립트
│   ├── setup.sh            #   venv + torch(cu128) + ultralytics
│   ├── train.sh            #   학습 (분리 실행)
│   ├── eval.sh             #   best.pt 평가
│   ├── data.yaml
│   └── README.md
├── hailo/                  # Hailo-8L NPU 배포 (코드만 git, 바이너리 gitignore)
│   ├── compile_to_hef.py   #   [x86] ONNX → HEF 컴파일 (hailomz 래핑)
│   ├── Dockerfile          #   [x86] DFC+HailoRT 컴파일 환경
│   ├── README.md           #   컴파일 가이드
│   └── rpi/                # [라즈베리파이] HEF 실행
│       ├── setup_rpi.sh    #     HailoRT 설치 + 장치 확인
│       ├── infer.py        #     HEF 추론 (이미지·영상·카메라)
│       └── README.md       #     Pi 실행 가이드
├── work/                   # 산출물 (gitignore): .venv, runs/(가중치), hn/(마이닝), manifest.csv, selected.csv
└── *.pt                    # 사전학습 체크포인트 (yolov8m/8n 등)
```

---

## 1. 데이터셋

화재 현장 CCTV 기반 거대한 원천데이터에서 **2클래스(fire·smoke)** 데이터셋을 추려 학습   
정상(none)은 박스 없는 **빈 라벨 네거티브**로 넣어 오탐을 억제하며, 이 규칙은 파이프라인 전체에 일관 적용

### 1.1 학습 데이터 (실사용)
#### 데이터셋 핵심 용어 정의
| 용어 | 정의 | 본 프로젝트 내 기준 |
| :--- | :--- | :--- |
| **클래스** | YOLO가 탐지할 객체 종류(정답) | **불꽃(0), 연기(1)**, 정상(None, 배경) |
| **클립** | 사건이 녹화된 연속 영상 단위 | 1 클립 = **12초 분량** (360프레임) |
| **프레임** | 영상의 순간 이미지 | 모델의 실제 입력값 (`.jpg`) |
| **라벨** | 해당 프레임 내 객체의 위치/종류 정보 | JSON(원천) → 정규화된 **TXT(학습용)** |

#### 학습 데이터 (Training Spec)

| 항목 | 상세 내용 |
| :--- | :--- |
| **학습 클래스** | `fire` (0), `smoke` (1) |
| **데이터 규모** | 총 23,958장 (Train 19,188 / Val 4,770) |
| **객체(BBox) 수** | 총 26,056개 |
| **입력 해상도** | 1280px |
| **네거티브 처리** | 화재/연기가 없는 '정상' 프레임은 빈 라벨(Empty txt)로 처리하여 배경 네거티브로 학습 |

```
LLM/DATA/datasets/fire_smoke_yolo/
├── dataset_24k/                  # 기본 학습셋 (train 19,188 / val 4,770)
│   ├── images/{train,val}/*.jpg
│   └── labels/{train,val}/*.txt  # 불꽃·연기 = YOLO 박스, 정상 = 빈 txt
├── dataset_48k/                  # 더 촘촘한 서브샘플 실험셋 (보관)
├── data.yaml                     # → dataset_24k (기본)
└── data_48k.yaml                 # → dataset_48k
```

### 1.2 원천 데이터

- `화재현장 주요객체`(소화기·소화전·표지판 등 15종)는 현장 사물 검출용이라 화재/연기 학습에서 제외
- **주의** : 이미지 경로는 `화재현상`(공백 없음), 라벨 경로는 `화재 현상`(공백 있음) — 표기가 다르니 주의.

```
원천데이터/                        # 이미지 (분할압축, 읽기전용)
  └ TS.zip + TS.z01~z08 ≈ 804GB
      └ (압축 내부) 화재현상/이미지/{클래스}/{장소}/{클립}/JPG/*.jpg
라벨링데이터/TL/                   # 라벨 (비압축, 직접 접근)
  ├ 화재 현상/이미지/{클래스}/{장소}/{클립}/JSON/*.json   ← 학습용 라벨
  └ 화재현장 주요객체/*.json       # 소화기 등 15종 (약 38.5만개, 학습 미사용)
```


**원천 규모**


| 항목 | 값 |
|---|---|
| 이미지 | 1920×1080 jpg, 30fps |
| 클립 | 360프레임(=12초) 단위 → 프레임 간 거의 중복 |
| 라벨 | COCO 스타일 JSON, `bbox = [x, y, w, h]` 픽셀 절대값 |

| 클래스 | 코드(id) | 프레임 | 클립 |
|---|---|---|---|
| 불꽃 fire | fl(1) | 610,920 | 1,697 |
| 연기 smoke | sm(2) | 610,920 | 1,697 |
| 정상 none | none(3) | 305,280 | 848 |
| **합계** | | **1,527,120** | **4,242** |


**장소 × 클래스 (클립 수)** — 8개 장소 유형, 장소별 비중 상이

| 장소 | 불꽃 | 연기 | 정상 |
|---|---|---|---|
| 일반주택_공동주택 | 434 | 267 | 72 |
| 공장_창고_작업장 | 277 | 172 | 64 |
| 시장_상점 | 256 | 51 | 180 |
| 노유자_숙박_의료시설 | 177 | 215 | 48 |
| 음식점_노래방_주점 | 160 | 389 | 37 |
| 교육연구시설_업무시설 | 150 | 188 | 126 |
| 종교_운동 | 131 | 172 | 129 |
| 차량_철도_선박_항공기 | 112 | 243 | 192 |

**파일명 규칙** — `{클립}_{클래스}_{장소코드}_{프레임}.jpg` (예: `0087_FL_FWW_00001`)

- 클래스: `FL` 불꽃 · `SM` 연기 · `NONE` 정상
- 장소코드: FWW 공장 · MS 시장 · ERBF 교육 · VTSP 차량 · OLMF 의료 · GAH 주택 · RE 종교 · ENB 음식점

---

## 2. 데이터 준비 — 핵심 설계

스크립트 단계는 상단 *파이프라인 개요*(`data_prep/` 01~04)를 참고. 핵심 결정은 다음 셋이다.

- **클립 단위 분할** — 같은 클립의 프레임이 train/val에 섞이면 데이터 누수로 성능이 거짓 상승한다. 프레임이 아니라 **클립 단위**로 split.
- **선택추출** — 800GB·30fps라 프레임이 거의 중복. 압축을 다 풀지 않고 라벨 색인으로 **필요한 jpg만** 7z 추출 (원본 불변).
- **2클래스 + 네거티브** — fire/smoke만 박스로 학습하고, 정상은 빈 라벨로 넣어 오탐을 억제.

결과: **23,958장** (train 19,188 / val 4,770, 박스 26,056).

---

## 3. 평가지표

- **검출**: mAP@50, mAP@50-95, 클래스별 AP/P/R
- **경보**: 정상 이미지를 네거티브로 써서 오탐율(FAR), image-level P/R/F1
- 화재 특성상 **recall(놓침 최소) 우선**, conf 임계값으로 FAR과 trade-off 튜닝

---

## 4. 모델 벤치마크 (6종)

6종을 동일 조건(화재/연기 24,000장 · 1280px · 100 epoch · 옵티마이저 `auto`=MuSGD)으로 학습 후 정확도·속도·크기를 비교. 측정은 **GPU는 .pt(PyTorch), CPU는 ONNX**(실제 배포 경로).

**결론**: 모든 체급에서 YOLOv8 > YOLOv5(u). CPU는 모델 선택이 속도를 좌우 → 균형은 **YOLOv8s**, 엣지·실시간은 **YOLOv8n**, GPU 최고정확도는 **YOLOv8m** (상단 추천 표).

### 4.1 정확도·크기

onnx는 fp32라 .pt의 약 2배 (fp16 변환 시 절반).

| 모델 | mAP@50 | mAP@50-95 | 파라미터 | .pt | onnx |
|---|---|---|---|---|---|
| YOLOv8m (기존) | 0.905 | **0.638** | 25.8M | 52MB | 104MB |
| YOLOv5m | 0.904 | 0.627 | 25.1M | 51MB | 101MB |
| YOLOv8s | 0.895 | 0.622 | 11.1M | 23MB | 45MB |
| YOLOv8n | 0.894 | 0.612 | 3.0M | 6MB | 12MB |
| YOLOv5s | 0.880 | 0.603 | 9.1M | 19MB | 37MB |
| YOLOv5n | 0.879 | 0.598 | 2.5M | 5MB | 10MB |

### 4.2 속도 · 실시간 처리율

프레임당 추론 지연(ms·낮을수록 빠름)과 처리율(fps)을 함께 표기. 라이브 30fps 기준 판정: **✓** 30↑ / **△** 20~29 / **✗** <20. **GPU(.pt)는 전 모델 235fps↑ → 전부 ✓.**

| 모델 | GPU .pt | CPU 640px | CPU 1280px |
|---|---|---|---|
| YOLOv8m | 4.3ms | 65ms · 15fps ✗ | 280ms · 3.6fps ✗ |
| YOLOv5m | 3.6ms | 50ms · 20fps ✗ | 204ms · 4.9fps ✗ |
| YOLOv8s | 1.8ms | 27ms · 37fps ✓ | 117ms · 8.5fps ✗ |
| YOLOv8n | 0.8ms | 12ms · 86fps ✓ | 45ms · 22fps △ |
| YOLOv5s | 1.7ms | 22ms · 46fps ✓ | 86ms · 12fps ✗ |
| YOLOv5n | 0.7ms | 9ms · 109fps ✓ | 34ms · 30fps ✓ |

### 4.3 영상 분석 시간 (참고)

10초 클립을 frame_stride=15로 약 16장만 샘플 분석 (CPU onnx). 4.2의 프레임 지연에서 파생되며, 실제는 디코드·NMS로 +20~30%.

| 모델 | 640px | 1280px |
|---|---|---|
| YOLOv8m | 1.0s | 4.5s |
| YOLOv5m | 0.8s | 3.3s |
| YOLOv8s | 0.4s | 1.9s |
| YOLOv8n | 0.2s | 0.7s |
| YOLOv5s | 0.4s | 1.4s |
| YOLOv5n | 0.1s | 0.5s |

### 4.4 배포 가이드

- **CPU vs GPU** — CPU는 1280px 기준 모델 간 최대 6배 차이라 모델 선택이 결정적. GPU는 전 모델 실시간 여유.
- **업로드 분석(이미지·영상)** — 프레임 샘플링이라 적은 프레임만 추론 → CPU로도 충분.
- **실시간(라이브)** — CPU 30fps는 640px + 소형/nano만, 1280px 라이브는 nano만 가능. GPU면 전부 가능.

---

## 5. 하이퍼파라미터 튜닝 (HPO)

**결론**: SGD로 재튜닝해도 `auto`(MuSGD) baseline을 확실히 넘지 못함 → **배포본은 MuSGD baseline 유지**. (8s만 자기 baseline을 소폭 초과, 8n·8m은 동일~미달 → MuSGD baseline이 이미 강한 선택이었음.)

CPU 후보 **YOLOv8s·8n** + baseline **8m**(MuSGD vs SGD 공정 비교용)을 대상으로 Optuna proxy 튜닝 후 채택값으로 본학습 1회.

### 5.1 방식

- **도구**: Optuna (TPESampler + MedianPruner), 모델별 study 분리 (8s/8n 동시 탐색)
- **proxy 탐색**: 데이터 15% · 30 epoch · 640px로 빠르게 많은 조합 시도. 매 epoch fitness 보고 → 나쁜 trial 조기중단
- **탐색 대상**: lr0/lrf/momentum/weight_decay/warmup/optimizer(SGD·AdamW)/box/cls/dfl/증강(hsv·degrees·translate·scale·shear·mosaic·mixup)
- **점수**: `fitness = 0.1·mAP50 + 0.9·mAP50-95` (ultralytics 기본)

### 5.2 탐색 결과

fitness 절대값이 낮은 건 proxy(15%·30ep·640px)라서이며 본학습(전체·1280px·100ep)에서 회복. 공통 신호는 **box 가중치 ↑ (7.5 → 10~11)**, cls·dfl 소폭 ↑, optimizer는 **SGD**.

| 모델 | 총 trial | 완료/가지치기 | best fitness(proxy) | 비고 |
|---|---|---|---|---|
| YOLOv8s | 15 | 9 / 6 | 0.362 | 기본 조합 대비 개선 |
| YOLOv8n | 14 | 6 / 8 | 0.314 | 개선폭 미미(첫 조합이 최고) |

### 5.3 채택 파라미터 (본학습 적용)

box/cls/dfl/증강은 전이성이 좋아 HPO값을 그대로 채택. 단 **lr0/lrf만 0.01로 오버라이드** — proxy(30ep)가 lr을 낮게 편향시켜(HPO값 ~0.0008) 100ep 본학습엔 부적합하기 때문.

| 파라미터 | YOLOv8s | YOLOv8n | YOLOv8m | 출처 |
|---|---|---|---|---|
| box | 10.26 | 11.06 | 11.06 | HPO |
| cls | 0.78 | 0.72 | 0.72 | HPO |
| dfl | 1.74 | 1.71 | 1.71 | HPO |
| mixup | 0.195 | 0.086 | 0.086 | HPO |
| optimizer | SGD | SGD | SGD | HPO |
| **lr0 / lrf** | **0.01 / 0.01** | **0.01 / 0.01** | **0.01 / 0.01** | **오버라이드** |

### 5.4 본학습 결과 (SGD 고정)

baseline은 `auto`(MuSGD), 튜닝 재학습은 `SGD` 고정으로 옵티마이저를 통일해 비교. 설정: 1280px · 전체데이터 · 100 epoch · patience=30 · cos_lr · batch=32.

| 모델 | 옵티마이저 | mAP50-95 | baseline 대비 |
|---|---|---|---|
| YOLOv8s 튜닝 | SGD | 0.629 (ep99) | baseline 0.622 **초과** |
| YOLOv8n 튜닝 | SGD | 0.610 (ep90) | baseline 0.612 동일 |
| YOLOv8m 튜닝 | SGD | 0.631 (ep91, best 61) | baseline 0.638 **미달** |

클래스별(튜닝 8m): 불꽃 0.684 / 연기 0.577 → **연기 박스 정밀도가 천장**. 연기는 비정형이라 엄격 IoU 정합에 한계.

---

## 6. 산출물

| 위치 | 내용 |
|---|---|
| `LLM/DATA/datasets/fire_smoke_yolo/dataset_24k` | 학습 데이터셋 24k (train 19,188 / val 4,770) |
| `work/manifest.csv` · `work/selected.csv` | 라벨 색인 / 서브샘플·split 결과 |
| `work/runs/<model>/weights/best.pt` (+ `best.onnx`) | 학습 가중치 + 배포용 ONNX |
| `model_comparison.csv` | 모델 비교표 |
