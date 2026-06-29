# ONNX → HEF 컴파일 (Hailo-8L)

`best.onnx` 를 Hailo-8L NPU용 `.hef` 로 변환. `hailomz compile` CLI를 파이썬으로 래핑.

## 산출물
- `compile_to_hef.py` : `hailomz compile` 래퍼 (로그 캡처·에러 힌트)
- `Dockerfile` : DFC + model_zoo 격리 환경
- `prep_inputs.py` : 컴파일 입력 스테이징 — onnx(저장소)+calib(val) → hailo/ (둘 다 gitignore)

> best.onnx / calib_data / *.whl / *.hef 는 레포에 영구 보관 안 함. onnx는 `work/onnx_final` 저장소, calib는 LLM/DATA val에서 매번 스테이징.

## ⚠ 사전 준비 (이거 안 맞으면 100% 실패)

### 1) 정적 640 ONNX export (저장소 자동 보관)
Hailo는 동적(dynamic) shape 불가. 정적 640 export는 저장소(`work/onnx_final`)에 자동 저장됨:
```
python model.py export <best.pt> --static     # -> work/onnx_final/<run>_static.onnx
```
prep_inputs.py 가 이걸 best.onnx 로 끌어옴. (1280은 model_zoo yaml `input_shape` 수정 + `--yaml`)

### 2) DFC whl 다운로드 (공개 pip 불가)
Hailo Developer Zone(로그인) → Software Downloads → **Dataflow Compiler 3.x** whl 받아서
이 폴더(Dockerfile 옆)에 둘 것:
```
hailo_dataflow_compiler-3.XX.X-py3-none-linux_x86_64.whl
```

### 3) 버전 조합 (Hailo-8L 핵심)
| 구성 | 값 | 이유 |
|---|---|---|
| DFC | **3.34.0** (보유) | hailo8l은 DFC 3.x 라인 |
| HailoRT | **4.24.0** (보유) | libhailort 임포트용 |
| model_zoo | **v2.19.0** | setup.py가 DFC 3.34.0 핀(정확 일치). master(5.x)는 hailo8l 드롭 |
| hw-arch | **hailo8l** | hailo8 HEF와 호환 안 됨 |

DFC 버전이 다르면 MZ 태그도 맞출 것: `3.32->v2.16` · `3.33.1->v2.18` · `3.34->v2.19`

## 실행

### 1. 빌드 (DFC whl 이 같은 폴더에 있어야 함)
```powershell
docker build -t hailo8l-compiler .
# (기본 MZ_TAG=v2.19.0 = DFC 3.34.0 매칭. 다른 DFC면 --build-arg MZ_TAG=... 로 변경)
```

### 2. 컴파일 입력 준비 (onnx + calib 스테이징)
best.onnx·calib_data는 레포에 보관 안 함. 저장소/원본에서 재생성:
```
python prep_inputs.py           # onnx_final/yolov8s_640_hn_static.onnx -> best.onnx + val -> calib_data/
```

### 3. 컴파일 실행 (best.onnx + calib_data/ 마운트)

PowerShell (Windows):
```powershell
docker run --rm -it `
  -v ${PWD}/best.onnx:/workspace/best.onnx `
  -v ${PWD}/calib_data:/workspace/calib_data `
  -v ${PWD}/hef_output:/workspace/hef_output `
  hailo8l-compiler `
  python3.10 compile_to_hef.py --onnx best.onnx --calib-path calib_data --classes 2 --hw-arch hailo8l
```

bash (WSL2/Linux):
```bash
docker run --rm -it \
  -v $(pwd)/best.onnx:/workspace/best.onnx \
  -v $(pwd)/calib_data:/workspace/calib_data \
  -v $(pwd)/hef_output:/workspace/hef_output \
  hailo8l-compiler \
  python3.10 compile_to_hef.py --onnx best.onnx --calib-path calib_data --classes 2 --hw-arch hailo8l
```

결과: `hef_output/yolov8s.hef` (+ `compile_log.txt`)

## 트러블슈팅
| 증상 | 원인 / 해결 |
|---|---|
| `invalid choice: 'hailo8l'` | model_zoo가 master(5.x). v2.x 태그로 재빌드 |
| `libhailort.so... cannot open` | HailoRT/DFC/MZ 버전 불일치. 호환 조합으로 통일 |
| 파싱(parse) 실패 / end_node | 커스텀 export 노드 불일치 → `--yaml` + `end_node_names` 지정 |
| 정확도 급락 | onnx 해상도와 yaml `input_shape`/calib 해상도 불일치 |

## 직접 실행 (도커 없이, DFC+MZ가 이미 깔린 venv)
```
python3.10 compile_to_hef.py --onnx best.onnx --calib-path calib_data --classes 2
```
