# 라즈베리파이 실행 (Hailo-8L HEF 추론)

x86에서 컴파일한 `yolov8s.hef`(화재/연기)를 **Raspberry Pi 5 + Hailo-8L AI Kit**에서 실행.
컴파일(x86)과 실행(Pi)은 다른 기계 — Pi엔 **DFC 불필요, HailoRT만** 있으면 됨.

## 준비물
- Raspberry Pi 5 + Hailo-8L AI Kit(M.2), RPi OS (Bookworm/Trixie)
- 이 폴더 파일 + `yolov8s.hef` 를 Pi로 복사 (scp 등)

## 실행 순서 (Pi에서)
```bash
# 1) 환경 설치 (한 번)
bash setup_rpi.sh
sudo reboot

# 2) 장치/버전 확인
hailortcli fw-control identify     # Device Architecture: HAILO8L 확인
hailortcli --version               # 아래 '버전 매칭' 참고

# 3) 추론
python3 infer.py --hef yolov8s.hef --source test.jpg              # 이미지
python3 infer.py --hef yolov8s.hef --source clip.mp4 --save out.mp4
python3 infer.py --hef yolov8s.hef --source 0 --show             # USB 카메라
```

## ⚠ 버전 매칭 (배포 실패 1순위)
| 구성 | 값 |
|---|---|
| HEF 컴파일 | DFC 3.34 / HailoRT **4.24** |
| Pi 런타임 | `hailortcli --version` 로 확인 |

- **Pi 버전 ≥ 4.24** → 우리 HEF 그대로 실행됨 ✓
- **Pi 버전 < 4.24** (예: apt 기본 4.23) → HEF 로드 실패(`HAILO_INVALID_HEF`)
  - (A) Developer Zone에서 4.24 arm64 .deb+드라이버 받아 Pi를 4.24로
  - (B) **[권장]** x86에서 HEF를 Pi 버전에 맞춰 재컴파일 (Dockerfile `MZ_TAG`/DFC 조정) → Pi는 `apt install hailo-all` 그대로
- 아키텍처: HEF=hailo8l, AI Kit=Hailo-8L ✓ (AI HAT+는 Hailo-8 → 재컴파일 필요)

## 동작 핵심 (infer.py)
- 입력: 640 letterbox + **uint8 RGB** (정규화는 HEF 내장 → /255 안 함)
- 출력: on-chip NMS라 **클래스별 detection 리스트** = [ymin,xmin,ymax,xmax,score] 정규화 → 코드가 픽셀 변환
- 성능: RPi5+8L, yolov8s@640 ≈ ~80fps → 실시간 CCTV 여유

## 트러블슈팅
| 증상 | 해결 |
|---|---|
| `HAILO_INVALID_HEF` | 버전 매칭(위) |
| 박스가 엉뚱/없음 | infer.py letterbox의 `cvtColor(BGR2RGB)` 줄 제거해 BGR로 시도 |
| `InputVStreamParams.make` 에러 | `make_from_network_group(ng, quantized=False, format_type=...)` 로 교체 |
| `Driver version != library version` | 커널 업데이트 후 흔함 → hailo-all 재설치로 드라이버/라이브러리 버전 일치 |
| 느림 | PCIe Gen3 적용됐는지 확인(setup이 config.txt에 추가) |

## production 모델 교체
현재 `yolov8s.hef`는 1280학습→640 export 차선책. **8s_640_hn**(640 재학습+hardneg) 완료 후 그 HEF로 교체 예정 — 같은 infer.py로 그대로 실행.
