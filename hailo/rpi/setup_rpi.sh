#!/usr/bin/env bash
# Raspberry Pi 5 + Hailo-8L AI Kit 셋업 (한 번 실행). 화재/연기 HEF 실행 환경.
# 사용: bash setup_rpi.sh
set -e
HEF_HRT_VER="4.24.0"   # 우리 HEF가 컴파일된 HailoRT 버전 (Pi 런타임과 맞아야 함)

echo "=== 1) 시스템 업데이트 + 의존 패키지 ==="
sudo apt update && sudo apt full-upgrade -y
sudo apt install -y dkms python3-opencv python3-numpy

echo "=== 2) PCIe Gen3 (대역폭 확보) ==="
CFG=/boot/firmware/config.txt
if [ -f "$CFG" ] && ! grep -q "pciex1_gen=3" "$CFG"; then
  echo "dtparam=pciex1_gen=3" | sudo tee -a "$CFG" >/dev/null
  echo "  config.txt에 추가됨 (재부팅 후 적용)"
fi

echo "=== 3) HailoRT 설치 (표준: hailo-all = 드라이버+펌웨어+HailoRT+pyhailort) ==="
sudo apt install -y hailo-all

echo "=== 4) 확인 ==="
echo "-- 장치 (Device Architecture = HAILO8L 여야 함) --"
hailortcli fw-control identify || true
echo "-- 설치된 HailoRT 버전 --"
hailortcli --version || true

echo ""
echo "================= ⚠ 버전 매칭 점검 (중요) ================="
echo " 우리 HEF는 HailoRT ${HEF_HRT_VER} 로 컴파일됨."
echo " 위 'hailortcli --version' 이 ${HEF_HRT_VER} 보다 낮으면 HEF 로드 실패:"
echo "   [HailoRT] Unsupported hef version / HAILO_INVALID_HEF(26)"
echo " 해결 (둘 중 하나):"
echo "   (A) Developer Zone에서 ${HEF_HRT_VER} arm64 .deb(+동일버전 드라이버) 받아 맞춤"
echo "   (B) [권장] x86에서 HEF를 Pi 버전에 맞춰 재컴파일 (Dockerfile의 MZ_TAG/DFC 조정)"
echo "       → Pi는 'apt install hailo-all' 그대로 두는 게 가장 단순"
echo "==========================================================="
echo ""
echo "다음: sudo reboot  →  python3 infer.py --hef yolov8s.hef --source test.jpg"
