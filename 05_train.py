# -*- coding: utf-8 -*-
"""
05. YOLOv8m 학습.

- 사용자 지정: YOLOv8 M 모델(yolov8m.pt).
- 입력 1920x1080 영상프레임 -> imgsz 1280 (불꽃/연기가 작거나 큰 경우 대비).
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
from ultralytics import YOLO
import config as C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)   # time 상한 쓰면 사실상 캡
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=-1)      # -1 = 자동(VRAM 맞춤)
    ap.add_argument("--device", default="0")
    ap.add_argument("--weights", default="yolov8m.pt")    # YOLOv8 M
    ap.add_argument("--time", type=float, default=None,   # 학습 시간 상한(시간). 지정시 시간 끝나면 best.pt 저장 후 종료
                    help="최대 학습 시간(시간 단위). 오버나이트용.")
    args = ap.parse_args()

    kw = dict(
        data=str(C.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(C.RUNS_DIR),
        name="yolov8m_fire_smoke",
        # 화재/연기 도메인: 색상 왜곡은 약하게(불꽃 색 보존), 좌우반전만 허용
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        fliplr=0.5, flipud=0.0,            # 상하반전 금지(불꽃은 위로 탐)
        mosaic=1.0, close_mosaic=10,       # 막판 10ep 모자이크 off -> 박스 정밀도↑
        # multi_scale=True 는 DDP+imgsz1280에서 Size를 128로 뽑아 interpolate 크래시 -> 비활성
        box=10.0,                          # box loss 가중(기본7.5) -> mAP@50-95(타이트 박스)↑
        patience=30, cos_lr=True,
    )
    if args.time:
        kw["time"] = args.time   # 시간 상한(에포크보다 우선). 끝나면 자동 종료+저장
    model = YOLO(args.weights)
    model.train(**kw)
    print("[완료] 학습 끝. 가중치: runs/yolov8m_fire_smoke/weights/best.pt")
    print("다음: python 06_eval.py --weights <best.pt 경로>")

if __name__ == "__main__":
    main()
