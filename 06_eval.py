# -*- coding: utf-8 -*-
"""
06. 평가지표 (2클래스 검출기로도 전부 산출 가능).

(A) 검출 지표  : mAP@50, mAP@50-95, 클래스별 AP/P/R  (YOLO val)
(B) 경보 지표  : image-level 오탐율(FAR)·precision·recall·F1·confusion
                 - 정상(빈 라벨) 이미지 = 네거티브 테스트셋
- 출력: eval_report.md  (val 이미지/라벨만 읽음, 원본 불변)
"""
import sys, argparse
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from ultralytics import YOLO
import config as C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25, help="경보 판정 임계값")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = YOLO(args.weights)

    # ---- (A) 검출 지표 ----
    m = model.val(data=str(C.DATA_YAML), imgsz=args.imgsz, device=args.device, conf=0.001)
    det = {
        "mAP50": float(m.box.map50), "mAP50-95": float(m.box.map),
        "per_class": {C.CLASS_NAMES[i]: {
            "AP50": float(m.box.ap50[i]), "P": float(m.box.p[i]), "R": float(m.box.r[i])
        } for i in range(len(C.CLASS_NAMES))},
    }

    # ---- (B) image-level 경보 지표 ----
    val_img = C.DATASET / "images" / "val"
    val_lbl = C.DATASET / "labels" / "val"
    imgs = sorted(val_img.glob("*.jpg"))
    TP = FP = FN = TN = 0
    for im in imgs:
        lbl = val_lbl / (im.stem + ".txt")
        gt_pos = lbl.exists() and lbl.read_text(encoding="utf-8").strip() != ""
        res = model.predict(str(im), imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        pred_pos = len(res.boxes) > 0
        if gt_pos and pred_pos: TP += 1
        elif gt_pos and not pred_pos: FN += 1
        elif (not gt_pos) and pred_pos: FP += 1
        else: TN += 1

    prec = TP / (TP + FP) if TP + FP else 0.0
    rec  = TP / (TP + FN) if TP + FN else 0.0
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    far  = FP / (FP + TN) if FP + TN else 0.0   # 정상 이미지 중 오탐 비율

    # ---- 리포트 ----
    rp = C.WORK / "eval_report.md"
    L = []
    L.append("# 화재/연기 검출 평가 리포트\n")
    L.append(f"weights: `{args.weights}`  |  conf(경보): {args.conf}\n")
    L.append("## (A) 검출 지표\n")
    L.append(f"- mAP@50: **{det['mAP50']:.3f}**, mAP@50-95: **{det['mAP50-95']:.3f}**\n")
    L.append("| 클래스 | AP50 | P | R |\n|---|---|---|---|")
    for c, v in det["per_class"].items():
        L.append(f"| {c} | {v['AP50']:.3f} | {v['P']:.3f} | {v['R']:.3f} |")
    L.append("\n## (B) 경보(image-level) 지표\n")
    L.append(f"- 검증 이미지 {len(imgs):,}장 (정상=네거티브 포함)\n")
    L.append("| | 예측 양성 | 예측 음성 |\n|---|---|---|")
    L.append(f"| 실제 양성(화재/연기) | TP={TP} | FN={FN} |")
    L.append(f"| 실제 음성(정상) | FP={FP} | TN={TN} |")
    L.append(f"\n- Precision **{prec:.3f}** / Recall **{rec:.3f}** / F1 **{f1:.3f}**")
    L.append(f"- 오탐율(FAR, 정상중 오경보) **{far:.3f}**")
    rp.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[완료] 리포트: {rp}")

if __name__ == "__main__":
    main()
