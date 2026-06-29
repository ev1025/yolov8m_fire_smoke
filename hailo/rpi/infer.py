#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi 5 + Hailo-8L : 화재/연기 HEF 추론 (HailoRT).

- on-chip NMS HEF → 출력은 '클래스별 detection 리스트' (HAILO_NMS_BY_CLASS)
  detections[class_id] = ndarray (N,5), 각 행 = [ymin, xmin, ymax, xmax, score], 정규화 0~1
- 입력은 uint8 RGB 640 letterbox (정규화는 HEF에 내장 → /255 하지 말 것)

사용:
  python3 infer.py --hef yolov8s.hef --source test.jpg              # 이미지 1장
  python3 infer.py --hef yolov8s.hef --source clip.mp4 --save out.mp4
  python3 infer.py --hef yolov8s.hef --source 0 --show             # USB 카메라
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
import cv2
import hailo_platform as hpf

NAMES = {0: "fire", 1: "smoke"}
COLORS = {0: (0, 0, 255), 1: (255, 60, 0)}      # BGR: fire=빨강, smoke=파랑
IMGSZ = 640


def letterbox(bgr, size=IMGSZ):
    """종횡비 유지 + gray(114) 패딩 → (size,size,3) RGB uint8 + 역변환 정보(r,dw,dh)."""
    h, w = bgr.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dh, dw = (size - nh) // 2, (size - nw) // 2
    canvas[dh:dh + nh, dw:dw + nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)        # HEF는 RGB 입력 (틀리면 이 줄 제거해 BGR 시도)
    return np.ascontiguousarray(rgb, np.uint8), r, dw, dh


def parse_nms(dets_per_class, conf, r, dw, dh, W, H):
    """클래스별 NMS 리스트 → [(cls, score, x1,y1,x2,y2)] 원본 픽셀."""
    out = []
    for cls_id, dets in enumerate(dets_per_class):
        if dets is None or len(dets) == 0:
            continue
        for d in dets:
            score = float(d[4])
            if score < conf:
                continue
            ymin, xmin, ymax, xmax = (float(v) * IMGSZ for v in d[:4])   # 정규화→640px
            x1 = max(0, min(W, (xmin - dw) / r)); y1 = max(0, min(H, (ymin - dh) / r))
            x2 = max(0, min(W, (xmax - dw) / r)); y2 = max(0, min(H, (ymax - dh) / r))
            out.append((cls_id, score, int(x1), int(y1), int(x2), int(y2)))
    return out


def draw(img, dets):
    for cls_id, score, x1, y1, x2, y2 in dets:
        c = COLORS.get(cls_id, (0, 255, 0))
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        cv2.putText(img, f"{NAMES.get(cls_id, cls_id)} {score:.2f}", (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img


def main():
    ap = argparse.ArgumentParser(description="Hailo-8L 화재/연기 HEF 추론")
    ap.add_argument("--hef", required=True, help="yolov8s.hef 경로")
    ap.add_argument("--source", required=True, help="이미지/영상 경로 또는 카메라번호(예 0)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--show", action="store_true", help="창에 표시(q 종료)")
    ap.add_argument("--save", default=None, help="결과 저장 경로(이미지/영상)")
    a = ap.parse_args()

    hef = hpf.HEF(a.hef)
    in_info = hef.get_input_vstream_infos()[0]
    out_info = hef.get_output_vstream_infos()[0]
    print(f"[HEF] in={in_info.name} {tuple(in_info.shape)} | out={out_info.name}")

    is_img = Path(a.source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    cap = None if is_img else cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    writer = None

    with hpf.VDevice() as vdev:
        cfg = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        ng = vdev.configure(hef, cfg)[0]
        in_p = hpf.InputVStreamParams.make(ng, format_type=hpf.FormatType.UINT8)
        out_p = hpf.OutputVStreamParams.make(ng, format_type=hpf.FormatType.FLOAT32)  # NMS=float32

        with ng.activate():
            with hpf.InferVStreams(ng, in_p, out_p) as pipe:

                def run_one(frame):
                    H, W = frame.shape[:2]
                    inp, r, dw, dh = letterbox(frame)
                    t = time.time()
                    res = pipe.infer({in_info.name: np.expand_dims(inp, 0)})
                    dt = (time.time() - t) * 1000
                    dets = parse_nms(res[out_info.name][0], a.conf, r, dw, dh, W, H)
                    return dets, dt

                if is_img:
                    frame = cv2.imread(a.source)
                    if frame is None:
                        sys.exit(f"이미지 못 읽음: {a.source}")
                    dets, dt = run_one(frame)
                    print(f"검출 {len(dets)}개 ({dt:.1f}ms): " +
                          (", ".join(f"{NAMES.get(c, c)}:{s:.2f}" for c, s, *_ in dets) or "없음"))
                    draw(frame, dets)
                    out = a.save or "result.jpg"
                    cv2.imwrite(out, frame); print(f"저장: {out}")
                    if a.show:
                        cv2.imshow("fire/smoke", frame); cv2.waitKey(0)
                else:
                    if cap is None or not cap.isOpened():
                        sys.exit(f"소스 못 엶: {a.source}")
                    n = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        dets, dt = run_one(frame)
                        draw(frame, dets)
                        if a.save:
                            if writer is None:
                                h, w = frame.shape[:2]
                                writer = cv2.VideoWriter(a.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                                         cap.get(cv2.CAP_PROP_FPS) or 25, (w, h))
                            writer.write(frame)
                        if a.show:
                            cv2.imshow("fire/smoke", frame)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break
                        n += 1
                        if n % 30 == 0:
                            print(f"  {n}프레임 | {dt:.1f}ms ({1000/max(dt,1e-3):.0f}fps) | 검출 {len(dets)}")
                    print(f"총 {n}프레임 처리")

    if cap: cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
