#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HEF 컴파일 입력 준비: 저장소/데이터 원본에서 hailo/ 로 스테이징.
둘 다 gitignore 대상 = 레포에 영구 보관하지 않고 컴파일 직전 재생성.

- ONNX : work/onnx_final/<model>.onnx  ->  hailo/best.onnx    (정적640 production, model.py export가 자동 저장)
- calib: LLM/DATA val 에서 N장 샘플    ->  hailo/calib_data/  (양자화용, seed 고정 = 재현성)

사용:
  python prep_inputs.py                                  # 기본 yolov8s_640_hn_static + calib 300
  python prep_inputs.py --model <onnx_final stem> --n 512
"""
import os, sys, shutil, random, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 루트의 config import
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))


def stage_onnx(model):
    src = C.WORK / "onnx_final" / f"{model}.onnx"
    if not src.exists():
        sys.exit(f"[에러] 저장소에 onnx 없음: {src}\n  먼저: python model.py export <best.pt> --static")
    shutil.copy2(src, os.path.join(HERE, "best.onnx"))
    print(f"onnx : {src.name} -> best.onnx")


def stage_calib(val, n):
    imgs = sorted(glob.glob(os.path.join(val, "*.jpg")))
    if not imgs:
        sys.exit(f"[에러] val 이미지 없음: {val}")
    random.seed(42)                       # 재현성: 항상 같은 샘플
    random.shuffle(imgs)
    out = os.path.join(HERE, "calib_data")
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    for p in imgs[:n]:
        shutil.copy2(p, out)
    print(f"calib: {min(n, len(imgs))}장 -> calib_data/  (원본 {val})")


def main():
    ap = argparse.ArgumentParser(description="HEF 컴파일 입력 스테이징 (onnx + calib)")
    ap.add_argument("--model", default="yolov8s_640_hn_static", help="work/onnx_final/<이것>.onnx")
    ap.add_argument("--n", type=int, default=300, help="calib 샘플 장수")
    ap.add_argument("--val", default=str(C.DATA_HOME / "dataset_24k" / "images" / "val"))
    ap.add_argument("--skip-onnx", action="store_true")
    ap.add_argument("--skip-calib", action="store_true")
    a = ap.parse_args()
    if not a.skip_onnx:
        stage_onnx(a.model)
    if not a.skip_calib:
        stage_calib(a.val, a.n)
    print("준비 완료 -> docker run ... compile_to_hef.py")


if __name__ == "__main__":
    main()
