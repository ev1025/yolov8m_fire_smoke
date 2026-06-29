#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX -> HAR -> HEF 자동 컴파일 (Hailo-8L).

Hailo Model Zoo의 `hailomz compile` CLI를 subprocess로 래핑한다.
ClientRunner로 직접 파싱/양자화하지 않음 (YOLO NMS .alls 수동 매칭 회피).

전제 (중요):
- ONNX는 정적(고정) shape 여야 함. dynamic export면 Hailo DFC가 거부 -> imgsz 고정 재export.
    yolo export model=best.pt imgsz=640 format=onnx opset=11
- yolov8s 기본 설정은 640x640. 다른 해상도면 model zoo yaml의 preprocessing.input_shape 수정 + --yaml.
- 환경: Hailo Dataflow Compiler(3.x) + hailo_model_zoo v2.x 설치되어 있어야 함 (hailo8l 지원).
    master/5.x 브랜치는 hailo8l 미지원 -> Dockerfile 사용 권장.

사용:
    python3.10 compile_to_hef.py --onnx best.onnx --calib-path calib_data --classes 2
"""
import argparse, subprocess, sys, shutil
from pathlib import Path

# ===================== 기본 설정 (필요시 수정) =====================
MODEL_NAME  = "yolov8s"     # Model Zoo 아키텍처명 (yolov8n / yolov8s / yolov8m ...)
NUM_CLASSES = 2             # 학습한 클래스 수 (fire, smoke)
HW_ARCH     = "hailo8l"     # ⚠ 반드시 hailo8l. HEF는 hailo8 <-> hailo8l 호환 안 됨.
IMG_EXTS    = (".jpg", ".jpeg", ".png", ".bmp")
# ================================================================


def warn_if_dynamic(onnx_path: Path):
    """onnx 입력이 dynamic shape면 경고 (Hailo는 정적 shape 필요). onnx 미설치면 건너뜀."""
    try:
        import onnx
    except ImportError:
        print("[정보] onnx 미설치 -> 입력 shape 점검 생략")
        return
    try:
        g = onnx.load(str(onnx_path)).graph.input[0]
        dims = g.type.tensor_type.shape.dim
        shape = [d.dim_value if d.HasField("dim_value") else (d.dim_param or "?") for d in dims]
        print(f"[정보] ONNX 입력 '{g.name}' shape = {shape}")
        if any((not isinstance(s, int)) or s <= 0 for s in shape):
            print("[경고] 입력이 동적(dynamic) shape. Hailo는 고정 shape 필요.")
            print("       -> yolo export model=best.pt imgsz=640 format=onnx opset=11  재export 권장")
    except Exception as e:
        print(f"[정보] onnx shape 점검 실패(무시): {e}")


def count_calib(calib_dir: Path) -> int:
    if not calib_dir.is_dir():
        sys.exit(f"[에러] 캘리브레이션 폴더 없음: {calib_dir}")
    n = sum(1 for p in calib_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if n == 0:
        sys.exit(f"[에러] 캘리브레이션 이미지 0장: {calib_dir} (jpg/png 필요)")
    if n < 64:
        print(f"[경고] 캘리브레이션 {n}장 (권장 500~1000장; 적으면 양자화 정확도 저하)")
    return n


def main():
    ap = argparse.ArgumentParser(description="ONNX -> HEF (Hailo-8L) via `hailomz compile`")
    ap.add_argument("--onnx", required=True, help="정적-shape ONNX (예: best.onnx)")
    ap.add_argument("--calib-path", required=True, help="캘리브레이션 이미지 폴더")
    ap.add_argument("--model", default=MODEL_NAME, help=f"Model Zoo 아키텍처명 (기본 {MODEL_NAME})")
    ap.add_argument("--classes", type=int, default=NUM_CLASSES, help=f"클래스 수 (기본 {NUM_CLASSES})")
    ap.add_argument("--hw-arch", default=HW_ARCH, help=f"타깃 칩 (기본 {HW_ARCH})")
    ap.add_argument("--output-dir", default="hef_output", help="작업/출력 폴더")
    ap.add_argument("--yaml", default=None, help="커스텀 네트워크 yaml (해상도/노드 변경 시)")
    ap.add_argument("--end-nodes", nargs="*", default=None, dest="end_nodes",
                    help="hailomz --end-node-names. YOLOv8은 dfl 디코드 앞 6개 conv 지정 "
                         "(미지정 시 자동파서가 dfl/Reshape에서 끊어 NMS 실패)")
    ap.add_argument("--performance", action="store_true", help="성능 최적화 컴파일(오래 걸림)")
    a = ap.parse_args()

    onnx = Path(a.onnx).resolve()
    calib = Path(a.calib_path).resolve()
    outdir = Path(a.output_dir).resolve()

    # ---- 사전 점검 ----
    if not onnx.is_file():
        sys.exit(f"[에러] ONNX 없음: {onnx}")
    if a.hw_arch != "hailo8l":
        print(f"[경고] hw-arch='{a.hw_arch}'. Hailo-8L 타깃이면 hailo8l 이어야 함.")
    n = count_calib(calib)
    warn_if_dynamic(onnx)
    if shutil.which("hailomz") is None:
        sys.exit("[에러] `hailomz` 없음. Hailo DFC(3.x) + hailo_model_zoo(v2.x) 설치 필요 (Dockerfile 권장).")
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- hailomz compile 명령 구성 ----
    # hailomz compile <model> --ckpt <onnx> --hw-arch <arch> --calib-path <dir> --classes <N>
    cmd = ["hailomz", "compile", a.model,
           "--ckpt", str(onnx),
           "--hw-arch", a.hw_arch,
           "--calib-path", str(calib),
           "--classes", str(a.classes)]
    if a.yaml:
        cmd += ["--yaml", str(Path(a.yaml).resolve())]
    if a.end_nodes:
        cmd += ["--end-node-names"] + a.end_nodes
    if a.performance:
        cmd.append("--performance")

    print("=" * 64)
    print(f" 모델 {a.model} | 클래스 {a.classes} | arch {a.hw_arch} | calib {n}장")
    print(f" ONNX  {onnx}")
    print(f" 실행  {' '.join(cmd)}")
    print(f" cwd   {outdir}  (.hef 여기 생성)")
    print("=" * 64)

    # ---- 실행: 로그 실시간 출력 + 캡처. .hef는 cwd에 생성되므로 cwd=outdir ----
    log = []
    try:
        proc = subprocess.Popen(cmd, cwd=str(outdir), text=True, bufsize=1,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.append(line)
        ret = proc.wait()
    except FileNotFoundError:
        sys.exit("[에러] hailomz 실행 불가 (PATH 확인).")

    log_path = outdir / "compile_log.txt"
    log_path.write_text("".join(log), encoding="utf-8")

    # ---- 실패 처리 + 흔한 원인 힌트 ----
    if ret != 0:
        joined = "".join(log).lower()
        print("\n" + "=" * 64)
        print(f"[실패] hailomz compile exit={ret}. 마지막 로그 ↓")
        sys.stdout.write("".join(log[-40:]))
        if "invalid choice" in joined and "hailo8l" in joined:
            print("\n[힌트] 이 model_zoo가 hailo8l 미지원(master/5.x). v2.x 브랜치 + DFC 3.x 필요.")
        if "libhailort" in joined or "cannot open shared object" in joined:
            print("\n[힌트] HailoRT/DFC/model_zoo 버전 불일치. 세 버전 호환 조합으로 맞출 것.")
        if "end_node" in joined or "start_node" in joined or "failed to parse" in joined:
            print("\n[힌트] 파싱 노드 불일치. --yaml + end_node_names 지정 필요할 수 있음.")
        print(f"\n전체 로그: {log_path}")
        sys.exit(ret)

    # ---- HEF 탐색 ----
    hefs = sorted(outdir.rglob("*.hef"), key=lambda p: p.stat().st_mtime)
    if hefs:
        h = hefs[-1]
        print("\n" + "=" * 64)
        print(f"[성공] HEF 생성: {h}  ({h.stat().st_size / 1e6:.1f} MB)")
        print(f"  디바이스 확인:  hailortcli run {h.name}")
    else:
        print(f"\n[경고] 컴파일 exit 0 이나 .hef 못 찾음. 로그 확인: {log_path}")


if __name__ == "__main__":
    main()
