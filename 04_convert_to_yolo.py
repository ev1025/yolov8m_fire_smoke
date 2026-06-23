# -*- coding: utf-8 -*-
"""
04. COCO JSON -> YOLO txt 변환 + 이미지 복사.

- 라벨 bbox = [x_min, y_min, w, h] 픽셀(절대) -> YOLO 정규화 [cls cx cy w h].
- 정상(none) 은 빈 txt(네거티브).
- 이미지는 staging 에서 dataset 으로 '복사'(원본/추출본 이동·삭제 안 함).
- 출력: dataset/images/{train,val}, dataset/labels/{train,val}
"""
import sys, csv, json, shutil
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
import config as C

def main():
    # utf-8-sig: PowerShell Set-Content -Encoding UTF8 가 붙이는 BOM 제거
    prefix = C.ARCHIVE_PREFIX.read_text(encoding="utf-8-sig").strip().lstrip("﻿") \
             if C.ARCHIVE_PREFIX.exists() else C.ARCHIVE_PREFIX_CANDIDATES[0]
    for sp in ("train", "val"):
        (C.DATASET / "images" / sp).mkdir(parents=True, exist_ok=True)
        (C.DATASET / "labels" / sp).mkdir(parents=True, exist_ok=True)

    ok, miss, boxes = 0, 0, 0
    use_staging = C.STAGING.exists()
    print(f"이미지 소스: {'staging(추출본)' if use_staging else 'TS(이미 풀린 분량)'}")

    with open(C.SELECTED_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            sp, stem = r["split"], r["stem"]
            img_rel = r["img_rel"]                       # 불꽃\장소\clip\JPG\stem.jpg
            # 이미지 위치: staging\prefix\img_rel  또는  TS\화재현상\이미지\img_rel
            if use_staging:
                src_img = C.STAGING / prefix / img_rel
            else:
                src_img = C.SRC_EXTRACTED / "화재현상" / "이미지" / img_rel
            if not src_img.exists():
                miss += 1
                continue

            dst_img = C.DATASET / "images" / sp / f"{stem}.jpg"
            shutil.copy2(src_img, dst_img)               # 복사(원본 보존)

            # 라벨 변환
            lines = []
            try:
                with open(r["json_path"], encoding="utf-8") as jf:
                    j = json.load(jf)
                W = j["image"]["width"]; H = j["image"]["height"]
                for a in j.get("annotations", []):
                    cid = C.CLASS_MAP.get(a["categories_id"])
                    if cid is None:
                        continue
                    x, y, w, h = a["bbox"]
                    cx = (x + w / 2) / W; cy = (y + h / 2) / H
                    nw = w / W; nh = h / H
                    cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
                    nw, nh = min(max(nw, 0), 1), min(max(nh, 0), 1)
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            except (OSError, KeyError, json.JSONDecodeError):
                pass
            (C.DATASET / "labels" / sp / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            boxes += len(lines); ok += 1
            if ok % 5000 == 0:
                print(f"  ...{ok:,} 변환", flush=True)

    print(f"\n[완료] 변환 {ok:,}장, 박스 {boxes:,}개, 누락(이미지없음) {miss:,}장")
    if miss:
        print("  누락이 많으면 03 추출이 덜 됐거나 prefix 가 다름. archive_prefix.txt 확인.")
    print(f"dataset: {C.DATASET}\nnext: python 05_train.py")

if __name__ == "__main__":
    main()
