# -*- coding: utf-8 -*-
"""
02. 서브샘플링 + 클립단위 train/val 분할.

- 클립당 FRAMES_PER_CLIP 장만 균등 간격으로 선택(30fps 연속프레임 중복 제거).
- 정상(none)은 NORMAL_RATIO 상한까지만 네거티브로 포함.
- train/val 은 반드시 '클립' 단위 분할(프레임 단위 분할 = 데이터 누수).
- (cls, place) 층화로 도메인 편향 방지.
- 출력: selected.csv, extract_list.txt, data.yaml  (원본은 읽기만)
"""
import sys, csv, random
sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict
import config as C

def pick_even(frames, k):
    """정렬된 frame 리스트에서 균등 간격 k개."""
    frames = sorted(frames)
    if len(frames) <= k:
        return frames
    step = len(frames) / k
    return [frames[int(i * step)] for i in range(k)]

def main():
    random.seed(C.SEED)
    # 1) manifest 읽어 클립별 그룹화
    clip_rows = defaultdict(list)   # (cls_kr, place_kr, clip) -> [row,...]
    with open(C.MANIFEST_CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            clip_rows[(r["cls_kr"], r["place_kr"], r["clip"])].append(r)

    # 2) 클립당 균등 서브샘플
    fs_clips, normal_clips = [], []   # (key, [rows선택])
    for key, rows in clip_rows.items():
        cls_kr = key[0]
        by_frame = {int(r["frame"]): r for r in rows if r["frame"].lstrip("-").isdigit()}
        chosen_frames = pick_even(list(by_frame.keys()), C.FRAMES_PER_CLIP)
        sel = [by_frame[fr] for fr in chosen_frames]
        (normal_clips if cls_kr == "정상" else fs_clips).append((key, sel))

    n_fs = sum(len(s) for _, s in fs_clips)
    # 3) 정상 네거티브 상한: normals = ratio/(1-ratio) * fire_smoke
    cap = int(C.NORMAL_RATIO / (1 - C.NORMAL_RATIO) * n_fs)
    random.shuffle(normal_clips)
    kept_normal, acc = [], 0
    for key, sel in normal_clips:
        if acc >= cap:
            break
        kept_normal.append((key, sel)); acc += len(sel)
    print(f"불꽃+연기 프레임 {n_fs:,} / 정상 상한 {cap:,} -> 정상 사용 {acc:,}")

    all_clips = fs_clips + kept_normal

    # 4) (cls, place) 층화 클립단위 분할
    strata = defaultdict(list)
    for key, sel in all_clips:
        strata[(key[0], key[1])].append((key, sel))
    split_of = {}   # clip key -> 'train'/'val'
    for st, items in strata.items():
        random.shuffle(items)
        n_val = max(1, round(len(items) * C.VAL_RATIO)) if len(items) > 1 else 0
        for i, (key, _) in enumerate(items):
            split_of[key] = "val" if i < n_val else "train"

    # 5) selected.csv + extract_list.txt
    C.WORK.mkdir(parents=True, exist_ok=True)
    extract = set()
    counts = defaultdict(int)
    with open(C.SELECTED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "cls_kr", "cls_code", "place_kr", "clip", "frame", "stem", "json_path", "img_rel"])
        for key, sel in all_clips:
            sp = split_of[key]
            for r in sel:
                w.writerow([sp, r["cls_kr"], r["cls_code"], r["place_kr"], r["clip"],
                            r["frame"], r["stem"], r["json_path"], r["img_rel"]])
                extract.add(r["img_rel"])
                counts[(sp, r["cls_kr"])] += 1

    with open(C.EXTRACT_LIST, "w", encoding="utf-8") as f:
        for rel in sorted(extract):
            f.write(rel + "\n")

    # 6) data.yaml
    ds = str(C.DATASET).replace("\\", "/")
    with open(C.DATA_YAML, "w", encoding="utf-8") as f:
        f.write(f"path: {ds}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(C.CLASS_NAMES)}\n")
        f.write(f"names: {C.CLASS_NAMES}\n")

    print(f"\n[완료] 선택 {len(extract):,}장")
    for k in sorted(counts):
        print(f"  {k[0]:5s} {k[1]:6s} {counts[k]:>8,}")
    print(f"\nselected.csv  : {C.SELECTED_CSV}")
    print(f"extract_list  : {C.EXTRACT_LIST}")
    print(f"data.yaml     : {C.DATA_YAML}")

if __name__ == "__main__":
    main()
