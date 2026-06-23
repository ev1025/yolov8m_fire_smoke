# -*- coding: utf-8 -*-
"""
01. 라벨 색인(매니페스트) 생성.

- zip 을 절대 열지 않는다. 이미 풀려 있는 라벨(JSON) 파일명만 훑어 전체 데이터 색인을 만든다.
- JSON 내용은 읽지 않는다(파일명에 클립/클래스/장소/프레임 인코딩). -> 빠름.
- 출력: manifest.csv  (이 단계는 원본을 읽기만 한다)
"""
import sys, os, csv
sys.stdout.reconfigure(encoding="utf-8")
import config as C

def scan_jsons(root):
    """os.scandir 재귀(빠름). *.json 전체 경로를 yield."""
    stack = [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
                    elif e.name.lower().endswith(".json"):
                        yield e.path
        except (PermissionError, FileNotFoundError):
            continue

def main():
    C.WORK.mkdir(parents=True, exist_ok=True)
    assert C.WORK not in C.DATA_ROOT.parents and C.WORK != C.DATA_ROOT, "출력은 원본 밖이어야 함"

    n = 0
    by_cls = {}      # cls_kr -> count
    by_place = {}    # place_kr -> count
    clips = set()    # (cls_kr, clip)
    with open(C.MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cls_kr", "cls_code", "place_kr", "clip", "frame", "stem", "json_path", "img_rel"])
        for jp in scan_jsons(C.LABEL_ROOT):
            # .../이미지/{cls_kr}/{place_kr}/{clip}/JSON/{stem}.json
            p = jp.replace("/", "\\").split("\\")
            try:
                json_idx = len(p) - 1
                stem     = p[json_idx][:-5]              # .json 제거
                clip     = p[json_idx - 2]
                place_kr = p[json_idx - 3]
                cls_kr   = p[json_idx - 4]
            except IndexError:
                continue
            cls_code = C.CLS_DIRS.get(cls_kr, "?")
            try:
                frame = int(stem.split("_")[-1])
            except ValueError:
                frame = -1
            # 아카이브 내부 상대경로(클래스폴더 기준, JPG 하위). 03에서 prefix 를 앞에 붙임.
            img_rel = f"{cls_kr}\\{place_kr}\\{clip}\\JPG\\{stem}.jpg"
            w.writerow([cls_kr, cls_code, place_kr, clip, frame, stem, jp, img_rel])

            n += 1
            by_cls[cls_kr] = by_cls.get(cls_kr, 0) + 1
            by_place[place_kr] = by_place.get(place_kr, 0) + 1
            clips.add((cls_kr, clip))
            if n % 100000 == 0:
                print(f"  ...{n:,} 처리", flush=True)

    print(f"\n[완료] manifest: {C.MANIFEST_CSV}")
    print(f"총 라벨 {n:,}개 / 클립 {len(clips):,}개\n")
    print("클래스별:")
    for k, v in sorted(by_cls.items()):
        print(f"  {k:6s} {v:>10,}")
    print("\n장소별:")
    for k, v in sorted(by_place.items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {v:>10,}")

if __name__ == "__main__":
    main()
