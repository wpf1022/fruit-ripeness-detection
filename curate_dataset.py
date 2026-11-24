"""按种类闸门清洗 dataset/yolo16_final：移出错标样本，保留全部正确样本。

判据：ImageNet 分类器对「标注框内区域」的物种概率 + 橙子的长宽比（芒果偏长卵形）。
被移出的图片与标注挪到 dataset/rejected_species，并写入清单，可追溯、可回滚。
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "yolo16_final"
REJ = BASE / "dataset" / "rejected_species"
GATE = json.loads((BASE / "models" / "species_gate.json").read_text())

THRESH = {"apple": 0.30, "banana": 0.50, "orange": 0.35, "strawberry": 0.20}
ORANGE_ASPECT = (0.85, 1.25)   # 橙子接近圆形；芒果明显偏长


def main():
    meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    REJ.mkdir(parents=True, exist_ok=True)

    moved = []
    counts = Counter()
    for split in ("train", "val"):
        img_dir, lbl_dir = DATA / "images" / split, DATA / "labels" / split
        for lbl in sorted(lbl_dir.glob("*.txt")):
            parts = lbl.read_text().split()
            if len(parts) < 5:
                continue
            cid = int(float(parts[0]))
            cname = names[cid]
            group = cname.split("-")[0]
            stat = GATE.get(lbl.stem)
            if stat is None:
                continue

            reason = None
            thr = THRESH.get(group)
            if thr is not None:
                key = {"apple": "apple", "banana": "banana", "orange": "orange", "strawberry": "straw"}[group]
                if stat[key] < thr:
                    reason = f"{key}={stat[key]:.2f}<{thr}"
            if reason is None and group == "orange":
                a = stat["aspect"]
                if not (ORANGE_ASPECT[0] <= a <= ORANGE_ASPECT[1]):
                    reason = f"aspect={a:.2f} 不像橙子"

            if reason:
                img = next((p for ext in (".jpg", ".png") if (p := img_dir / f"{lbl.stem}{ext}").exists()), None)
                dest = REJ / split
                dest.mkdir(parents=True, exist_ok=True)
                if img is not None:
                    shutil.move(str(img), dest / img.name)
                shutil.move(str(lbl), dest / lbl.name)
                moved.append({"split": split, "stem": lbl.stem, "class": cname, "reason": reason})
            else:
                counts[cname] += 1

    meta["per_class"] = {n: counts[n] for n in names}
    meta["rejected_species"] = len(moved)
    meta["rejected_log"] = "dataset/rejected_species/_log.json"
    (DATA / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (REJ / "_log.json").write_text(json.dumps(moved, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"移出错标 {len(moved)} 张，清洗后各类：")
    for n in names:
        print(f"  {n:14s} {counts[n]:3d}")
    print("总计", sum(counts.values()))


if __name__ == "__main__":
    main()
