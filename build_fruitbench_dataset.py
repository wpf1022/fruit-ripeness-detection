"""Build the final 16-class YOLO dataset from FruitBench.

Classes: apple / banana / grape / orange / strawberry x (Unripe=L, Mature=M,
Rotten=H) + bad fruit (Pest-damage).

FruitBench ships classification images without boxes, so a foreground box is
derived per image: the background colour is estimated from the four corners,
pixels far from it are treated as fruit, and the largest blob gives the box.
When that fails (busy background) a centred box is used instead.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "dataset" / "fruitbench"
OUT = ROOT / "dataset" / "yolo16_fb"

FRUITS = ["apple", "banana", "grape", "orange", "strawberry"]
STAGE_TO_LEVEL = {"Unripe": "L", "Mature": "M", "Rotten": "H"}
CLASS_NAMES = [f"{f}-{lv}" for f in FRUITS for lv in ("L", "M", "H")] + ["bad fruit"]
CN = {"apple": "苹果", "banana": "香蕉", "grape": "葡萄", "orange": "橙子", "strawberry": "草莓"}
CN_LEVEL = {"L": "未熟", "M": "成熟", "H": "过熟"}
CN_NAMES = [f"{CN[f]}-{CN_LEVEL[lv]}" for f in FRUITS for lv in ("L", "M", "H")] + ["腐烂水果"]


def foreground_box(path: Path) -> tuple[float, float, float, float] | None:
    """Estimate a tight box around the fruit."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((320, 320))
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape[:2]
    k = max(6, min(h, w) // 12)
    corners = np.concatenate([
        a[:k, :k].reshape(-1, 3), a[:k, -k:].reshape(-1, 3),
        a[-k:, :k].reshape(-1, 3), a[-k:, -k:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(a - bg, axis=2)
    thr = max(45.0, float(np.percentile(dist, 55)))
    mask = dist > thr
    if mask.mean() < 0.05 or mask.mean() > 0.96:
        return None
    # keep the largest connected component using a coarse flood fill on a grid
    ys, xs = np.nonzero(mask)
    if len(xs) < 200:
        return None
    # dominant cluster: drop outliers at the 2/98 percentile
    x0, x1 = np.percentile(xs, [1, 99])
    y0, y1 = np.percentile(ys, [1, 99])
    bw, bh = (x1 - x0) / w, (y1 - y0) / h
    if bw < 0.15 or bh < 0.15 or bw > 0.99 or bh > 0.99:
        return None
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    bw, bh = min(bw, 0.98), min(bh, 0.98)
    cx = min(1 - bw / 2, max(bw / 2, cx))
    cy = min(1 - bh / 2, max(bh / 2, cy))
    return (cx, cy, bw, bh)


def fallback_box(w: int, h: int):
    # fruit photos are usually centred and cover most of the frame
    side_w, side_h = 0.86, 0.86
    return (0.5, 0.5, side_w, side_h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # gather pools
    pools: dict[int, list[Path]] = {i: [] for i in range(16)}
    for fi, fruit in enumerate(FRUITS):
        for stage, level in STAGE_TO_LEVEL.items():
            cid = fi * 3 + ("LMH".index(level))
            d = SRC / fruit / stage
            if d.exists():
                pools[cid].extend(sorted(d.glob("*.jpg")))
    pest = []
    for fruit in FRUITS:
        d = SRC / fruit / "Pest-damage"
        if d.exists():
            pest.extend(sorted(d.glob("*.jpg")))
    pools[15] = pest

    box_stats = {"estimated": 0, "fallback": 0}
    counts = {split: [0] * 16 for split in ("train", "val")}
    missing = []

    for cid in range(16):
        items = pools[cid][:]
        rng.shuffle(items)
        items = items[: args.per_class]
        if len(items) < args.per_class:
            missing.append((CLASS_NAMES[cid], len(items)))
        n_val = max(1, int(round(len(items) * args.val_ratio)))
        n_train = len(items) - n_val
        for split, chunk in (("train", items[:n_train]), ("val", items[n_train:])):
            for src in chunk:
                box = foreground_box(src)
                if box is None:
                    with Image.open(src) as im:
                        box = fallback_box(*im.size)
                    box_stats["fallback"] += 1
                else:
                    box_stats["estimated"] += 1
                stem = f"{CLASS_NAMES[cid]}_{split}_{counts[split][cid]:03d}"
                counts[split][cid] += 1
                shutil.copyfile(src, OUT / "images" / split / f"{stem}.jpg")
                (OUT / "labels" / split / f"{stem}.txt").write_text(
                    f"{cid} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n",
                    encoding="utf-8")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 16\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8")

    meta = {
        "class_names": CLASS_NAMES,
        "cn_names": CN_NAMES,
        "per_class": args.per_class,
        "source": "FruitBench (Hugging Face TJIET/FruitBench, expert-labelled growth stages, public 10% subset)",
        "maturity_semantics": "L=Unripe(未熟), M=Mature(成熟), H=Rotten(过熟/腐烂); bad fruit=Pest-damage(病虫害次果)",
        "boxes": "auto foreground box from background-difference; centred box as fallback",
        "box_stats": box_stats,
        "splits": {s: sum(counts[s]) for s in counts},
        "per_class_counts": {CLASS_NAMES[i]: {"train": None, "val": None} for i in range(16)},
    }
    # fill per-class counts from filenames
    for i, name in enumerate(CLASS_NAMES):
        meta["per_class_counts"][name] = {
            "train": len(list((OUT / "labels" / "train").glob(f"{name}_*.txt"))),
            "val": len(list((OUT / "labels" / "val").glob(f"{name}_*.txt"))),
        }
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("box stats", box_stats)
    if missing:
        print("short classes:", missing)
    print(f"{'class':16s} {'train':>6s} {'val':>5s}")
    for name in CLASS_NAMES:
        c = meta["per_class_counts"][name]
        print(f"{name:16s} {c['train']:6d} {c['val']:5d}")
    print("total images", sum(meta["splits"].values()))


if __name__ == "__main__":
    main()
