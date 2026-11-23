"""Build a 16-class YOLO detection dataset from freely licensed sources.

Target classes (same schema as the requested 3,198-image dataset):
  0 apple-H  1 apple-M  2 apple-L
  3 banana-H 4 banana-M 5 banana-L
  6 grape-H  7 grape-M  8 grape-L
  9 strawberry-H 10 strawberry-M 11 strawberry-L
  12 persimmon-H 13 persimmon-M 14 persimmon-L
  15 bad fruit

Sources
-------
* Kaggle/Roboflow "Fruit Ripeness Object Detection" (CC BY 4.0): supplies
  apple / banana / grape at ripe / overripe / unripe stages and the rotten
  class used as "bad fruit".
* Kaggle "Multi-Class Strawberry Ripeness Detection v2" (CC BY 4.0):
  Fullripe / Semiripe / Unripe strawberries.
* Persimmon images are added separately by ``prepare_persimmon.py``.
"""
from __future__ import annotations

import collections
import os
import random
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
OUT = DATA / "yolo16"
KAGGLE_ZIP = DATA / "kaggle_fruit_full.zip"
STRAWBERRY_ZIP = DATA / "strawberry_ripeness.zip"

CLASS_NAMES = [
    "apple-H", "apple-M", "apple-L",
    "banana-H", "banana-M", "banana-L",
    "grape-H", "grape-M", "grape-L",
    "strawberry-H", "strawberry-M", "strawberry-L",
    "persimmon-H", "persimmon-M", "persimmon-L",
    "bad fruit",
]
CN_NAMES = [
    "苹果-高成熟", "苹果-中成熟", "苹果-低成熟",
    "香蕉-高成熟", "香蕉-中成熟", "香蕉-低成熟",
    "葡萄-高成熟", "葡萄-中成熟", "葡萄-低成熟",
    "草莓-高成熟", "草莓-中成熟", "草莓-低成熟",
    "柿子-高成熟", "柿子-中成熟", "柿子-低成熟",
    "腐烂水果",
]

# source class index -> target class index
FRUIT_MAP = {
    1: 0,   # Apple Ripe      -> apple-H
    0: 1,   # Apple Overripe  -> apple-M
    3: 2,   # Apple Unripe    -> apple-L
    5: 3,   # Banana Ripe     -> banana-H
    4: 4,   # Banana Overripe -> banana-M
    7: 5,   # Banana Unripe   -> banana-L
    9: 6,   # Grape Ripe      -> grape-H
    8: 7,   # Grape Overripe  -> grape-M
    11: 8,  # Grape Unripe    -> grape-L
}
ROTTEN_IDS = {2, 6, 10, 14, 18, 22, 26, 30}  # every "Rotten" class in the 32-class source
STRAWBERRY_MAP = {0: 9, 1: 10, 2: 11}  # Fullripe, Semiripe, Unripe

PER_CLASS_TRAIN = 200
PER_CLASS_VAL = 55
SEED = 20261123
rng = random.Random(SEED)


def ensure_dirs() -> None:
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def read_labels(zf: zipfile.ZipFile, name: str) -> list[list[str]]:
    try:
        raw = zf.read(name).decode("utf-8", "ignore").splitlines()
    except KeyError:
        return []
    rows: list[list[str]] = []
    for line in raw:
        p = line.split()
        if len(p) < 5:
            continue
        try:
            if len(p) == 5:
                # already YOLO format: class cx cy w h
                cx, cy, w, h = [float(x) for x in p[1:5]]
            else:
                # strawberry set stores polygons: class x1 y1 x2 y2 ... -> cx cy w h
                coords = [float(x) for x in p[1:]]
                xs, ys = coords[0::2], coords[1::2]
                x0, x1 = min(xs), max(xs)
                y0, y1 = min(ys), max(ys)
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                w, h = x1 - x0, y1 - y0
            # clamp so the box stays fully inside the image
            w, h = min(1.0, max(1e-4, w)), min(1.0, max(1e-4, h))
            cx = min(1 - w / 2, max(w / 2, cx))
            cy = min(1 - h / 2, max(h / 2, cy))
            rows.append([p[0], *[f"{v:.6f}" for v in (cx, cy, w, h)]])
        except (ValueError, IndexError):
            continue
    return rows


def collect_source(zf: zipfile.ZipFile, prefix: str, mapping: dict[int, int],
                   rotten_ids: set[int] | None = None,
                   image_prefix: str | None = None) -> dict[int, list[tuple[str, str]]]:
    """Return target_class -> [(image_member, remapped_label_text), ...].

    ``prefix`` points at the label directory.  When the archive keeps labels
    and images as sibling folders (e.g. ``labels/train`` + ``images/train``),
    pass ``image_prefix`` explicitly.
    """
    by_class: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)
    names = set(zf.namelist())
    label_members = [n for n in names if n.startswith(prefix) and n.endswith(".txt")]
    for label_name in label_members:
        stem = Path(label_name).stem
        if image_prefix:
            image_name = f"{image_prefix}{stem}.jpg"
            if image_name not in names:
                image_name = f"{image_prefix}{stem}.png"
        else:
            image_name = label_name.replace("/labels/", "/images/").replace(".txt", ".jpg")
            if image_name not in names:
                image_name = image_name[:-4] + ".png"
        if image_name not in names:
            continue
        rows = read_labels(zf, label_name)
        if not rows:
            continue
        new_rows = []
        for row in rows:
            cid = int(row[0])
            if cid in mapping:
                new_rows.append([str(mapping[cid]), *row[1:]])
            elif rotten_ids and cid in rotten_ids:
                new_rows.append(["15", *row[1:]])
        if not new_rows:
            continue
        text = "\n".join(" ".join(r) for r in new_rows) + "\n"
        for cid in {int(r[0]) for r in new_rows}:
            by_class[cid].append((image_name, text))
    for cid in by_class:
        rng.shuffle(by_class[cid])
    return by_class


def select_balanced(pools: list[dict[int, list[tuple[str, str]]]], n_train: int, n_val: int,
                    exclude: set[str] | None = None):
    """Greedy selection so every class gets a fair quota."""
    combined: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)
    for pool in pools:
        for cid, items in pool.items():
            combined[cid].extend(items)
    for cid in combined:
        rng.shuffle(combined[cid])

    exclude = exclude or set()
    chosen: dict[str, str] = {}
    used_count = collections.Counter()
    for target in (n_train, n_val):
        # fill classes in increasing order of availability (rarest first)
        order = sorted(combined, key=lambda c: len(combined[c]))
        for cid in order:
            need = target - used_count[cid]
            if need <= 0:
                continue
            added = 0
            for image_name, text in combined[cid]:
                if added >= need:
                    break
                if image_name in chosen or image_name in exclude:
                    continue
                chosen[image_name] = text
                used_count[cid] += 1
                added += 1
    return chosen, used_count


def write_split(zf: zipfile.ZipFile, member: str, text: str, split: str, stem: str) -> None:
    ext = Path(member).suffix.lower()
    dst_img = OUT / "images" / split / f"{stem}{ext}"
    dst_lbl = OUT / "labels" / split / f"{stem}.txt"
    dst_img.write_bytes(zf.read(member))
    dst_lbl.write_text(text, encoding="utf-8")


def write_data_yaml() -> None:
    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" +
        "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    meta = {
        "class_names": CLASS_NAMES,
        "cn_names": CN_NAMES,
        "sources": [
            "Kaggle/Roboflow fruit-ripeness-unjex/2 (CC BY 4.0)",
            "Kaggle samatsauranbek/multi-class-strawberry-ripeness-detection-v2 (CC BY 4.0)",
            "Persimmon samples: see prepare_persimmon.py",
        ],
    }
    import json
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    with zipfile.ZipFile(KAGGLE_ZIP) as zf:
        names = set(zf.namelist())
        train_pool = collect_source(zf, "train/", FRUIT_MAP, ROTTEN_IDS)
        val_pool = collect_source(zf, "valid/", FRUIT_MAP, ROTTEN_IDS)
        test_pool = collect_source(zf, "test/", FRUIT_MAP, ROTTEN_IDS)
        strawberry = zipfile.ZipFile(STRAWBERRY_ZIP)
        sb_names = set(strawberry.namelist())
        sb_train = collect_source(strawberry, "labels/train/", STRAWBERRY_MAP,
                                  image_prefix="images/train/")
        sb_val = collect_source(strawberry, "labels/val/", STRAWBERRY_MAP,
                                image_prefix="images/val/")

        train_choice, train_counts = select_balanced([train_pool, sb_train], PER_CLASS_TRAIN, 0)
        val_choice, val_counts = select_balanced(
            [val_pool, test_pool, sb_val, train_pool, sb_train], 0, PER_CLASS_VAL,
            exclude=set(train_choice),
        )

        for split, choice in (("train", train_choice), ("val", val_choice)):
            for i, (member, text) in enumerate(sorted(choice.items())):
                stem = f"{split}_{i:05d}"
                if member in names:
                    write_split(zf, member, text, split, stem)
                elif member in sb_names:
                    write_split(strawberry, member, text, split, stem)
        strawberry.close()

    write_data_yaml()
    print("train counts", dict(sorted(train_counts.items())))
    print("val counts", dict(sorted(val_counts.items())))
    for split in ("train", "val"):
        print(split, "images", len(list((OUT / "images" / split).iterdir())))


if __name__ == "__main__":
    main()
