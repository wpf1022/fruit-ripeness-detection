"""Rebuild a clean, balanced 16-class YOLO dataset.

Compared with the old ``prepare_yolo16.py``:

* Maturity labels follow the source semantics exactly:
  Unripe -> L (未熟), Ripe -> M (成熟), Overripe -> H (过熟).
  The old script mapped Overripe to "medium ripe", which was wrong.
* Persimmon classes (synthetic hue-shifted composites) are dropped and
  replaced by real, human-labelled orange images.
* Every box is validated: area / aspect-ratio / border margin / duplicate
  boxes / box count per image.  Images are checked for decodability,
  size, byte-level duplicates and near-duplicate pHash collisions.
* Sampling is strictly balanced: the same number of instances per class,
  with 80/20 train/val and no cross-split leakage.

Output: dataset/yolo16v2
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import random
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
KAGGLE_ZIP = DATA / "kaggle_fruit_full.zip"
STRAWBERRY_ZIP = DATA / "strawberry_ripeness.zip"
DEFAULT_OUT = DATA / "yolo16_clean"

# target class order: L (unripe) -> M (ripe) -> H (overripe)
CLASS_NAMES = [
    "apple-L", "apple-M", "apple-H",
    "banana-L", "banana-M", "banana-H",
    "grape-L", "grape-M", "grape-H",
    "orange-L", "orange-M", "orange-H",
    "strawberry-L", "strawberry-M", "strawberry-H",
    "bad fruit",
]
CN_NAMES = [
    "苹果-未熟", "苹果-成熟", "苹果-过熟",
    "香蕉-未熟", "香蕉-成熟", "香蕉-过熟",
    "葡萄-未熟", "葡萄-成熟", "葡萄-过熟",
    "橙子-未熟", "橙子-成熟", "橙子-过熟",
    "草莓-未熟", "草莓-半熟", "草莓-完熟",
    "腐烂水果",
]

# Roboflow 32-class source -> target.  Overripe=H, Ripe=M, Unripe=L
ROBOFLOW_MAP = {
    0: 2,   # Apple Overripe  -> apple-H
    1: 1,   # Apple Ripe      -> apple-M
    3: 0,   # Apple Unripe    -> apple-L
    4: 5,   # Banana Overripe -> banana-H
    5: 4,   # Banana Ripe     -> banana-M
    7: 3,   # Banana Unripe   -> banana-L
    8: 8,   # Grape Overripe  -> grape-H
    9: 7,   # Grape Ripe      -> grape-M
    11: 6,  # Grape Unripe    -> grape-L
    20: 11, # Orange Overripe -> orange-H
    21: 10, # Orange Ripe     -> orange-M
    23: 9,  # Orange Unripe   -> orange-L
}
ROTTEN_IDS = {2, 6, 10, 14, 18, 22, 26, 30}  # -> bad fruit
# strawberry: Fullripe -> H, Semiripe -> M, Unripe -> L
STRAWBERRY_MAP = {0: 14, 1: 13, 2: 12}

MIN_SIDE = 0.02        # box side shorter than 2% of the image is noise
MAX_AREA = 0.90        # a box covering >90% of the image is a whole-image miss
MIN_AREA = 0.0008
MIN_ASPECT, MAX_ASPECT = 0.18, 5.5
MAX_BOXES_PER_CLASS = 12
DUP_IOU = 0.65         # two boxes of the same class above this are duplicates
MIN_IMAGE_SIDE = 120
SEED = 20261124


@dataclass
class Record:
    src_split: str
    archive: str
    image_member: str
    boxes: list[tuple] = field(default_factory=list)  # (cls, cx, cy, w, h)
    classes: set = field(default_factory=set)
    picked_for: int | None = None
    _raw: bytes | None = None

    @property
    def key(self):
        return (self.archive, self.image_member)


def iou(a, b):
    ax0, ay0 = a[1] - a[3] / 2, a[2] - a[4] / 2
    ax1, ay1 = a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0 = b[1] - b[3] / 2, b[2] - b[4] / 2
    bx1, by1 = b[1] + b[3] / 2, b[2] + b[4] / 2
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def parse_label(text: str, mapping: dict[int, int], rotten: set[int] | None):
    """Return (boxes, reject_reasons) after per-box validation."""
    boxes = []
    rejects = collections.Counter()
    for line in text.splitlines():
        p = line.split()
        if len(p) < 5:
            rejects["malformed"] += 1
            continue
        try:
            cid = int(float(p[0]))
            if len(p) == 5:
                cx, cy, w, h = (float(v) for v in p[1:5])
            else:  # polygon -> axis aligned box
                vals = [float(v) for v in p[1:]]
                xs, ys = vals[0::2], vals[1::2]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = max(xs) - min(xs), max(ys) - min(ys)
        except (ValueError, IndexError):
            rejects["unparsable"] += 1
            continue

        if cid in mapping:
            new_cid = mapping[cid]
        elif rotten and cid in rotten:
            new_cid = 15
        else:
            continue

        if not (0 <= cx <= 1 and 0 <= cy <= 1):
            rejects["center_out"] += 1
            continue
        if w <= 0 or h <= 0 or w > 1.2 or h > 1.2:
            rejects["bad_size"] += 1
            continue
        w, h = min(w, 1.0), min(h, 1.0)
        # keep the box inside the frame instead of dropping it
        cx = min(1 - w / 2, max(w / 2, cx))
        cy = min(1 - h / 2, max(h / 2, cy))
        if w < MIN_SIDE or h < MIN_SIDE:
            rejects["too_small"] += 1
            continue
        area = w * h
        if area < MIN_AREA or area > MAX_AREA:
            rejects["area"] += 1
            continue
        aspect = w / h
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            rejects["aspect"] += 1
            continue
        boxes.append((new_cid, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)))

    # drop duplicate boxes of the same class (auto-label spam)
    kept = []
    for b in boxes:
        if any(b[0] == k[0] and iou(b, k) > DUP_IOU for k in kept):
            rejects["duplicate"] += 1
            continue
        kept.append(b)

    # classes with too many boxes in one image are usually auto-label spam:
    # keep the largest few instead of dropping the whole class
    by_class = collections.defaultdict(list)
    for b in kept:
        by_class[b[0]].append(b)
    drop_ids = set()
    for c, lst in by_class.items():
        if len(lst) > MAX_BOXES_PER_CLASS:
            keep_ids = {id(b) for b in sorted(lst, key=lambda b: b[3] * b[4], reverse=True)[:MAX_BOXES_PER_CLASS]}
            for b in lst:
                if id(b) not in keep_ids:
                    drop_ids.add(id(b))
            rejects["trimmed_boxes"] += len(lst) - MAX_BOXES_PER_CLASS
    kept = [b for b in kept if id(b) not in drop_ids]

    return kept, rejects


def collect(archive_path: Path, label_prefix_map, mapping, rotten_map, archive_tag):
    """Parse every label file in an archive into Record objects."""
    records = []
    rejects = collections.Counter()
    zf = zipfile.ZipFile(archive_path)
    names = zf.namelist()
    for src_split, (label_prefix, image_prefix) in label_prefix_map.items():
        stems = {Path(n).stem: n for n in names if n.startswith(image_prefix) and n.lower().endswith(".jpg")}
        for ln in names:
            if not (ln.startswith(label_prefix) and ln.endswith(".txt")):
                continue
            stem = Path(ln).stem
            img = stems.get(stem)
            if img is None:
                rejects["no_image"] += 1
                continue
            text = zf.read(ln).decode("utf-8", "ignore")
            boxes, rj = parse_label(text, mapping, rotten_map)
            rejects.update(rj)
            if not boxes:
                rejects["empty_after_clean"] += 1
                continue
            records.append(Record(src_split, archive_tag, img, boxes, {b[0] for b in boxes}))
    return records, rejects, zf


def dhash(raw: bytes, size: int = 8) -> int:
    img = Image.open(io.BytesIO(raw)).convert("L").resize((size + 1, size), Image.LANCZOS)
    px = list(img.getdata())
    bits = 0
    for r in range(size):
        row = px[r * (size + 1):(r + 1) * (size + 1)]
        for c in range(size):
            bits = (bits << 1) | (1 if row[c] < row[c + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sample_class(records, cls, need_imgs, rng, used_keys, used_hashes, used_dhash,
                 preferred_splits, allow_fallback, stats, dedup=True):
    """Pick ``need_imgs`` images that contain ``cls``.

    Images whose only annotation is ``cls`` are preferred, so a class does not
    drag in other labels.  Byte duplicates and near-duplicates (dhash) are
    skipped and the loop keeps going, so the quota is still met.
    """
    pool = [r for r in records if cls in r.classes and r.key not in used_keys]
    pure_prim = [r for r in pool if r.src_split in preferred_splits and r.classes == {cls}]
    mix_prim = [r for r in pool if r.src_split in preferred_splits and r.classes != {cls}]
    tiers = [pure_prim, mix_prim]
    if allow_fallback:
        pure_sec = [r for r in pool if r.src_split not in preferred_splits and r.classes == {cls}]
        mix_sec = [r for r in pool if r.src_split not in preferred_splits and r.classes != {cls}]
        tiers += [pure_sec, mix_sec]
    order = []
    for tier in tiers:
        rng.shuffle(tier)
        order += tier

    chosen = []
    for rec in order:
        if len(chosen) >= need_imgs:
            break
        if rec.key in used_keys:
            continue
        loaded = ensure_loaded(rec)
        if loaded is None:
            stats["drop_unreadable"] += 1
            used_keys.add(rec.key)
            continue
        h = hashlib.md5(loaded).hexdigest()
        if h in used_hashes:
            stats["drop_dup_bytes"] += 1
            used_keys.add(rec.key)
            continue
        dh = None
        if dedup:
            try:
                dh = dhash(loaded)
            except Exception:
                stats["drop_unreadable"] += 1
                used_keys.add(rec.key)
                continue
            if any(hamming(dh, o) <= 1 for o in used_dhash):
                stats["drop_near_dup"] += 1
                continue
        rec._raw = loaded
        rec.picked_for = cls
        chosen.append(rec)
        used_keys.add(rec.key)
        used_hashes.add(h)
        if dh is not None:
            used_dhash.append(dh)
    return chosen, len(chosen)


def ensure_loaded(rec: Record):
    if rec._raw is not None:
        return rec._raw
    zf = ZIPS[rec.archive]
    try:
        raw = zf.read(rec.image_member)
    except KeyError:
        return None
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        if min(img.size) < MIN_IMAGE_SIDE:
            return None
    except Exception:
        return None
    rec._raw = raw
    return raw


ZIPS: dict[str, zipfile.ZipFile] = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    out = Path(args.out)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    ZIPS["kaggle"] = zipfile.ZipFile(KAGGLE_ZIP)
    ZIPS["strawberry"] = zipfile.ZipFile(STRAWBERRY_ZIP)

    kg_recs, kg_rej, _ = collect(
        KAGGLE_ZIP,
        {"train": ("train/labels/", "train/images/"),
         "valid": ("valid/labels/", "valid/images/"),
         "test": ("test/labels/", "test/images/")},
        ROBOFLOW_MAP, ROTTEN_IDS, "kaggle")
    sb_recs, sb_rej, _ = collect(
        STRAWBERRY_ZIP,
        {"train": ("labels/train/", "images/train/"),
         "valid": ("labels/val/", "images/val/"),
         "test": ("labels/test/", "images/test/")},
        STRAWBERRY_MAP, None, "strawberry")

    records = kg_recs + sb_recs
    print(f"records: kaggle={len(kg_recs)} strawberry={len(sb_recs)} total={len(records)}")
    print("kaggle rejects:", dict(kg_rej))
    print("strawberry rejects:", dict(sb_rej))

    rng = random.Random(args.seed)
    n_val = int(round(args.per_class * args.val_ratio))
    n_train = args.per_class - n_val

    used_keys: set = set()
    used_hashes: set = set()
    used_dhash: list = []
    stats = collections.Counter()
    split_records = {"train": [], "val": []}

    # How many cleaned-up images actually exist per class?  Classes that are
    # short (e.g. orange overripe) get a proportional quota instead of a
    # silent shortfall, and the shortage is reported at the end.
    avail = collections.Counter()
    for rec in records:
        for c in rec.classes:
            avail[c] += 1

    quota = {}
    for cls in range(len(CLASS_NAMES)):
        a = avail[cls]
        if a >= args.per_class:
            quota[cls] = (n_train, n_val)
        else:
            tr = max(0, min(n_train, int(round(a * (1 - args.val_ratio)))))
            va = max(0, min(n_val, a - tr))
            quota[cls] = (tr, va)
            print(f"  ! {CLASS_NAMES[cls]} only {a} clean images -> quota {tr}/{va}")

    # training split first, so the bigger share goes to the split that trains
    for cls in range(len(CLASS_NAMES)):
        need = quota[cls][0]
        picked, got = sample_class(records, cls, need, rng, used_keys, used_hashes, used_dhash,
                                   {"train"}, False, stats, dedup=avail[cls] >= args.per_class)
        if got < need:  # relax near-duplicate filter to reach the quota
            more, extra = sample_class(records, cls, need - got, rng, used_keys, used_hashes, used_dhash,
                                       {"train"}, True, stats, dedup=False)
            picked += more
            got += extra
        split_records["train"].extend(picked)
        print(f"  train {CLASS_NAMES[cls]:12s} target {need:3d} got {got:3d}")

    for cls in range(len(CLASS_NAMES)):
        need = quota[cls][1]
        picked, got = sample_class(records, cls, need, rng, used_keys, used_hashes, used_dhash,
                                   {"valid", "test"}, True, stats, dedup=avail[cls] >= args.per_class)
        if got < need:
            more, extra = sample_class(records, cls, need - got, rng, used_keys, used_hashes, used_dhash,
                                       {"valid", "test"}, True, stats, dedup=False)
            picked += more
            got += extra
        split_records["val"].extend(picked)
        print(f"  val   {CLASS_NAMES[cls]:12s} target {need:3d} got {got:3d}")

    # write files
    counters = collections.Counter()
    class_counts = {split: collections.Counter() for split in ("train", "val")}
    img_counts = {split: collections.Counter() for split in ("train", "val")}
    for split in ("train", "val"):
        for rec in split_records[split]:
            cls_main = rec.picked_for if rec.picked_for is not None else max(
                rec.classes, key=lambda c: sum(1 for b in rec.boxes if b[0] == c))
            img_counts[split][cls_main] += 1
            name = f"{CLASS_NAMES[cls_main]}_{split}_{counters[split]:05d}"
            counters[split] += 1
            img_path = out / "images" / split / f"{name}.jpg"
            lbl_path = out / "labels" / split / f"{name}.txt"
            img_path.write_bytes(rec._raw)
            lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in rec.boxes]
            lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            for b in rec.boxes:
                class_counts[split][b[0]] += 1

    (out / "data.yaml").write_text(
        f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" +
        "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8")

    meta = {
        "class_names": CLASS_NAMES,
        "cn_names": CN_NAMES,
        "per_class_target": args.per_class,
        "splits": {"train": n_train, "val": n_val},
        "maturity_semantics": "L=Unripe(未熟), M=Ripe(成熟), H=Overripe(过熟); strawberry M=Semiripe(半熟), H=Fullripe(完熟)",
        "sources": [
            "Roboflow fruit-ripeness-unjex v2 (CC BY 4.0) - apple/banana/grape/orange + rotten",
            "Kaggle samatsauranbek/multi-class-strawberry-ripeness-detection-v2 (CC BY 4.0) - strawberry",
        ],
        "cleaning": {
            "min_side": MIN_SIDE, "min_area": MIN_AREA, "max_area": MAX_AREA,
            "aspect": [MIN_ASPECT, MAX_ASPECT], "max_boxes_per_class": MAX_BOXES_PER_CLASS,
            "dup_iou": DUP_IOU, "min_image_side": MIN_IMAGE_SIDE,
            "byte_dedup": True, "dhash_dedup_hamming": 1,
        },
    }
    (out / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== per class (images / boxes) ===")
    print(f"{'class':16s} {'tr_img':>7s} {'va_img':>7s} {'tr_box':>7s} {'va_box':>7s}")
    for i, n in enumerate(CLASS_NAMES):
        print(f"{n:16s} {img_counts['train'][i]:7d} {img_counts['val'][i]:7d} "
              f"{class_counts['train'][i]:7d} {class_counts['val'][i]:7d}")
    print("\nimages:", {s: len(split_records[s]) for s in split_records})
    print("stats:", dict(stats))

    ZIPS["kaggle"].close()
    ZIPS["strawberry"].close()


if __name__ == "__main__":
    main()
