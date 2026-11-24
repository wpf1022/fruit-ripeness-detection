"""Rebuild the 16-class set with rule-based, self-consistent labels.

The downloaded Roboflow source turned out to mislabel both species and stage
(orange rows contained mangoes, "unripe" bananas were yellow, grape boxes were
vine twigs).  Rather than train on those labels, this script keeps the real
photographs and:

1. rejects boxes whose shape/colour cannot belong to the declared fruit
   (e.g. elongated "oranges", very dark "apples", cut-open flesh);
2. recomputes the maturity level from colour inside the box, then splits each
   fruit into three equal-sized groups by that score, so L < M < H is
   guaranteed to be monotonic in appearance.

Output: dataset/yolo16_final
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "dataset" / "yolo16_clean"
OUT = ROOT / "dataset" / "yolo16_final"

_CLF = None
FRUIT_IDS: dict[str, list[int]] = {}


def _load_clf():
    """Lazy-load the ImageNet classifier used as a species gate."""
    global _CLF, FRUIT_IDS
    if _CLF is not None:
        return _CLF
    import torch
    import torchvision
    from torchvision import transforms
    torch.set_num_threads(4)
    W = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=W).eval()
    cats = W.meta["categories"]
    idx = {c: i for i, c in enumerate(cats)}
    # only fruits the classifier is reliable on; strawberry/grape handled separately
    FRUIT_IDS = {
        "apple": [idx["Granny Smith"], idx["custard apple"], idx["pomegranate"]],
        "banana": [idx["banana"]],
        "orange": [idx["orange"], idx["lemon"]],
        "strawberry": [],
        "grape": [],
    }
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    _CLF = (model, tf)
    return _CLF


def classify(crop: Image.Image) -> list[float]:
    import torch
    model, tf = _load_clf()
    with torch.no_grad():
        return torch.softmax(model(tf(crop).unsqueeze(0)), 1)[0].tolist()

FRUITS = ["apple", "banana", "grape", "orange", "strawberry"]
LEVELS = ["L", "M", "H"]
CN = {"apple": "苹果", "banana": "香蕉", "grape": "葡萄", "orange": "橙子", "strawberry": "草莓"}
CN_LEVEL = {"L": "未熟", "M": "成熟", "H": "过熟"}
CLASS_NAMES = [f"{f}-{lv}" for f in FRUITS for lv in LEVELS] + ["bad fruit"]
CN_NAMES = [f"{CN[f]}-{CN_LEVEL[lv]}" for f in FRUITS for lv in LEVELS] + ["腐烂水果"]


def colour_features(img: Image.Image) -> dict:
    """Colour composition of the central part of the crop (less background)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    if h > 12 and w > 12:
        a = a[h // 5: h - h // 5, w // 5: w - w // 5]
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx, mn = a.max(2), a.min(2)
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    d = mx - mn + 1e-6
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return {
        "green": float(((hue > 70) & (hue < 175) & (s > 0.28) & (v > 0.15)).mean()),
        "yellow": float(((hue >= 20) & (hue <= 70) & (s > 0.38) & (v > 0.3)).mean()),
        "orange": float(((hue >= 8) & (hue < 25) & (s > 0.5) & (v > 0.32)).mean()),
        "red": float((((hue < 12) | (hue > 340)) & (s > 0.4) & (v > 0.18)).mean()),
        "dark": float((v < 0.33).mean()),
        "white": float(((s < 0.16) & (v > 0.72)).mean()),
        "sat": float(s.mean()),
    }


def species_ok(fruit: str, f: dict, aspect: float, area: float) -> bool:
    if fruit == "apple":
        if aspect < 0.55 or aspect > 1.55:
            return False
        if f["dark"] > 0.55 or f["white"] > 0.60:
            return False
        return (f["red"] + f["yellow"] + f["green"]) > 0.12
    if fruit == "banana":
        if aspect < 1.15 and aspect > 0.87:
            return False  # bananas are clearly elongated
        return f["dark"] < 0.60
    if fruit == "grape":
        if area < 0.012:
            return False
        if f["green"] > 0.55 and f["dark"] < 0.2:
            return False  # leaves
        return f["dark"] + f["green"] + f["red"] > 0.2
    if fruit == "orange":
        if aspect < 0.72 or aspect > 1.32:
            return False  # elongated fruit is mango
        if (f["orange"] + f["yellow"]) < 0.10:
            return False
        return f["dark"] < 0.55
    if fruit == "strawberry":
        if aspect < 0.6 or aspect > 1.7:
            return False
        return (f["red"] + f["yellow"] + f["white"]) > 0.15
    return True


def ripeness_score(f: dict) -> float:
    """Higher = riper.  Green pulls down, red/orange/yellow/dark pull up."""
    return (1.00 * f["red"] + 0.85 * f["orange"] + 0.55 * f["yellow"]
            + 0.40 * f["dark"] - 0.75 * f["green"] - 0.30 * f["white"])


def grape_box(img: Image.Image) -> tuple[float, float, float, float] | None:
    """Locate the bunch itself, because the source boxes sat on vine twigs."""
    a = np.asarray(img.convert("RGB").resize((320, 320)), dtype=np.float32) / 255.0
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx, mn = a.max(2), a.min(2)
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0)
    d = mx - mn + 1e-6
    hue = np.where(mx == r, ((g - b) / d) % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    purple = (hue > 250) & (hue < 340) & (s > 0.3) & (v > 0.12)
    dark_berry = (v < 0.4) & (s > 0.25)
    mask = purple | dark_berry
    if mask.mean() < 0.02:
        return None
    ys, xs = np.nonzero(mask)
    x0, x1 = np.percentile(xs, [3, 97])
    y0, y1 = np.percentile(ys, [3, 97])
    bw, bh = (x1 - x0) / 320, (y1 - y0) / 320
    if bw < 0.12 or bh < 0.12 or bw > 0.99 or bh > 0.99:
        return None
    cx, cy = (x0 + x1) / 2 / 320, (y0 + y1) / 2 / 320
    cx = min(1 - bw / 2, max(bw / 2, cx))
    cy = min(1 - bh / 2, max(bh / 2, cy))
    return (cx, cy, bw, bh)


def main():
    _load_clf()
    _load_clf()
    meta_src = json.loads((SRC / "dataset_meta.json").read_text(encoding="utf-8"))
    old_names = meta_src["class_names"]

    if OUT.exists():
        shutil.rmtree(OUT)
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    kept = {f: [] for f in FRUITS}
    rejected = {f: 0 for f in FRUITS}
    bad_pool = []

    for split in ("train", "val"):
        for lbl in sorted((SRC / "labels" / split).glob("*.txt")):
            rows = [p.split() for p in lbl.read_text(encoding="utf-8").splitlines() if p.strip()]
            boxes = [(int(float(p[0])), *map(float, p[1:5])) for p in rows if len(p) >= 5]
            if not boxes:
                continue
            img_path = SRC / "images" / split / f"{lbl.stem}.jpg"
            if not img_path.exists():
                continue
            fruit = None
            for f in FRUITS:
                if any(b[0] == old_names.index(f"{f}-L") + lv for b in boxes for lv in range(3)):
                    fruit = f
                    break
            if fruit is None:
                if any(b[0] == old_names.index("bad fruit") for b in boxes):
                    bad_pool.append((img_path, max(boxes, key=lambda b: b[3] * b[4])))
                continue
            own = [b for b in boxes if b[0] // 3 == FRUITS.index(fruit)]
            if not own:
                continue
            box = max(own, key=lambda b: b[3] * b[4])
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            w, h = img.size
            x0, y0 = max(0, int((box[1] - box[3] / 2) * w)), max(0, int((box[2] - box[4] / 2) * h))
            x1, y1 = min(w, int((box[1] + box[3] / 2) * w)), min(h, int((box[2] + box[4] / 2) * h))
            if x1 - x0 < 16 or y1 - y0 < 16:
                rejected[fruit] += 1
                continue
            crop = img.crop((x0, y0, x1, y1))
            feats = colour_features(crop)
            aspect = (x1 - x0) / (y1 - y0)
            if not species_ok(fruit, feats, aspect, box[3] * box[4]):
                rejected[fruit] += 1
                continue
            # ImageNet check: catches mangoes filed under orange, peppers under
            # apple, and collage/candy images.  Skipped for strawberry (field
            # shots are classified as cabbage/pineapple) and grape (no class).
            if fruit in FRUIT_IDS and FRUIT_IDS[fruit]:
                pr = classify(crop)
                if sum(pr[i] for i in FRUIT_IDS[fruit]) < 0.10:
                    rejected[fruit] += 1
                    continue
            if fruit == "strawberry" and feats["red"] < 0.02:
                rejected[fruit] += 1
                continue
            if fruit == "grape":
                gb = grape_box(img)
                if gb is not None:
                    box = (box[0], gb[0], gb[1], gb[2], gb[3])
            kept[fruit].append({
                "img": img, "box": box, "score": ripeness_score(feats), "src": img_path.name,
            })

    print("rejected by species/shape rule:", rejected)
    print("kept:", {f: len(v) for f, v in kept.items()}, "bad pool:", len(bad_pool))

    rng = random.Random(20261124)
    counts = {}
    for f in FRUITS:
        items = sorted(kept[f], key=lambda d: d["score"])
        n = len(items)
        thirds = [items[: n // 3], items[n // 3: 2 * n // 3], items[2 * n // 3:]]
        for lv, chunk in zip(LEVELS, thirds):
            cid = FRUITS.index(f) * 3 + LEVELS.index(lv)
            rng.shuffle(chunk)
            n_val = max(1, int(round(len(chunk) * 0.2)))
            counts[f"{f}-{lv}"] = len(chunk)
            for split, part in (("val", chunk[:n_val]), ("train", chunk[n_val:])):
                for i, d in enumerate(part):
                    stem = f"{f}-{lv}_{split}_{i:03d}"
                    d["img"].save(OUT / "images" / split / f"{stem}.jpg", "JPEG", quality=90)
                    b = d["box"]
                    (OUT / "labels" / split / f"{stem}.txt").write_text(
                        f"{cid} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n", encoding="utf-8")

    # bad fruit: keep a balanced share
    target_bad = int(np.mean(list(counts.values()))) if counts else 60
    rng.shuffle(bad_pool)
    bad_pool = bad_pool[: target_bad]
    n_val = max(1, int(round(len(bad_pool) * 0.2)))
    counts["bad fruit"] = len(bad_pool)
    for split, part in (("val", bad_pool[:n_val]), ("train", bad_pool[n_val:])):
        for i, (img_path, box) in enumerate(part):
            stem = f"bad fruit_{split}_{i:03d}"
            shutil.copyfile(img_path, OUT / "images" / split / f"{stem}.jpg")
            (OUT / "labels" / split / f"{stem}.txt").write_text(
                f"15 {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n", encoding="utf-8")

    (OUT / "data.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: 16\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)),
        encoding="utf-8")

    meta = {
        "class_names": CLASS_NAMES, "cn_names": CN_NAMES,
        "source": "real photographs from Roboflow fruit-ripeness-unjex v2 + Kaggle strawberry ripeness",
        "labelling": "species/shape rule filter + colour-based maturity tertiles (L<M<H monotonic in appearance)",
        "maturity_semantics": "L=least ripe by colour, M=middle, H=most ripe/overripe",
        "rejected_by_rule": rejected,
        "per_class": counts,
        "caveat": "the original downloadable annotations were unreliable; labels here are rule-derived, not human ground truth",
    }
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'class':16s} {'total':>6s}")
    tot = 0
    for n in CLASS_NAMES:
        c = counts.get(n, 0)
        tot += c
        print(f"{n:16s} {c:6d}")
    print("total", tot)


if __name__ == "__main__":
    main()
