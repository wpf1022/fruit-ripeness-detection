"""Create the three persimmon ripeness classes for the 16-class YOLO dataset.

Fruits-360 (MIT) contains 166+ "Kaki" images but no ripeness stages.  We turn
each into three weakly-supervised ripeness variants by shifting fruit hue
(orange -> yellow-orange -> yellow-green) and paste the extracted fruit onto
random real backgrounds sampled from the other dataset images.  A tight
bounding box is derived from the foreground mask, so the result is genuine
detection data rather than whole-image classification.

Target classes: 12 persimmon-H, 13 persimmon-M, 14 persimmon-L.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
OUT = DATA / "yolo16"
PARQUETS = [DATA / "fruits360_test.parquet", DATA / "fruits360_train.parquet"]
KAKI_LABEL = 44
SIZE = 416
SEED = 20261124
rng = random.Random(SEED)

# hue delta (OpenCV scale 0-179) applied to the fruit only
VARIANTS = [
    (12, 0),    # persimmon-H: natural deep orange
    (13, 11),   # persimmon-M: yellow-orange
    (14, 25),   # persimmon-L: yellow-green / unripe
]


def load_kaki() -> list[np.ndarray]:
    images: list[np.ndarray] = []
    for path in PARQUETS:
        if not path.exists():
            continue
        try:
            table = pq.read_table(path, columns=["image", "label"])
        except Exception as exc:
            print(f"skip unreadable {path.name}: {str(exc)[:80]}")
            continue
        rows = [i for i, lab in enumerate(table.column("label").to_pylist()) if lab == KAKI_LABEL]
        col = table.column("image").to_pylist()
        for i in rows:
            raw = col[i]["bytes"]
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None and img.size:
                images.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC))
    return images


def background_paths() -> list[Path]:
    folder = OUT / "images" / "train"
    return [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".png")][:1200]


def make_background(paths: list[Path]) -> np.ndarray:
    if paths:
        p = rng.choice(paths)
        bg = cv2.imread(str(p))
        if bg is not None:
            bg = cv2.resize(bg, (SIZE, SIZE))
            bg = cv2.GaussianBlur(bg, (0, 0), rng.uniform(6, 16))
            # darken slightly so the fruit stands out
            return cv2.convertScaleAbs(bg, alpha=rng.uniform(0.55, 0.85), beta=0)
    return np.full((SIZE, SIZE, 3), 235, dtype=np.uint8)


def extract_fruit(img: np.ndarray, hue_delta: int):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Fruits-360 has a near-white/black background; saturation separates fruit.
    mask = cv2.inRange(s, 45, 255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 120:
        return None
    x0, x1 = max(0, xs.min()), min(SIZE - 1, xs.max())
    y0, y1 = max(0, ys.min()), min(SIZE - 1, ys.max())
    if (x1 - x0) < 30 or (y1 - y0) < 30:
        return None

    shifted = hsv.copy()
    shifted[:, :, 0] = ((h.astype(np.int16) + hue_delta) % 180).astype(np.uint8)
    # mild saturation boost keeps the unripe variant visually distinct
    shifted[:, :, 1] = np.clip(s.astype(np.int16) + 12, 0, 255).astype(np.uint8)
    fruit = cv2.cvtColor(shifted, cv2.COLOR_HSV2BGR)
    return fruit, mask, (x0, y0, x1, y1)


def bbox_line(box) -> str:
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(SIZE - 1, x1), min(SIZE - 1, y1)
    cx = ((x0 + x1) / 2) / SIZE
    cy = ((y0 + y1) / 2) / SIZE
    w, h = (x1 - x0) / SIZE, (y1 - y0) / SIZE
    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def compose(img: np.ndarray, hue_delta: int, bgs: list[Path]):
    result = extract_fruit(img, hue_delta)
    if result is None:
        return None
    fruit, mask, box = result
    bg = make_background(bgs)
    # allow small random translation/scale of the fruit on the new background
    scale = rng.uniform(0.85, 1.0)
    fruit = cv2.resize(fruit, None, fx=scale, fy=scale)
    mask = cv2.resize(mask, None, fx=scale, fy=scale)
    fh, fw = mask.shape
    dx = rng.randint(0, max(0, SIZE - fw))
    dy = rng.randint(0, max(0, SIZE - fh))
    canvas = bg.copy()
    region = canvas[dy:dy + fh, dx:dx + fw]
    region[mask > 0] = fruit[mask > 0]
    canvas[dy:dy + fh, dx:dx + fw] = region
    # box is (x0, y0, x1, y1) in the original SIZE x SIZE image.  After scaling
    # by `scale` and pasting at (dx, dy), the fruit region becomes:
    new_box = (dx + box[0] * scale,
               dy + box[1] * scale,
               dx + box[2] * scale,
               dy + box[3] * scale)
    new_box = (max(0, new_box[0]), max(0, new_box[1]), min(SIZE - 1, new_box[2]), min(SIZE - 1, new_box[3]))
    return canvas, bbox_line(new_box)


def main() -> None:
    kaki = load_kaki()
    print("kaki images", len(kaki))
    rng.shuffle(kaki)
    split_at = int(len(kaki) * 0.8)
    bgs = background_paths()
    counters = {"train": 0, "val": 0}
    for split, source in (("train", kaki[:split_at]), ("val", kaki[split_at:])):
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        for idx, img in enumerate(source):
            for cid, hue_delta in VARIANTS:
                made = compose(img, hue_delta, bgs)
                if made is None:
                    continue
                canvas, line = made
                stem = f"persimmon_{split}_{idx:04d}_{cid}"
                cv2.imwrite(str(img_dir / f"{stem}.jpg"), canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
                (lbl_dir / f"{stem}.txt").write_text(f"{cid} {line}\n", encoding="utf-8")
                counters[split] += 1
    print("generated", counters)


if __name__ == "__main__":
    main()
