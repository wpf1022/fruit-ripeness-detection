"""Render per-class contact sheets so labels can be checked by eye.

Usage: python make_contact_sheet.py [--data dataset/yolo16_clean] [--per-class 6]
Writes contacts/sheet_1.jpg, sheet_2.jpg ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
COLORS = [
    (230, 57, 70), (244, 162, 97), (42, 157, 143),
    (69, 123, 157), (131, 56, 236), (255, 143, 171),
    (6, 214, 160), (255, 183, 3), (58, 134, 255),
    (247, 37, 133), (114, 9, 183), (0, 150, 199),
    (188, 108, 37), (82, 183, 136), (255, 111, 97),
    (108, 117, 125),
]
CELL = 236
HEAD = 30
ROWS_PER_SHEET = 8


def load_font(size):
    for p in ("C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def fit(img: Image.Image, size: int) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (238, 240, 243))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def draw_boxes(base: Image.Image, boxes, names, font):
    d = ImageDraw.Draw(base)
    w, h = base.size
    for cls, cx, cy, bw, bh in boxes:
        x0, y0 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x1, y1 = (cx + bw / 2) * w, (cy + bh / 2) * h
        color = COLORS[cls % len(COLORS)]
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
        tag = names[cls]
        tw = d.textlength(tag, font=font)
        ty = max(0, y0 - 12)
        d.rectangle([x0, ty, x0 + tw + 4, ty + 12], fill=color)
        d.text((x0 + 2, ty), tag, fill="white", font=font)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "dataset" / "yolo16_clean"))
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "contacts"))
    args = ap.parse_args()

    data = Path(args.data)
    meta = json.loads((data / "dataset_meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    label_font = load_font(11)
    head_font = load_font(15)

    for split in ("train", "val"):
        # collect per class
        per_class = {i: [] for i in range(len(names))}
        for lbl in sorted((data / "labels" / split).glob("*.txt")):
            boxes = []
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                boxes.append((int(float(p[0])), *map(float, p[1:5])))
            if not boxes:
                continue
            img_path = data / "images" / split / f"{lbl.stem}.jpg"
            if not img_path.exists():
                continue
            for cls in {b[0] for b in boxes}:
                if len(per_class[cls]) < args.per_class:
                    per_class[cls].append((img_path, boxes))
        for cls in per_class:
            while len(per_class[cls]) < args.per_class:
                per_class[cls].append(None)

        # paginate
        cls_list = list(range(len(names)))
        pages = [cls_list[i:i + ROWS_PER_SHEET] for i in range(0, len(cls_list), ROWS_PER_SHEET)]
        for pi, page in enumerate(pages, 1):
            cols = args.per_class
            W = cols * (CELL + 8) + 8
            H = len(page) * (CELL + HEAD + 8) + 8
            sheet = Image.new("RGB", (W, H), "white")
            d = ImageDraw.Draw(sheet)
            for r, cls in enumerate(page):
                y = 8 + r * (CELL + HEAD + 8)
                d.text((10, y + 6), f"{cls:02d}  {names[cls]}", fill=(20, 24, 33), font=head_font)
                for c, item in enumerate(per_class[cls]):
                    x = 8 + c * (CELL + 8)
                    cy = y + HEAD
                    if item is None:
                        cell = Image.new("RGB", (CELL, CELL), (248, 249, 250))
                        ImageDraw.Draw(cell).text((8, CELL // 2), "no image", fill=(150, 150, 150), font=head_font)
                    else:
                        img_path, boxes = item
                        cell = fit(Image.open(img_path), CELL)
                        draw_boxes(cell, boxes, names, label_font)
                    sheet.paste(cell, (x, cy))
            path = out / f"{split}_sheet_{pi}.jpg"
            sheet.save(path, quality=88)
            print("wrote", path, sheet.size)


if __name__ == "__main__":
    main()
