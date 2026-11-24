"""为每类生成候选实例拼图，便于逐张目视核对（只读，不改数据）。"""
import json
from pathlib import Path

from PIL import Image, ImageDraw

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "yolo16_final"
OUT = BASE / "contacts_pick"
CAND = 10          # 每类候选数
CELL = 300
COLS = 5
COLORS = [
    "#f87171", "#ef4444", "#b91c1c", "#fbbf24", "#f59e0b", "#b45309",
    "#a78bfa", "#8b5cf6", "#6d28d9", "#fb923c", "#f97316", "#c2410c",
    "#fb7185", "#f43f5e", "#be123c", "#64748b",
]


def main():
    OUT.mkdir(exist_ok=True)
    meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
    names, cn = meta["class_names"], meta["cn_names"]
    index = {}
    for split in ("train", "val"):
        img_dir, lbl_dir = DATA / "images" / split, DATA / "labels" / split
        for lbl in sorted(lbl_dir.glob("*.txt")):
            boxes = []
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    boxes.append((int(float(p[0])), *(float(v) for v in p[1:5])))
            if not boxes:
                continue
            img = next((p for ext in (".jpg", ".png") if (p := img_dir / f"{lbl.stem}{ext}").exists()), None)
            if img is None:
                continue
            index.setdefault(boxes[0][0], []).append(
                {"split": split, "stem": lbl.stem, "img": img, "boxes": boxes}
            )

    picks = {}
    for cid, name in enumerate(names):
        cands = sorted(index.get(cid, []), key=lambda x: (x["split"] != "train", x["stem"]))[:CAND]
        picks[name] = [f"{c['split']}/{c['stem']}" for c in cands]
        rows = (len(cands) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * CELL, rows * CELL), "#eef0f3")
        for i, c in enumerate(cands):
            im = Image.open(c["img"]).convert("RGB")
            W, H = im.size
            d = ImageDraw.Draw(im)
            for cls, cx, cy, bw, bh in c["boxes"]:
                d.rectangle([(cx - bw / 2) * W, (cy - bh / 2) * H,
                             (cx + bw / 2) * W, (cy + bh / 2) * H],
                            outline=COLORS[cls], width=max(3, W // 140))
            im.thumbnail((CELL - 6, CELL - 6), Image.LANCZOS)
            canvas = Image.new("RGB", (CELL, CELL), "#ffffff")
            canvas.paste(im, ((CELL - im.width) // 2, (CELL - im.height) // 2))
            sheet.paste(canvas, ((i % COLS) * CELL, (i // COLS) * CELL))
        sheet.save(OUT / f"{cid:02d}_{name}.jpg", quality=88)

    (OUT / "candidates.json").write_text(json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("sheets:", len(picks), "->", OUT)


if __name__ == "__main__":
    main()
