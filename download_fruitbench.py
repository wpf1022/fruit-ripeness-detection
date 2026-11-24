"""Download the FruitBench public subset (expert-labelled growth stages).

Source: https://huggingface.co/datasets/TJIET/FruitBench (via hf-mirror)
Layout: Data/<Fruit>/<Stage>/0001.png
Stages: Unripe / Mature / Rotten / Pest-damage, 50 images each.

Images are converted to 960px wide JPEG so the dataset stays small.
"""
from __future__ import annotations

import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "dataset" / "fruitbench"
BASE = "https://hf-mirror.com/datasets/TJIET/FruitBench/resolve/main/Data"
FRUITS = {"Apple": "apple", "Banana": "banana", "Orange": "orange",
          "Grape": "grape", "Strawberry": "strawberry"}
STAGES = ["Unripe", "Mature", "Rotten", "Pest-damage"]
PER_STAGE = 50
MAX_W = 960
QUALITY = 88


def fetch(fruit: str, stage: str, idx: int) -> tuple[str, bool]:
    dst = OUT / FRUITS[fruit] / stage / f"{idx:04d}.jpg"
    if dst.exists() and dst.stat().st_size > 2000:
        return str(dst), True
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{fruit}/{stage}/{idx:04d}.png"
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=90) as r:
                raw = r.read()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            if img.width > MAX_W:
                img = img.resize((MAX_W, round(img.height * MAX_W / img.width)), Image.LANCZOS)
            img.save(dst, "JPEG", quality=QUALITY, optimize=True)
            return str(dst), True
        except Exception as exc:
            if attempt == 2:
                return f"{url} :: {type(exc).__name__} {exc}", False
    return url, False


def main():
    tasks = [(f, s, i) for f in FRUITS for s in STAGES for i in range(1, PER_STAGE + 1)]
    print(f"downloading {len(tasks)} images -> {OUT}")
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, *t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            msg, good = fut.result()
            if good:
                ok += 1
            else:
                fail += 1
                print("  FAIL", msg)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)} ok={ok} fail={fail}", flush=True)
    print(f"done ok={ok} fail={fail}")


if __name__ == "__main__":
    main()
