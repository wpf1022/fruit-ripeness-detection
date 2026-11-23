import io
import json
import os
from collections import Counter

import pyarrow.parquet as pq
from PIL import Image

RAW = "dataset/raw"
OUT = "dataset"
SAMPLES = "samples"
CLASS_NAMES = ["Raw_Banana", "Raw_Mango", "Ripe_Banana", "Ripe_Mango"]
CLASS_CN = {
    "Raw_Banana": "未熟香蕉",
    "Raw_Mango": "未熟芒果",
    "Ripe_Banana": "成熟香蕉",
    "Ripe_Mango": "成熟芒果",
}


def save_image(value, path):
    if isinstance(value, dict):
        data = value.get("bytes")
    else:
        data = value
    if data is None:
        raise ValueError("image field has no bytes")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((320, 320))
    img.save(path, "JPEG", quality=90)


def convert_split(src_name, split):
    src = os.path.join(RAW, src_name)
    pf = pq.ParquetFile(src)
    counts = Counter()
    for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
        for row in batch.to_pylist():
            label = int(row["label"])
            name = CLASS_NAMES[label]
            folder = os.path.join(OUT, split, name)
            os.makedirs(folder, exist_ok=True)
            counts[name] += 1
            filename = f"{split}_{name}_{counts[name]:05d}.jpg"
            save_image(row["image"], os.path.join(folder, filename))
    return dict(counts)


def main():
    os.makedirs(SAMPLES, exist_ok=True)
    train_counts = convert_split("train.parquet", "train")
    val_counts = convert_split("test.parquet", "val")
    for name in CLASS_NAMES:
        src = os.path.join(OUT, "val", name)
        files = sorted(os.listdir(src))[:2]
        for i, f in enumerate(files, 1):
            data = open(os.path.join(src, f), "rb").read()
            with open(os.path.join(SAMPLES, f"sample_{name}_{i}.jpg"), "wb") as fh:
                fh.write(data)
    meta = {
        "class_names": CLASS_NAMES,
        "class_cn": CLASS_CN,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "source": "https://huggingface.co/datasets/darthraider/fruit-ripeness-detection-dataset",
        "license": "apache-2.0",
    }
    with open("models/dataset_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
