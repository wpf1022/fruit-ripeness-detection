"""Command-line fruit ripeness detection: python detect_photo.py <image> [conf]"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "fruit_yolov8_detector.pt"
META = json.loads((ROOT / "dataset" / "yolo16" / "dataset_meta.json").read_text(encoding="utf-8"))
NAMES, CN = META["class_names"], META["cn_names"]


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python detect_photo.py <图片路径> [置信度阈值]")
        return
    image = Path(sys.argv[1])
    conf = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    if not MODEL.exists():
        print("模型不存在，请先运行 python train_detector.py")
        return
    model = YOLO(str(MODEL))
    result = model.predict(str(image), conf=conf, imgsz=416, device="cpu", verbose=False)[0]
    print(f"图像: {image}")
    print(f"检出目标数: {len(result.boxes) if result.boxes else 0}")
    if result.boxes:
        for box in result.boxes:
            cid = int(box.cls.item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            print(f"  {NAMES[cid]:16s} {CN[cid]:10s} 置信度={float(box.conf.item()):.3f} 框={xyxy}")
    out = ROOT / "uploads_detect" / f"{image.stem}_detected.jpg"
    result.save(filename=str(out))
    print(f"标注图已保存: {out}")


if __name__ == "__main__":
    main()
