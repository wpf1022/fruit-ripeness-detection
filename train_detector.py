"""Train a 16-class YOLOv8 detector for fruit ripeness and quality."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import sys
import types

import ultralytics.data.utils as _data_utils
import ultralytics.utils.checks as _checks

# polars is only used to read results.csv when saving; its caller already
# falls back to {} on failure, so a tiny stub avoids a heavy download.
if "polars" not in sys.modules:
    _stub = types.ModuleType("polars")

    def _noop_read_csv(*args, **kwargs):
        raise RuntimeError("polars stub")

    _stub.read_csv = _noop_read_csv
    sys.modules["polars"] = _stub

# Ultralytics insists on importing matplotlib to locate a font during dataset
# checks.  It is only needed for annotated result images, not for training.
# Stub it out so training works in a minimal environment.
_checks.check_font = lambda *args, **kwargs: None
_data_utils.check_font = lambda *args, **kwargs: None

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA_CFG = ROOT / "dataset" / "yolo16" / "data.yaml"
META = ROOT / "dataset" / "yolo16" / "dataset_meta.json"
MODELS = ROOT / "models"
WEIGHTS = MODELS / "yolov8n.pt"
BEST = MODELS / "fruit_yolov8_detector.pt"
REPORT = MODELS / "detection_evaluation.json"


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    src = WEIGHTS if WEIGHTS.exists() else "yolov8n.pt"
    model = YOLO(str(src))
    t0 = time.time()
    model.train(
        data=str(DATA_CFG),
        epochs=8,
        imgsz=416,
        batch=8,
        device="cpu",
        workers=0,
        lr0=0.001,
        patience=8,
        project=str(ROOT / "runs" / "detect"),
        name="fruit_ripeness_yolov8",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    elapsed = round(time.time() - t0, 1)

    metrics = model.val(data=str(DATA_CFG), imgsz=416, device="cpu", workers=0,
                        verbose=False, plots=False)
    run_best = ROOT / "runs" / "detect" / "fruit_ripeness_yolov8" / "weights" / "best.pt"
    if run_best.exists():
        shutil.copy2(run_best, BEST)
    else:
        # polars is unavailable in this environment, so Ultralytics could not
        # write its checkpoint.  Save the in-memory trained weights directly.
        model.save(str(BEST))
        print(f"saved final weights to {BEST}")

    names = json.loads(META.read_text(encoding="utf-8"))["class_names"]
    per_class = {}
    try:
        maps = metrics.box.maps
        for i, name in enumerate(names):
            per_class[name] = round(float(maps[i]), 6) if i < len(maps) else None
    except Exception:
        pass

    report = {
        "task": "16-class fruit ripeness object detection",
        "model": "YOLOv8n (Ultralytics)",
        "dataset": "dataset/yolo16",
        "train_seconds": elapsed,
        "epochs": 8,
        "imgsz": 416,
        "mAP50": round(float(metrics.box.map50), 6),
        "mAP50_95": round(float(metrics.box.map), 6),
        "precision": round(float(metrics.box.mp), 6),
        "recall": round(float(metrics.box.mr), 6),
        "per_class_AP50": per_class,
        "weights": str(BEST.relative_to(ROOT)),
        "sources": json.loads(META.read_text(encoding="utf-8"))["sources"],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
