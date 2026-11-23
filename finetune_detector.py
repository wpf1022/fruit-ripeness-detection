"""Short fine-tune of the trained detector after fixing persimmon box coords."""
from __future__ import annotations

import sys
import types

import ultralytics.data.utils as _data_utils
import ultralytics.utils.checks as _checks

_checks.check_font = lambda *args, **kwargs: None
_data_utils.check_font = lambda *args, **kwargs: None
if "polars" not in sys.modules:
    stub = types.ModuleType("polars")
    stub.read_csv = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
    sys.modules["polars"] = stub

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CFG = ROOT / "dataset" / "yolo16" / "data.yaml"
SRC = ROOT / "models" / "fruit_yolov8_detector.pt"
DST = ROOT / "models" / "fruit_yolov8_detector.pt"

model = YOLO(str(SRC))
model.train(data=str(CFG), epochs=4, imgsz=416, batch=8, device="cpu",
            workers=0, lr0=0.0005, patience=4, plots=False, verbose=False,
            project=str(ROOT / "runs" / "detect"), name="fruit_finetune", exist_ok=True)
model.save(str(DST))
print("fine-tuned weights saved:", DST)
