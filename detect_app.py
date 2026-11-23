"""16-class fruit ripeness YOLO detection demo (Flask)."""
from __future__ import annotations

import io
import json
import sqlite3
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fruit_yolov8_detector.pt"
META_PATH = ROOT / "dataset" / "yolo16" / "dataset_meta.json"
REPORT_PATH = ROOT / "models" / "detection_evaluation.json"
UPLOADS = ROOT / "uploads_detect"
DB_PATH = ROOT / "detection_history.db"
SAMPLES = ROOT / "dataset" / "yolo16" / "images" / "val"

UPLOADS.mkdir(exist_ok=True)
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

META = json.loads(META_PATH.read_text(encoding="utf-8"))
NAMES = META["class_names"]
CN = META["cn_names"]
COLORS = [
    "#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e", "#14b8a6",
    "#06b6d4", "#3b82f6", "#6366f1", "#8b5cf6", "#a855f7", "#d946ef",
    "#ec4899", "#f43f5e", "#0ea5e9", "#64748b",
]

model = YOLO(str(MODEL_PATH)) if MODEL_PATH.exists() else None


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT, filename TEXT, count INTEGER,
                labels TEXT, confidence REAL, duration REAL
            )"""
        )


init_db()


def hex_rgb(value: str):
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def load_font(size=16):
    for p in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(14, int(canvas.width / 55)))
    for det in detections:
        color = hex_rgb(COLORS[det["class_id"] % len(COLORS)])
        x1, y1, x2, y2 = det["box"]
        for i in range(3):
            draw.rectangle([x1 - i, y1 - i, x2 + i, y2 + i], outline=color, width=2)
        text = f'{det["cn_name"]} {det["confidence"]:.0%}'
        box = draw.textbbox((0, 0), text, font=font)
        tw, th = box[2] - box[0], box[3] - box[1]
        ty = max(0, y1 - th - 10)
        draw.rectangle([x1, ty, min(canvas.width, x1 + tw + 12), ty + th + 8], fill=color)
        draw.text((x1 + 6, ty + 2), text, fill="white", font=font)
    return canvas


def run_inference(image: Image.Image, conf: float = 0.25):
    if model is None:
        raise RuntimeError("模型文件不存在：models/fruit_yolov8_detector.pt")
    t0 = time.time()
    results = model.predict(image, conf=conf, imgsz=416, device="cpu", verbose=False)
    result = results[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            cid = int(box.cls.item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            conf_v = float(box.conf.item())
            detections.append({
                "class_id": cid,
                "label": NAMES[cid],
                "cn_name": CN[cid],
                "confidence": round(conf_v, 4),
                "box": xyxy,
            })
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections, round(time.time() - t0, 3)


def save_record(filename: str, detections: list[dict], duration: float):
    labels = ", ".join(sorted({d["cn_name"] for d in detections})) or "未检出"
    conf = max((d["confidence"] for d in detections), default=0.0)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO records(ts, filename, count, labels, confidence, duration) VALUES(datetime('now','localtime'),?,?,?,?,?)",
            (filename, len(detections), labels, conf, duration),
        )


@app.route("/")
def index():
    return send_from_directory(ROOT / "static", "detect_index.html")


@app.route("/sample/<path:name>")
def sample(name: str):
    return send_from_directory(SAMPLES, name)


@app.route("/result/<path:name>")
def result_image(name: str):
    return send_from_directory(UPLOADS, name)


@app.route("/api/meta")
def api_meta():
    report = {}
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    samples = sorted(p.name for p in SAMPLES.glob("*.jpg"))[:12]
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            "SELECT ts, filename, count, labels, confidence, duration FROM records ORDER BY id DESC LIMIT 12"
        ).fetchall()
    return jsonify({
        "names": NAMES, "cn_names": CN, "colors": COLORS,
        "report": report, "samples": samples,
        "history": [dict(zip(["ts", "filename", "count", "labels", "confidence", "duration"], r)) for r in rows],
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        conf = float(request.form.get("conf", 0.25))
        sample_name = request.form.get("sample", "")
        if sample_name:
            image = Image.open(SAMPLES / sample_name).convert("RGB")
            filename = sample_name
        elif "image" in request.files:
            file = request.files["image"]
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            filename = file.filename or "upload.jpg"
        elif request.is_json:
            data = request.get_json(silent=True) or {}
            sample_name = data.get("sample", "")
            if not sample_name:
                return jsonify({"error": "缺少 sample 字段"}), 400
            image = Image.open(SAMPLES / sample_name).convert("RGB")
            filename = sample_name
        else:
            return jsonify({"error": "请上传图片或指定示例图"}), 400

        detections, duration = run_inference(image, conf)
        out_name = f"{uuid.uuid4().hex[:12]}.jpg"
        draw_detections(image, detections).save(UPLOADS / out_name, quality=92)
        save_record(filename, detections, duration)
        return jsonify({
            "success": True,
            "result_image": f"/result/{out_name}",
            "count": len(detections),
            "detections": detections,
            "duration": duration,
        })
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    print("Fruit detection demo: http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
