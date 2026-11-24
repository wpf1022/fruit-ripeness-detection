"""推理入口：种类由 CNN 判定，成熟度由颜色语义量化。

流程：
1. MobileNetV2（6 类：苹果/香蕉/葡萄/橙子/草莓/腐烂水果）做 TTA 推理
   （整图 + 中心 0.8 裁剪 + 水平翻转），得到种类概率；
2. 若判为「腐烂水果」直接给出，不再套成熟度；
3. 否则用 ripeness.py 定位果实区域，按颜色语义给出未熟 / 成熟 / 完熟。

这样既不会再出现“苹果被当成芒果”的硬分错，也不再用“非支持类别”把正常水果挡回去。
"""

from __future__ import annotations

import base64
import io
import os
from functools import lru_cache

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

import ripeness

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE, "models", "species_mobilenetv2.pth")
BAD = "bad fruit"
VAL_ACC_PATH = os.path.join(BASE, "models", "evaluation_species.json")


@lru_cache(maxsize=1)
def _load():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = models.mobilenet_v2(weights=None)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(ckpt["input_dim"], ckpt["hidden"]),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(ckpt["hidden"], len(ckpt["classes"])),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tfms = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    return model, ckpt["classes"], ckpt["class_cn"], tfms


def _views(image: Image.Image):
    rgb = image.convert("RGB")
    w, h = rgb.size
    s = max(8, int(min(w, h) * 0.8))
    center = rgb.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    return [rgb, center, rgb.transpose(Image.FLIP_LEFT_RIGHT)]


@lru_cache(maxsize=1)
def _load_detector():
    """YOLOv8 检测器，只用来定位果实（类别由分类模型重判）。缺失则返回 None。"""
    try:
        import sys
        import types

        if "polars" not in sys.modules:       # ultralytics 的懒加载依赖，环境里没有就打个桩
            stub = types.ModuleType("polars")
            stub.read_csv = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stub"))
            sys.modules["polars"] = stub
        from ultralytics import YOLO

        path = os.path.join(BASE, "models", "fruit_yolov8_detector.pt")
        return YOLO(path) if os.path.exists(path) else None
    except Exception:
        return None


def _locate(image: Image.Image):
    """返回 (果实框, 来源说明)。检测器优先，失败退回颜色连通域。"""
    rgb = image.convert("RGB")
    det = _load_detector()
    box = None
    if det is not None:
        try:
            res = det.predict(rgb, imgsz=480, conf=0.25, verbose=False)[0]
            if len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                conf = res.boxes.conf.cpu().numpy()
                area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                best = int((area * conf).argmax())
                box = [float(v) for v in xyxy[best]]
        except Exception:
            box = None
    if box is not None:
        return _pad(box, rgb.size, 0.10), "detector"
    fallback = ripeness.fruit_region(rgb)
    if fallback:
        return fallback, "colour"
    return None, "whole"


def _pad(box, size, ratio):
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    x0, y0 = max(0, x0 - w * ratio), max(0, y0 - h * ratio)
    x1, y1 = min(size[0], x1 + w * ratio), min(size[1], y1 + h * ratio)
    return (int(x0), int(y0), int(x1), int(y1))


def _predict_pil(image: Image.Image) -> dict:
    model, classes, class_cn, tfms = _load()

    # 1) 先定位果实：田间小果（草莓）整图会被大片绿叶带偏，聚焦后再判
    region, how = _locate(image)
    focus = image.crop(region) if region else image
    views = _views(focus) + (_views(image) if region else [])
    blocks = [3, 3] if region else [3]
    with torch.no_grad():
        probs = torch.softmax(model(torch.stack([tfms(v) for v in views])), dim=1)
    parts = probs.split(blocks)
    if len(parts) == 1:
        probs = parts[0].mean(0)
    else:
        # 果实主体权重更高，整图只作上下文约束
        probs = 0.75 * parts[0].mean(0) + 0.25 * parts[1].mean(0)
    probs = probs.tolist()

    order = sorted(range(len(classes)), key=lambda i: -probs[i])
    idx = order[0]
    species = classes[idx]
    species_conf = float(probs[idx])

    if species_conf < 0.40:
        return {
            "label": "Unknown",
            "label_cn": "未能确定",
            "confidence": round(species_conf, 4),
            "unsupported": True,
            "species": None,
            "species_cn": "未能确定",
            "all": [{"label": c, "cn": class_cn[c], "prob": round(float(p), 4)}
                    for c, p in zip(classes, probs)],
            "level": None,
            "level_cn": None,
            "level_probs": None,
            "bbox": region,
            "reason": "图片里没有清晰可辨的水果主体，请换一张主体更完整、光线更均匀的照片",
        }

    if species == BAD:
        return {
            "label": BAD,
            "label_cn": "腐烂水果",
            "confidence": round(species_conf, 4),
            "unsupported": False,
            "species": BAD,
            "species_cn": "腐烂水果",
            "all": [{"label": c, "cn": class_cn[c], "prob": round(float(p), 4)}
                    for c, p in zip(classes, probs)],
            "level": None,
            "level_cn": None,
            "level_probs": None,
            "bbox": region,
            "note": "检测到明显腐烂/霉变特征，建议直接丢弃，不再做成熟度分级",
        }

    ripe = ripeness.analyse(image, species, region=region)
    level_conf = ripe["level_probs"][ripe["level"]]
    # 联合置信度：种类对、档位也对，才算这次判断成立
    joint = species_conf * (0.65 + 0.35 * level_conf)

    result = {
        "label": species,
        "label_cn": f"{class_cn[species]} · {ripe['level_cn']}",
        "confidence": round(joint, 4),
        "unsupported": False,
        "species": species,
        "species_cn": class_cn[species],
        "species_confidence": round(species_conf, 4),
        "all": [{"label": c, "cn": class_cn[c], "prob": round(float(p), 4)}
                for c, p in zip(classes, probs)],
        "level": ripe["level"],
        "level_cn": ripe["level_cn"],
        "level_probs": ripe["level_probs"],
        "level_confidence": level_conf,
        "level_hint": ripe["level_hint"],
        "bbox": ripe["bbox"],
        "located_by": how,
        "ripeness": {
            "score": ripe["score"],
            "features": ripe["features"],
            "bbox": ripe["bbox"],
            "region_reliable": ripe["region_reliable"],
        },
        "note": (f"颜色证据：绿 {ripe['features']['green'] * 100:.0f}% · "
                 f"褐斑 {ripe['features']['brown'] * 100:.0f}% · "
                 f"着色度 {ripe['score'] * 100:.0f}%"),
    }
    return result


def predict_path(path: str) -> dict:
    with Image.open(path) as im:
        return _predict_pil(im)


def predict_bytes(data: bytes) -> dict:
    with Image.open(io.BytesIO(data)) as im:
        return _predict_pil(im)


def predict_base64(data_url: str) -> dict:
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return predict_bytes(base64.b64decode(data_url))


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(predict_path(sys.argv[1]), ensure_ascii=False, indent=2))
