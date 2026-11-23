import base64
import io
from functools import lru_cache

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

CKPT_PATH = "models/fruit_ripeness_mobilenetv2.pth"


@lru_cache(maxsize=1)
def _load():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    classes = ckpt["classes"]
    head = nn.Sequential(
        nn.Linear(ckpt["input_dim"], ckpt["hidden"]),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(ckpt["hidden"], len(classes)),
    )
    model = models.mobilenet_v2(weights=None)
    model.classifier = head
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    tfms = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    return model, classes, ckpt["class_cn"], tfms


def _predict_pil(image):
    model, classes, class_cn, tfms = _load()
    x = tfms(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return {
        "label": classes[idx],
        "label_cn": class_cn[classes[idx]],
        "confidence": round(float(probs[idx]), 4),
        "all": [
            {
                "label": name,
                "cn": class_cn[name],
                "prob": round(float(probs[i]), 4),
            }
            for i, name in enumerate(classes)
        ],
    }


def predict_path(path):
    return _predict_pil(Image.open(path))


def predict_bytes(data):
    return _predict_pil(Image.open(io.BytesIO(data)))


def predict_base64(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return predict_bytes(base64.b64decode(data_url))


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(predict_path(sys.argv[1]), ensure_ascii=False, indent=2))
