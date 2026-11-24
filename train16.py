"""在 dataset/yolo16_final 上训练 16 类成熟度分类模型。

训练样本按标注框裁剪（外扩 25% 保留少量背景），与网页端“整图缩放”推理口径保持一致。
"""
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

BASE = Path(__file__).resolve().parent
DATA = BASE / "dataset" / "yolo16_final"
MODELS = BASE / "models"
CROP_DIR = BASE / "dataset" / "crops16"
IMG_SIZE = 224
MARGIN = 0.25          # 标注框外扩比例
EPOCHS = 22
BATCH = 32
SEED = 20261124
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_boxes(label_path: Path):
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append((cls, cx, cy, w, h))
    return boxes


def build_crops():
    """把每张图的标注框裁出来存成 jpg，返回样本索引。"""
    meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
    names = meta["class_names"]
    samples = {"train": [], "val": []}
    for split in ("train", "val"):
        out_dir = CROP_DIR / split
        for cls_id, cls_name in enumerate(names):
            (out_dir / cls_name).mkdir(parents=True, exist_ok=True)

        for lbl in sorted((DATA / "labels" / split).glob("*.txt")):
            for box in read_boxes(lbl):
                cls, cx, cy, w, h = box
                src = next((DATA / "images" / split / f"{lbl.stem}{ext}"
                            for ext in (".jpg", ".jpeg", ".png")
                            if (DATA / "images" / split / f"{lbl.stem}{ext}").exists()), None)
                if src is None:
                    continue
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    W, H = im.size
                    bw, bh = w * W, h * H
                    mx, my = bw * MARGIN, bh * MARGIN
                    x0 = max(0, int(cx * W - bw / 2 - mx))
                    y0 = max(0, int(cy * H - bh / 2 - my))
                    x1 = min(W, int(cx * W + bw / 2 + mx))
                    y1 = min(H, int(cy * H + bh / 2 + my))
                    if x1 - x0 < 16 or y1 - y0 < 16:
                        continue
                    crop = im.crop((x0, y0, x1, y1))
                    dst = out_dir / names[cls] / f"{lbl.stem}.jpg"
                    crop.save(dst, quality=92)
                samples[split].append({"path": str(dst), "cls": cls, "src": f"{split}/{src.name}"})
    (MODELS / "crop_index.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")
    return samples


class CropDS(Dataset):
    def __init__(self, rows, train: bool):
        self.rows = rows
        if train:
            self.tfms = transforms.Compose([
                transforms.RandomResizedCrop(IMG_SIZE, scale=(0.55, 1.0), ratio=(0.85, 1.18)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
                transforms.ColorJitter(brightness=0.12, contrast=0.10),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])
        else:
            self.tfms = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tfms(img), row["cls"]


def build_model(n_cls, hidden=512):
    model = models.mobilenet_v2(weights=None)
    state = torch.load(MODELS / "mobilenet_v2-b0353104.pth", map_location="cpu")
    model.load_state_dict(state)
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_cls),
    )
    return model


def main():
    set_seed()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    meta = json.loads((DATA / "dataset_meta.json").read_text(encoding="utf-8"))
    names, cn_names = meta["class_names"], meta["cn_names"]

    if not (CROP_DIR / "train").exists():
        print("裁剪训练样本…")
    samples = build_crops()
    tr_rows, va_rows = samples["train"], samples["val"]
    print(f"train crops {len(tr_rows)}  val crops {len(va_rows)}")

    tr_ds, va_ds = CropDS(tr_rows, True), CropDS(va_rows, False)
    counts = Counter(r["cls"] for r in tr_rows)
    weights = [1.0 / counts[r["cls"]] for r in tr_rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(tr_rows), replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH, sampler=sampler, num_workers=0, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model(len(names))
    head_params = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
    base_params = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": 2.5e-4},
        {"params": head_params, "lr": 2.0e-3},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for x, y in tr_loader:
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        sched.step()

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for x, y in va_loader:
                pred = model(x).argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        v_acc = v_correct / max(1, v_total)
        print(f"epoch {ep:2d}  loss {loss_sum/total:.3f}  train {correct/total*100:5.1f}%  "
              f"val {v_acc*100:5.1f}%  {time.time()-t0:5.0f}s")
        if v_acc > best_acc:
            best_acc = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # 最终评估：混淆矩阵 + 每类指标
    cm = np.zeros((len(names), len(names)), dtype=int)
    with torch.no_grad():
        for x, y in va_loader:
            for t, p in zip(y.tolist(), model(x).argmax(1).tolist()):
                cm[t][p] += 1
    per_class = {}
    for i, name in enumerate(names):
        tp = int(cm[i][i])
        support = int(cm[i].sum())
        prec = tp / max(1, int(cm[:, i].sum()))
        rec = tp / max(1, support)
        per_class[name] = {
            "cn": cn_names[i],
            "accuracy": round(rec, 4),
            "precision": round(prec, 4),
            "support": support,
        }
    overall = float(np.trace(cm) / max(1, cm.sum()))

    ckpt = {
        "state_dict": model.state_dict(),
        "classes": names,
        "class_cn": {n: c for n, c in zip(names, cn_names)},
        "img_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "input_dim": model.classifier[1].in_features,
        "hidden": model.classifier[1].out_features,
        "val_accuracy": round(overall, 4),
    }
    torch.save(ckpt, MODELS / "fruit16_mobilenetv2.pth")

    ev = {
        "accuracy": round(overall, 4),
        "train_samples": len(tr_rows),
        "val_samples": len(va_rows),
        "class_order": names,
        "class_cn": {n: c for n, c in zip(names, cn_names)},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "train_counts": {names[i]: counts.get(i, 0) for i in range(len(names))},
        "dataset": "dataset/yolo16_final",
        "epochs": EPOCHS,
    }
    (MODELS / "evaluation16.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nbest val acc {best_acc*100:.1f}%  saved models/fruit16_mobilenetv2.pth")
    for i, n in enumerate(names):
        print(f"  {n:14s} acc {per_class[n]['accuracy']*100:5.1f}%  n={per_class[n]['support']}")


if __name__ == "__main__":
    main()
