"""训练“水果种类”分类模型（6 类：苹果/香蕉/葡萄/橙子/草莓/腐烂水果）。

为什么不直接训 16 类：dataset/yolo16_final 的三档成熟度标签是按颜色分位切出来
的，相邻两档本身高度重叠，硬学 16 类会把误差全压在“未熟↔成熟”上（实测 68.7%，
错误几乎全发生在同一水果的相邻档位）。成熟度改由 ripeness.py 的颜色语义规则
判定，模型只负责“是什么水果”，每类样本量因此翻三倍，种类准确率显著提高。

训练输入：标注框外扩 25% 的裁剪（保留少量背景）；增强模拟用户实拍（缩放/旋转/
亮度对比度饱和度抖动/随机擦除）；验证集用 TTA（原图 + 中心 0.8 裁剪 + 水平翻转）平均。

输出：models/species_mobilenetv2.pth、models/evaluation_species.json
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
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
IMG_SIZE = 224
MARGIN = 0.25
EPOCHS = 30
BATCH = 32
SEED = 20261124
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

SPECIES = ["apple", "banana", "grape", "orange", "strawberry", "bad fruit"]
SPECIES_CN = {"apple": "苹果", "banana": "香蕉", "grape": "葡萄",
              "orange": "橙子", "strawberry": "草莓", "bad fruit": "腐烂水果"}
# 16 类 → 6 类
CLS16 = [f"{s}-{lv}" for s in SPECIES[:5] for lv in "LMH"] + ["bad fruit"]
MAP16 = {name: (SPECIES.index(name.split("-")[0]) if name != "bad fruit" else 5)
         for name in CLS16}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_box(label_path: Path):
    best = None
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        if best is None or w * h > best[1] * best[2]:
            best = (cls, cx, cy, w, h)
    return best


def build_index():
    """返回 [(图片路径, 6 类标签, 外扩裁剪框)]，按 8:2 保留数据集的 train/val 划分。"""
    rows = {"train": [], "val": []}
    for split in ("train", "val"):
        for lbl in sorted((DATA / "labels" / split).glob("*.txt")):
            box = read_box(lbl)
            if box is None:
                continue
            src = DATA / "images" / split / f"{lbl.stem}.jpg"
            if not src.exists():
                continue
            cls = MAP16[CLS16[box[0]]]
            rows[split].append({"path": str(src), "cls": cls, "box": box[1:]})
    return rows


def crop_margin(img: Image.Image, box):
    cx, cy, w, h = box
    W, H = img.size
    bw, bh = w * W, h * H
    mx, my = bw * MARGIN, bh * MARGIN
    x0 = max(0, int(cx * W - bw / 2 - mx))
    y0 = max(0, int(cy * H - bh / 2 - my))
    x1 = min(W, int(cx * W + bw / 2 + mx))
    y1 = min(H, int(cy * H + bh / 2 + my))
    if x1 - x0 < 16 or y1 - y0 < 16:
        x0, y0, x1, y1 = 0, 0, W, H
    return img.crop((x0, y0, x1, y1))


TRAIN_TFMS = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.55, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.22, saturation=0.25, hue=0.03),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08)),
    transforms.Normalize(MEAN, STD),
])

EVAL_TFMS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class FruitDS(Dataset):
    def __init__(self, rows, train: bool):
        self.rows, self.train = rows, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(row["path"]) as im:
            img = crop_margin(im.convert("RGB"), row["box"])
        if self.train:
            return TRAIN_TFMS(img), row["cls"]
        return EVAL_TFMS(img), row["cls"]


def build_model(n_cls, hidden=512):
    model = models.mobilenet_v2(weights=None)
    model.load_state_dict(torch.load(MODELS / "mobilenet_v2-b0353104.pth", map_location="cpu"))
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, hidden),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden, n_cls),
    )
    return model


@torch.no_grad()
def tta_probs(model, imgs: list[Image.Image]) -> torch.Tensor:
    """整图 + 中心 0.8 裁剪 + 水平翻转，三种视图概率平均。"""
    views = []
    for img in imgs:
        img = img.convert("RGB")
        w, h = img.size
        s = int(min(w, h) * 0.8)
        center = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        views += [img, center, img.transpose(Image.FLIP_LEFT_RIGHT)]
    x = torch.stack([EVAL_TFMS(v) for v in views])
    p = torch.softmax(model(x), 1)
    return p.view(len(imgs), 3, -1).mean(1)


def main():
    set_seed()
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    rows = build_index()
    tr_rows, va_rows = rows["train"], rows["val"]
    print(f"train {len(tr_rows)}  val {len(va_rows)}  每类: {Counter(r['cls'] for r in tr_rows)}")

    tr_ds, va_ds = FruitDS(tr_rows, True), FruitDS(va_rows, False)
    counts = Counter(r["cls"] for r in tr_rows)
    weights = [1.0 / counts[r["cls"]] for r in tr_rows]
    tr_loader = DataLoader(tr_ds, batch_size=BATCH,
                           sampler=WeightedRandomSampler(weights, len(tr_rows), replacement=True),
                           num_workers=0, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    model = build_model(len(SPECIES))
    head = [p for n, p in model.named_parameters() if n.startswith("classifier.")]
    body = [p for n, p in model.named_parameters() if not n.startswith("classifier.")]
    opt = torch.optim.AdamW([{"params": body, "lr": 2.5e-4},
                             {"params": head, "lr": 2.0e-3}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc, best_state = -1.0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        loss_sum = correct = total = 0
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
                v_correct += (model(x).argmax(1) == y).sum().item()
                v_total += y.size(0)
        acc = v_correct / max(1, v_total)
        print(f"epoch {ep:2d}  loss {loss_sum/total:.3f}  train {correct/total*100:5.1f}%  "
              f"val {acc*100:5.1f}%  {time.time()-t0:4.0f}s")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # 验证集：普通推理 + TTA
    cm = np.zeros((len(SPECIES), len(SPECIES)), dtype=int)
    tta_hit = 0
    imgs, labels = [], []
    for row in va_rows:
        with Image.open(row["path"]) as im:
            imgs.append(crop_margin(im.convert("RGB"), row["box"]))
        labels.append(row["cls"])
    plain_imgs = [EVAL_TFMS(i) for i in imgs]
    with torch.no_grad():
        plain = torch.softmax(model(torch.stack(plain_imgs)), 1).argmax(1).tolist()
    for t, p in zip(labels, plain):
        cm[t][p] += 1
    for start in range(0, len(imgs), 32):
        p = tta_probs(model, imgs[start:start + 32]).argmax(1).tolist()
        tta_hit += sum(int(a == b) for a, b in zip(p, labels[start:start + 32]))

    per_class = {}
    for i, name in enumerate(SPECIES):
        tp = int(cm[i][i])
        support = int(cm[i].sum())
        per_class[name] = {
            "cn": SPECIES_CN[name],
            "accuracy": round(tp / max(1, support), 4),
            "precision": round(tp / max(1, int(cm[:, i].sum())), 4),
            "support": support,
        }
    overall = float(np.trace(cm) / max(1, cm.sum()))

    ckpt = {
        "state_dict": model.state_dict(),
        "classes": SPECIES,
        "class_cn": SPECIES_CN,
        "img_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
        "input_dim": model.classifier[1].in_features,
        "hidden": model.classifier[1].out_features,
        "val_accuracy": round(overall, 4),
    }
    torch.save(ckpt, MODELS / "species_mobilenetv2.pth")

    ev = {
        "accuracy": round(overall, 4),
        "tta_accuracy": round(tta_hit / max(1, len(labels)), 4),
        "train_samples": len(tr_rows),
        "val_samples": len(va_rows),
        "class_order": SPECIES,
        "class_cn": SPECIES_CN,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "train_counts": {SPECIES[i]: counts.get(i, 0) for i in range(len(SPECIES))},
        "dataset": "dataset/yolo16_final（三档合并为种类）",
        "epochs": EPOCHS,
        "note": "成熟度不在此模型内判定，由 ripeness.py 的颜色语义规则给出",
    }
    (MODELS / "evaluation_species.json").write_text(json.dumps(ev, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(f"\n最佳验证准确率 {best_acc*100:.1f}%    TTA {ev['tta_accuracy']*100:.1f}%")
    for name in SPECIES:
        print(f"  {name:11s} acc {per_class[name]['accuracy']*100:5.1f}%  n={per_class[name]['support']}")


if __name__ == "__main__":
    main()
