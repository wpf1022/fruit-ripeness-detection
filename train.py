import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = "dataset"
MODEL_DIR = "models"
WEIGHTS_PATH = os.path.join(MODEL_DIR, "mobilenet_v2-b0353104.pth")
META_PATH = os.path.join(MODEL_DIR, "dataset_meta.json")
CKPT_PATH = os.path.join(MODEL_DIR, "fruit_ripeness_mobilenetv2.pth")
EVAL_PATH = os.path.join(MODEL_DIR, "evaluation.json")

BATCH_SIZE = 64
IMG_SIZE = 224
EPOCHS = 40
LR = 0.002
HIDDEN = 256

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

with open(META_PATH, encoding="utf-8") as fh:
    META = json.load(fh)
CLASSES = META["class_names"]
CLASS_CN = META["class_cn"]

TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class FruitDataset(Dataset):
    def __init__(self, split):
        self.items = []
        for idx, name in enumerate(CLASSES):
            folder = os.path.join(DATA_DIR, split, name)
            for filename in sorted(os.listdir(folder)):
                if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.items.append((os.path.join(folder, filename), idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        image = Image.open(path).convert("RGB")
        return TRANSFORM(image), label


def load_backbone():
    model = models.mobilenet_v2(weights=None)
    if os.path.exists(WEIGHTS_PATH):
        state = torch.load(WEIGHTS_PATH, map_location="cpu")
        model.load_state_dict(state)
    return model


def extract_features(model, loader):
    feats, labels = [], []
    with torch.no_grad():
        for x, y, *_ in loader:
            f = model(x)
            feats.append(f.view(f.size(0), -1).numpy())
            labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)


def build_head(in_dim, hidden, n_classes):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(hidden, n_classes),
    )


def main():
    started = time.time()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    train_loader = DataLoader(FruitDataset("train"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(FruitDataset("val"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练样本 {len(train_loader.dataset)}，验证样本 {len(val_loader.dataset)}")

    backbone = load_backbone()
    backbone.classifier = nn.Identity()
    backbone.eval()
    print("提取图像特征...")
    x_train, y_train = extract_features(backbone, train_loader)
    x_val, y_val = extract_features(backbone, val_loader)
    in_dim = x_train.shape[1]
    print(f"特征维度 {in_dim}，提取耗时 {time.time()-started:.1f}s")

    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    xv = torch.tensor(x_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    head = build_head(in_dim, HIDDEN, len(CLASSES))
    optimizer = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None

    for epoch in range(1, EPOCHS + 1):
        head.train()
        perm = torch.randperm(len(xt))
        total_loss = 0.0
        for start in range(0, len(perm), 256):
            idx = perm[start:start + 256]
            optimizer.zero_grad()
            loss = criterion(head(xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        head.eval()
        with torch.no_grad():
            val_acc = (head(xv).argmax(1) == yv).float().mean().item()
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{EPOCHS} loss={total_loss/len(xt):.4f} val_acc={val_acc:.4f}")

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        preds = head(xv).argmax(1).numpy()
    cm = [[int(((y_val == i) & (preds == j)).sum()) for j in range(len(CLASSES))] for i in range(len(CLASSES))]
    per_class = {}
    for i, name in enumerate(CLASSES):
        total = int((y_val == i).sum())
        per_class[name] = {
            "cn": CLASS_CN[name],
            "accuracy": round(float((preds[y_val == i] == i).mean()), 4) if total else 0.0,
            "support": total,
        }
    accuracy = float((preds == y_val).mean())

    full_model = load_backbone()
    full_model.classifier = head
    torch.save({
        "state_dict": full_model.state_dict(),
        "classes": CLASSES,
        "class_cn": CLASS_CN,
        "input_dim": in_dim,
        "hidden": HIDDEN,
        "img_size": IMG_SIZE,
        "mean": MEAN,
        "std": STD,
    }, CKPT_PATH)

    evaluation = {
        "model": "MobileNetV2 + 分类头 (迁移学习)",
        "accuracy": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "class_order": CLASSES,
        "train_samples": int(len(xt)),
        "val_samples": int(len(yv)),
        "elapsed_seconds": round(time.time() - started, 1),
        "dataset_source": META["source"],
        "license": META["license"],
    }
    with open(EVAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(evaluation, fh, ensure_ascii=False, indent=2)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
