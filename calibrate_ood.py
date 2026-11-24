"""校准“未知类别拒绝”判据。

MobileNetV2 演示模型只有 4 个输出类（未熟/成熟香蕉、未熟/成熟芒果），
softmax 会把苹果等图片也强行归到其中一类。这里用 backbone 特征空间的
两种分数做 OOD（out-of-distribution）检测：

1. 每类马氏距离：到各训练类高斯分布的最小距离；
2. 近邻相似度：与训练特征最近邻的余弦相似度（值越大越像未知类）。

两个阈值都按验证集 0 误拒校准，任一分数超阈值即拒绝。校准结果保存到
models/ood_stats.npz，供 predict.py 推理时使用。

用法：
    python calibrate_ood.py
    python calibrate_ood.py --ood-dir D:/水果1
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

BASE = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE, "models", "fruit_ripeness_mobilenetv2.pth")
OUT_PATH = os.path.join(BASE, "models", "ood_stats.npz")
SAMPLE_DIR = os.path.join(BASE, "demo_samples")
SHRINKAGE = 0.05
PROTOTYPES_PER_CLASS = 400
SEED = 42


def load_backbone():
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model = models.mobilenet_v2(weights=None)
    head = nn.Sequential(
        nn.Linear(ckpt["input_dim"], ckpt["hidden"]),
        nn.ReLU(),
        nn.Dropout(0.25),
        nn.Linear(ckpt["hidden"], len(ckpt["classes"])),
    )
    model.classifier = head
    model.load_state_dict(ckpt["state_dict"])
    backbone = models.mobilenet_v2(weights=None)
    backbone.classifier = nn.Identity()
    backbone.load_state_dict({k: v for k, v in ckpt["state_dict"].items()
                              if not k.startswith("classifier.")})
    backbone.eval()
    tfms = transforms.Compose([
        transforms.Resize((ckpt["img_size"], ckpt["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(ckpt["mean"], ckpt["std"]),
    ])
    return backbone, tfms, ckpt


def extract(paths, model, tfms, batch_size=64):
    features = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = [Image.open(p).convert("RGB") for p in paths[start:start + batch_size]]
            x = torch.stack([tfms(im) for im in images])
            features.append(model(x).numpy())
    return np.vstack(features)


def collect_split(split, classes):
    paths, labels = [], []
    for idx, name in enumerate(classes):
        found = sorted(glob.glob(os.path.join(BASE, "dataset", split, name, "*.*")))
        found = [p for p in found if p.lower().endswith((".jpg", ".jpeg", ".png"))]
        paths.extend(found)
        labels.extend([idx] * len(found))
    return paths, np.asarray(labels, dtype=np.int64)


def collect_ood(extra_dirs):
    """仓库内置演示图中非香蕉/芒果的样本 + 可选的外部目录。"""
    paths = []
    for directory in [SAMPLE_DIR] + list(extra_dirs):
        for p in sorted(glob.glob(os.path.join(directory, "*.*"))):
            name = os.path.basename(p).lower()
            if p.lower().endswith((".jpg", ".jpeg", ".png")) and "banana" not in name:
                paths.append(p)
    return sorted(set(paths))


def build_detector(x_train, y_train):
    classes = int(y_train.max()) + 1
    means = np.stack([x_train[y_train == i].mean(axis=0) for i in range(classes)])
    precisions = []
    for i in range(classes):
        centered = x_train[y_train == i] - means[i]
        cov = centered.T @ centered / len(centered)
        cov += np.eye(cov.shape[0]) * (np.trace(cov) / cov.shape[0]) * SHRINKAGE
        precisions.append(np.linalg.inv(cov))

    rng = np.random.default_rng(SEED)
    keep = np.concatenate([
        rng.choice(np.where(y_train == i)[0],
                   size=min(PROTOTYPES_PER_CLASS, int((y_train == i).sum())),
                   replace=False)
        for i in range(classes)
    ])
    prototypes = x_train[keep]
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

    def scores(features):
        maha = np.stack([
            np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", features - m, p, features - m), 0))
            for m, p in zip(means, precisions)
        ], axis=1).min(axis=1)
        normalized = features / np.linalg.norm(features, axis=1, keepdims=True)
        knn = -(normalized @ prototypes.T).max(axis=1)
        return maha, knn

    return means, np.stack(precisions), prototypes, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ood-dir", action="append", default=[],
                        help="额外的 OOD 图片目录（可重复传入），会排除文件名含 banana 的图片")
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    backbone, tfms, ckpt = load_backbone()
    classes = ckpt["classes"]

    train_paths, train_y = collect_split("train", classes)
    val_paths, _ = collect_split("val", classes)
    ood_paths = collect_ood(args.ood_dir)
    if not train_paths or not val_paths or not ood_paths:
        raise SystemExit("训练/验证/OOD 图片不足，无法校准")

    print(f"训练 {len(train_paths)} 张，验证 {len(val_paths)} 张，OOD 校准 {len(ood_paths)} 张")
    x_train = extract(train_paths, backbone, tfms)
    x_val = extract(val_paths, backbone, tfms)
    x_ood = extract(ood_paths, backbone, tfms)

    means, precisions, prototypes, scores = build_detector(x_train, train_y)
    m_val, k_val = scores(x_val)
    m_ood, k_ood = scores(x_ood)

    # 两个分数都按“验证集 0 误拒”设阈值，并留一点余量
    m_thr = max(float(np.quantile(m_val, 0.999)), float(m_val.max()) + 0.5)
    k_thr = max(float(np.quantile(k_val, 0.999)), float(k_val.max()) + 0.005)

    np.savez_compressed(
        OUT_PATH,
        means=means.astype(np.float32),
        # 半精度存储即可，实测与 float32 判据差异 < 0.01
        precisions=precisions.astype(np.float16),
        prototypes=prototypes.astype(np.float16),
        classes=np.array(classes),
        maha_threshold=np.float32(m_thr),
        knn_threshold=np.float32(k_thr),
        val_maha_max=np.float32(m_val.max()),
        val_knn_max=np.float32(k_val.max()),
        ood_maha_min=np.float32(m_ood.min()),
        ood_knn_min=np.float32(k_ood.min()),
        shrinkage=np.float32(SHRINKAGE),
    )

    rejected = (m_ood > m_thr) | (k_ood > k_thr)
    print(f"马氏距离阈值 {m_thr:.2f}，近邻分数阈值 {k_thr:.4f}")
    print(f"验证集误拒率: {((m_val > m_thr) | (k_val > k_thr)).mean() * 100:.2f}%")
    print(f"OOD 拦截率: {rejected.mean() * 100:.1f}% ({int(rejected.sum())}/{len(ood_paths)})")
    print(f"已保存 → {OUT_PATH}")


if __name__ == "__main__":
    main()
