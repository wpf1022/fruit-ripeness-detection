"""颜色成熟度判定。

系统原来的三档标签（L/M/H）是按“框内颜色得分”的三分位切出来的，相邻两档本身
就高度重叠，靠统计切点复现只有 55% 一致率，还会把红苹果判成“未熟”。所以成熟度
不再死抠数据集标签，而用可解释的颜色语义：

    果实区域里绿多  → 未熟（青果）
    有褐斑/暗沉多  → 完熟偏过熟
    其余（着色充分）→ 成熟

难点是“在哪片像素上算颜色”。用户上传的是整图（背景可能有大片绿叶），直接整图
统计会把成熟草莓算成未熟。这里先按种类找到果实自身颜色的最大连通区域（纯 numpy
的粗网格 BFS，不依赖 scipy），再只在该区域统计颜色。

对外接口：
    analyse(img, species) -> dict(level, level_probs, features, bbox, ...)
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

LEVELS = ["L", "M", "H"]
LEVEL_CN = {"L": "未熟", "M": "成熟", "H": "完熟"}
LEVEL_HINT = {
    "L": "着色不足，偏生，建议继续放置或催熟",
    "M": "着色与风味处于适宜食用区间",
    "H": "着色充分、出现褐斑，已接近或超过最佳食用期",
}

# 找果实时使用的特征色
_DETECT = {
    "apple": ("red", "orange", "yellow", "green"),
    "banana": ("yellow", "orange"),
    "grape": ("purple", "dark"),
    "orange": ("orange", "yellow"),
    "strawberry": ("red", "orange"),
}

# 绿多 → 未熟；褐斑多 → 完熟（阈值按种类微调，因为各果实的“正常底色”不同）
GREEN_T = {"apple": 0.40, "banana": 0.35, "grape": 0.42, "orange": 0.38, "strawberry": 0.30}
BROWN_T = {"apple": 0.10, "banana": 0.18, "grape": 0.10, "orange": 0.12, "strawberry": 0.10}


def _hsv(a: np.ndarray):
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx, mn = a.max(2), a.min(2)
    v = mx
    s = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    d = mx - mn + 1e-6
    hue = np.where(mx == r, ((g - b) / d) % 6,
                   np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return hue, s, v


def _masks(hue, s, v):
    return {
        "green": (hue > 70) & (hue < 175) & (s > 0.28) & (v > 0.15),
        "yellow": (hue >= 20) & (hue <= 70) & (s > 0.38) & (v > 0.30),
        "orange": (hue >= 8) & (hue < 25) & (s > 0.50) & (v > 0.32),
        "red": ((hue < 12) | (hue > 340)) & (s > 0.40) & (v > 0.18),
        "purple": (hue >= 250) & (hue <= 340) & (s > 0.28) & (v > 0.10),
        # 褐斑：香蕉/苹果的过熟斑点，暗黄褐色，明显区别于紫色与纯红
        "brown": (hue >= 12) & (hue <= 50) & (s > 0.25) & (v > 0.12) & (v < 0.45),
        "dark": v < 0.33,
        "white": (s < 0.16) & (v > 0.72),
    }


def colour_features(rgb01: np.ndarray) -> dict:
    if rgb01.size == 0:
        rgb01 = np.zeros((1, 1, 3), dtype=np.float32)
    hue, s, v = _hsv(rgb01)
    m = _masks(hue, s, v)
    return {k: float(m[k].mean()) for k in
            ("green", "yellow", "orange", "red", "purple", "brown", "dark", "white")} | {
        "sat": float(s.mean())}


def ripeness_score(f: dict) -> float:
    """0~1 的“着色度”，仅用于展示解释，不参与分档。"""
    return float(np.clip(0.85 * (f["red"] + f["orange"] + 0.7 * f["yellow"] + 0.6 * f["purple"])
                         + 0.15 * (1 - f["green"]), 0, 1))


def _largest_cluster(cell_mask: np.ndarray) -> np.ndarray:
    h, w = cell_mask.shape
    seen = np.zeros_like(cell_mask, dtype=bool)
    best = []
    for y0 in range(h):
        for x0 in range(w):
            if not cell_mask[y0, x0] or seen[y0, x0]:
                continue
            q = deque([(y0, x0)])
            seen[y0, x0] = True
            comp = []
            while q:
                y, x = q.popleft()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and cell_mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if len(comp) > len(best):
                best = comp
    out = np.zeros_like(cell_mask, dtype=bool)
    for y, x in best:
        out[y, x] = True
    return out


def fruit_bbox(img: Image.Image, species: str, grid: int = 24):
    """返回 (框, 是否可靠定位)。定位不到特征色时回退中心区域并标记不可靠。"""
    small = img.convert("RGB").resize((grid * 4, grid * 4), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32) / 255.0
    hue, s, v = _hsv(a)
    m = _masks(hue, s, v)
    mask = np.zeros(a.shape[:2], dtype=bool)
    for key in _DETECT.get(species, ("red", "orange", "yellow")):
        mask |= m[key]

    cells = mask.reshape(grid, 4, grid, 4).mean(axis=(1, 3)) > 0.45
    if cells.mean() > 0.85:
        return (0, 0, img.width, img.height), True
    if cells.sum() < 4:      # 青果/白底：没有明显果实色
        w, h = img.size
        return (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)), False

    ys, xs = np.nonzero(_largest_cluster(cells))
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sx, sy = img.width / grid, img.height / grid
    box = (max(0, int((x0 - 0.5) * sx)), max(0, int((y0 - 0.5) * sy)),
           min(img.width, int((x1 + 0.5) * sx)), min(img.height, int((y1 + 0.5) * sy)))
    area = (box[2] - box[0]) * (box[3] - box[1]) / max(1, img.width * img.height)
    return box, area > 0.015


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def level_probs(f: dict, species: str) -> dict:
    """绿→未熟、褐斑→完熟，两个软判据 + 其余为成熟。"""
    gt = GREEN_T.get(species, 0.38)
    bt = BROWN_T.get(species, 0.12)
    under = _sigmoid((f["green"] - gt) / 0.06)          # 越绿越可能是未熟
    over = _sigmoid((f["brown"] - bt) / 0.035)          # 褐斑越多越可能是完熟
    p_l = under * (1 - over)
    p_h = over * (1 - under)
    p_m = max(0.0, 1.0 - p_l - p_h)
    tot = p_l + p_m + p_h
    return {"L": round(p_l / tot, 4), "M": round(p_m / tot, 4), "H": round(p_h / tot, 4)}


def fruit_region(img: Image.Image, grid: int = 24,
                 min_area: float = 0.02, max_area: float = 0.65):
    """不依赖种类的“果实主体”定位：红/橙/黄/紫的大块连通区域。

    用于判种类前的聚焦：田间草莓照里果实只占很小一块，整图缩放会让模型看到
    大片绿叶而认成葡萄；先框出果实主体再判，草莓就能判对。区域太大（整幅都是
    果实）或太小（噪点）时返回 None，调用方回退整图。
    """
    small = img.convert("RGB").resize((grid * 4, grid * 4), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32) / 255.0
    hue, s, v = _hsv(a)
    m = _masks(hue, s, v)
    mask = m["red"] | m["orange"] | m["yellow"] | m["purple"]
    cells = mask.reshape(grid, 4, grid, 4).mean(axis=(1, 3)) > 0.40
    if cells.sum() < 4:
        return None
    ys, xs = np.nonzero(_largest_cluster(cells))
    sx, sy = img.width / grid, img.height / grid
    box = (max(0, int((xs.min() - 1) * sx)), max(0, int((ys.min() - 1) * sy)),
           min(img.width, int((xs.max() + 2) * sx)), min(img.height, int((ys.max() + 2) * sy)))
    if box[2] - box[0] < 24 or box[3] - box[1] < 24:
        return None
    area = (box[2] - box[0]) * (box[3] - box[1]) / max(1, img.width * img.height)
    return box if min_area <= area <= max_area else None


def analyse(img: Image.Image, species: str, region=None) -> dict:
    """region 为外部（种类判定阶段）已定位的果实框，传进来可让颜色统计与种类判定同一口径。"""
    rgb = img.convert("RGB")
    if region:
        box, reliable = region, True
    else:
        box, reliable = fruit_bbox(rgb, species)
    feats = colour_features(np.asarray(rgb.crop(box), dtype=np.float32) / 255.0)
    probs = level_probs(feats, species)
    level = max(probs, key=probs.get)
    return {
        "level": level,
        "level_cn": LEVEL_CN[level],
        "level_probs": probs,
        "level_hint": LEVEL_HINT[level],
        "score": round(ripeness_score(feats), 4),
        "features": {k: round(v, 4) for k, v in feats.items()},
        "bbox": [int(v) for v in box],
        "region_reliable": bool(reliable),
    }
