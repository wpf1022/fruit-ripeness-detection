"""训练集标注查看器。

只读展示 dataset/yolo16 下的图片与 YOLO 标注：
- 训练集 / 验证集切换
- 按类别筛选、文件名搜索、只看多目标
- 缩略图叠加标注框，点开看大图与框列表

启动：
    python label_viewer.py
浏览器打开 http://127.0.0.1:5003
"""

import io
import json
import os
import random
from functools import lru_cache

from flask import Flask, jsonify, render_template_string, request, send_file, send_from_directory
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATASET_DIR") or os.path.join(BASE, "dataset", "yolo16_final")
IMG_DIR = os.path.join(DATA_DIR, "images")
LBL_DIR = os.path.join(DATA_DIR, "labels")
META_PATH = os.path.join(DATA_DIR, "dataset_meta.json")
SPLITS = ("train", "val")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

with open(META_PATH, encoding="utf-8") as fh:
    META = json.load(fh)
CLASS_NAMES = META["class_names"]
CLASS_CN = META["cn_names"]
COLORS = [
    "#dc2626", "#ef4444", "#f87171",
    "#ca8a04", "#eab308", "#facc15",
    "#6d28d9", "#8b5cf6", "#c084fc",
    "#be123c", "#e11d48", "#fb7185",
    "#c2410c", "#ea580c", "#fb923c",
    "#334155",
]

app = Flask(__name__)


def parse_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, w, h = (float(v) for v in parts[1:5])
            except ValueError:
                continue
            if 0 <= cls < len(CLASS_NAMES):
                boxes.append([cls, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)])
    return boxes


def build_index():
    items = []
    for split in SPLITS:
        image_dir = os.path.join(IMG_DIR, split)
        label_dir = os.path.join(LBL_DIR, split)
        for name in sorted(os.listdir(image_dir)):
            if not name.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(name)[0]
            boxes = parse_label(os.path.join(label_dir, stem + ".txt"))
            items.append({
                "split": split,
                "name": name,
                "boxes": boxes,
                "classes": sorted({b[0] for b in boxes}),
            })
    # 固定种子打散，首屏能同时看到多种水果，而不是整页柿子
    random.Random(42).shuffle(items)
    return items


INDEX = build_index()
CLASS_COUNTS = {
    split: [0] * len(CLASS_NAMES)
    for split in ("all",) + SPLITS
}
for item in INDEX:
    for box in item["boxes"]:
        CLASS_COUNTS["all"][box[0]] += 1
        CLASS_COUNTS[item["split"]][box[0]] += 1

STATS = {
    "images": len(INDEX),
    "train": sum(1 for x in INDEX if x["split"] == "train"),
    "val": sum(1 for x in INDEX if x["split"] == "val"),
    "boxes": sum(len(x["boxes"]) for x in INDEX),
    "empty": sum(1 for x in INDEX if not x["boxes"]),
    "multi": sum(1 for x in INDEX if len(x["boxes"]) > 1),
    "classes": len(CLASS_NAMES),
}


def public_item(item):
    return {
        "split": item["split"],
        "name": item["name"],
        "boxes": item["boxes"],
        "classes": item["classes"],
        "count": len(item["boxes"]),
        "url": f"/media/{item['split']}/{item['name']}",
        "thumb": f"/thumb/{item['split']}/{item['name']}?w=560",
    }


def filtered_items():
    split = request.args.get("split", "all")
    query = request.args.get("q", "").strip().lower()
    cls = request.args.get("cls", "").strip()
    multi = request.args.get("multi", "0") == "1"

    items = INDEX
    if split in SPLITS:
        items = [x for x in items if x["split"] == split]
    if query:
        matched = {
            i for i, name in enumerate(CLASS_NAMES)
            if query in name.lower() or query in CLASS_CN[i].lower()
        }
        items = [
            x for x in items
            if query in x["name"].lower() or matched.intersection(x["classes"])
        ]
    if cls.isdigit():
        target = int(cls)
        items = [x for x in items if target in x["classes"]]
    if multi:
        items = [x for x in items if len(x["boxes"]) > 1]
    return items


@app.route("/api/items")
def api_items():
    items = filtered_items()
    try:
        page = max(1, int(request.args.get("page", 1)))
        per = min(60, max(6, int(request.args.get("per", 24))))
    except ValueError:
        page, per = 1, 24
    total = len(items)
    pages = max(1, (total + per - 1) // per)
    page = min(page, pages)
    start = (page - 1) * per
    return jsonify({
        "total": total,
        "page": page,
        "pages": pages,
        "per": per,
        "items": [public_item(x) for x in items[start:start + per]],
    })


@app.route("/api/gallery")
def api_gallery():
    n = int(request.args.get("n", 4))
    groups = {cid: [] for cid in range(len(CLASS_NAMES))}
    for it in INDEX:
        for cid in it["classes"]:
            if len(groups[cid]) < n:
                groups[cid].append(it)
    out = []
    for cid in range(len(CLASS_NAMES)):
        out.append({
            "cls": cid,
            "name": CLASS_CN[cid],
            "code": CLASS_NAMES[cid],
            "color": COLORS[cid],
            "count": CLASS_COUNTS["all"][cid],
            "images": [{"split": x["split"], "name": x["name"], "boxes": x["boxes"]}
                       for x in groups[cid]],
        })
    return jsonify(out)


@app.route("/api/random")
def api_random():
    items = filtered_items()
    if not items:
        return jsonify(error="没有符合条件的图片"), 404
    return jsonify(public_item(random.choice(items)))


@lru_cache(maxsize=600)
def thumbnail(path, width):
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, 10000))
    buf = io.BytesIO()
    image.save(buf, "JPEG", quality=85, optimize=True)
    return buf.getvalue()


def safe_image_path(split, filename):
    if split not in SPLITS:
        return None
    name = os.path.basename(filename)
    path = os.path.join(IMG_DIR, split, name)
    return path if os.path.exists(path) else None


@app.route("/thumb/<split>/<path:filename>")
def thumb(split, filename):
    path = safe_image_path(split, filename)
    if not path:
        return jsonify(error="图片不存在"), 404
    try:
        width = min(1200, max(200, int(request.args.get("w", 560))))
    except ValueError:
        width = 560
    data = thumbnail(path, width)
    response = send_file(io.BytesIO(data), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "public, max-age=86400"
    return response


@app.route("/media/<split>/<path:filename>")
def media(split, filename):
    if split not in SPLITS:
        return jsonify(error="数据集不存在"), 404
    return send_from_directory(os.path.join(IMG_DIR, split), os.path.basename(filename))


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>训练集标注查看器 · 水果成熟度检测</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--line:#e5e7eb;--text:#1f2329;--muted:#6b7280;--blue:#2563eb;--blue-bg:#eff6ff;--amber:#d97706}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.55}
.wrap{max-width:1440px;margin:0 auto;padding:26px 22px 60px}
header{display:flex;justify-content:space-between;align-items:flex-end;gap:18px;flex-wrap:wrap;margin-bottom:18px}
h1{font-size:25px;margin:0;font-weight:650;letter-spacing:-.3px}
.sub{color:var(--muted);font-size:13px;margin-top:5px}
.stats{display:flex;gap:10px;flex-wrap:wrap}
.stat{background:#fff;border:1px solid var(--line);border-radius:10px;padding:7px 12px;font-size:12px;color:var(--muted)}
.stat b{display:block;font-size:18px;color:var(--text);font-variant-numeric:tabular-nums;line-height:1.2}
.layout{display:grid;grid-template-columns:250px 1fr;gap:18px;align-items:start}
.panel{background:#fff;border:1px solid var(--line);border-radius:14px;padding:16px}
.panel h3{font-size:12px;color:var(--muted);font-weight:600;margin:0 0 9px;letter-spacing:.04em}
.seg{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:18px}
.seg button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:7px 0;font:inherit;font-size:13px;cursor:pointer;color:var(--text)}
.seg button.on{background:var(--blue);border-color:var(--blue);color:#fff}
.gal{margin-bottom:18px}
.gal-head{display:flex;align-items:baseline;gap:12px;margin:0 0 10px}
.gal-head h2{font-size:15px;font-weight:650;margin:0;color:#0f172a}
.gal-head .hint{font-size:12px;color:#94a3b8}
.gal-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px}
@media(max-width:1100px){.gal-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.gcard{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:8px;cursor:pointer;transition:.15s}
.gcard:hover{border-color:#94a3b8;box-shadow:0 4px 14px rgba(15,23,42,.08);transform:translateY(-1px)}
.gcard .gtop{display:flex;align-items:center;gap:6px;margin-bottom:6px}
.gcard .gname{font-size:12.5px;font-weight:600;color:#0f172a}
.gcard .gcount{margin-left:auto;font-size:11px;color:#64748b;font-variant-numeric:tabular-nums}
.gcard .gshots{display:grid;grid-template-columns:repeat(4,1fr);gap:4px}
.gcard .gshot{position:relative;aspect-ratio:1/1;background:#eef0f3;border-radius:5px;overflow:hidden}
.gcard .gshot img{width:100%;height:100%;object-fit:cover;display:block}
.classList{display:flex;flex-direction:column;gap:2px;margin:0 -6px;padding:0 6px}
.cls{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:7px;cursor:pointer;font-size:13px;border:1px solid transparent}
.cls:hover{background:#f8fafc}
.cls.on{background:var(--blue-bg);border-color:#bfdbfe}
.dot{width:9px;height:9px;border-radius:3px;flex:none}
.cls .n{margin-left:auto;color:var(--muted);font-size:12px;font-variant-numeric:tabular-nums}
.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
.search{flex:1;min-width:200px;display:flex;align-items:center;gap:8px;background:#fff;border:1px solid var(--line);border-radius:9px;padding:8px 11px}
.search input{border:0;outline:0;font:inherit;font-size:13px;width:100%;background:transparent;color:var(--text)}
.btn{border:1px solid var(--line);background:#fff;border-radius:9px;padding:8px 13px;font:inherit;font-size:13px;cursor:pointer;color:var(--text);white-space:nowrap}
.btn:hover{border-color:var(--blue);color:var(--blue)}
.btn.primary{background:var(--blue);border-color:var(--blue);color:#fff}
.chk{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--muted);cursor:pointer;user-select:none;white-space:nowrap}
.chk input{accent-color:var(--blue)}
.result-line{display:flex;justify-content:space-between;align-items:center;font-size:12px;color:var(--muted);margin:0 2px 10px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(228px,1fr));gap:14px}
.card{background:#fff;border:1px solid var(--line);border-radius:12px;overflow:hidden;cursor:zoom-in;transition:.15s}
.card:hover{transform:translateY(-2px);box-shadow:0 8px 22px rgba(15,23,42,.08);border-color:#cbd5e1}
.thumb{position:relative;background:#eef0f3;aspect-ratio:4/3;overflow:hidden;display:flex;align-items:center;justify-content:center}
.thumb img{max-width:100%;max-height:100%;width:auto;height:auto;display:block}
.thumb canvas{position:absolute;pointer-events:none}
.card-body{padding:9px 11px 10px}
.fname{font-size:12px;font-weight:550;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:6px}
.tags{display:flex;gap:4px;flex-wrap:wrap}
.tag{font-size:10px;line-height:1;padding:4px 6px;border-radius:5px;color:#fff;white-space:nowrap}
.count{margin-left:auto;font-size:11px;color:var(--muted)}
.card-foot{display:flex;align-items:center;gap:5px;margin-top:7px}
.pager{display:flex;justify-content:center;align-items:center;gap:6px;margin-top:22px;flex-wrap:wrap}
.pager button{min-width:34px;height:34px;padding:0 10px;border:1px solid var(--line);background:#fff;border-radius:8px;font:inherit;font-size:13px;cursor:pointer;color:var(--text)}
.pager button.on{background:var(--blue);border-color:var(--blue);color:#fff}
.pager button:disabled{opacity:.45;cursor:default}
.empty{padding:60px 20px;text-align:center;color:var(--muted);background:#fff;border:1px dashed var(--line);border-radius:12px}
.modal{position:fixed;inset:0;background:rgba(15,23,42,.66);display:none;align-items:center;justify-content:center;padding:24px;z-index:50}
.modal.on{display:flex}
.modal-box{background:#fff;border-radius:14px;max-width:1180px;width:100%;max-height:90vh;display:grid;grid-template-columns:1fr 280px;overflow:hidden}
.modal-stage{background:#0f172a;display:flex;align-items:center;justify-content:center;min-height:300px;position:relative;overflow:hidden}
.modal-stage img{max-width:100%;max-height:88vh;width:auto;height:auto;display:block}
.modal-stage canvas{position:absolute;pointer-events:none}
.modal-side{padding:18px;overflow:auto;border-left:1px solid var(--line)}
.modal-side h2{font-size:15px;margin:0 0 4px;word-break:break-all}
.modal-side .meta{font-size:12px;color:var(--muted);margin-bottom:14px}
.box-row{display:flex;align-items:center;gap:8px;padding:7px 0;border-bottom:1px solid #f1f5f9;font-size:12px}
.box-row:last-child{border:0}
.num-badge{width:18px;height:18px;border-radius:50%;color:#fff;font-size:10px;font-weight:700;display:inline-flex;align-items:center;justify-content:center;flex:none}
.coords{margin-left:auto;color:#94a3b8;font-size:10px;font-variant-numeric:tabular-nums;white-space:nowrap}
.close{position:absolute;top:12px;right:12px;z-index:3;background:rgba(255,255,255,.92);border:0;border-radius:50%;width:32px;height:32px;font-size:18px;cursor:pointer;color:#0f172a}
.legend{font-size:11px;color:var(--muted);margin-top:12px;line-height:1.7}
@media(max-width:900px){.layout{grid-template-columns:1fr}.panel.side{order:2}.modal-box{grid-template-columns:1fr}.modal-side{border-left:0;border-top:1px solid var(--line);max-height:32vh}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <h1>训练集标注查看器</h1>
    <div class="sub">16 类水果成熟度检测 · 数据集 {{ dataset_name }} · 只读预览</div>
  </div>
  <div class="stats">
    <div class="stat">图片总数<b>{{ stats.images }}</b></div>
    <div class="stat">训练 / 验证<b>{{ stats.train }} / {{ stats.val }}</b></div>
    <div class="stat">标注框<b>{{ stats.boxes }}</b></div>
    <div class="stat">多目标图<b>{{ stats.multi }}</b></div>
    <div class="stat">无标注<b>{{ stats.empty }}</b></div>
  </div>
</header>

<section class="gal">
  <div class="gal-head">
    <h2>16 类实例总览</h2>
    <div class="hint">每类都有真实样例 · 点击卡片筛选该类</div>
  </div>
  <div class="gal-grid" id="gallery"></div>
</section>

<div class="layout">
  <aside class="panel side">
    <h3>数据集</h3>
    <div class="seg" id="seg">
      <button data-split="all" class="on">全部</button>
      <button data-split="train">训练集</button>
      <button data-split="val">验证集</button>
    </div>
    <h3>类别筛选</h3>
    <div class="classList" id="classList">
      <div class="cls on" data-cls=""><span class="dot" style="background:#94a3b8"></span><span>全部类别</span><span class="n" id="allCount">{{ stats.boxes }}</span></div>
      {% for name in class_cn %}
      <div class="cls" data-cls="{{ loop.index0 }}"><span class="dot" style="background:{{ colors[loop.index0] }}"></span><span>{{ name }}</span><span class="n" data-count="{{ loop.index0 }}">{{ class_counts['all'][loop.index0] }}</span></div>
      {% endfor %}
    </div>
  </aside>

  <main>
    <div class="tools">
      <div class="search"><span style="color:#94a3b8">⌕</span><input id="q" placeholder="搜索文件名或类别，如 apple、苹果、0001"></div>
      <label class="chk"><input type="checkbox" id="multi"> 只看多目标</label>
      <button class="btn" id="randomBtn">随机抽一张</button>
      <button class="btn primary" id="resetBtn">重置筛选</button>
    </div>
    <div class="result-line"><span id="summary">加载中…</span><span>点击图片查看大图与框列表</span></div>
    <div class="grid" id="grid"></div>
    <div class="pager" id="pager"></div>
  </main>
</div>
</div>

<div class="modal" id="modal">
  <div class="modal-box">
    <div class="modal-stage">
      <button class="close" id="closeBtn">×</button>
      <img id="modalImg" alt="">
      <canvas id="modalCanvas"></canvas>
    </div>
    <div class="modal-side">
      <h2 id="modalName"></h2>
      <div class="meta" id="modalMeta"></div>
      <div id="boxList"></div>
      <div class="legend">框的颜色对应该类别；坐标为 YOLO 归一化中心点/宽高换算后的像素框。</div>
    </div>
  </div>
</div>

<script>
const CLASSES = {{ class_cn|tojson }};
const COLORS = {{ colors|tojson }};
const CLASS_COUNTS = {{ class_counts|tojson }};
const state = {split:'all', cls:'', q:'', multi:false, page:1, per:24};
const grid = document.getElementById('grid');
const pager = document.getElementById('pager');
const summary = document.getElementById('summary');

function query(){
  const p = new URLSearchParams({split:state.split, q:state.q, cls:state.cls, multi:state.multi?1:0, page:state.page, per:state.per});
  return p.toString();
}

function drawBoxes(canvas, img, boxes, numbered){
  const w = img.naturalWidth || 1, h = img.naturalHeight || 1;
  // 让 canvas 精确覆盖图片实际显示区域（避免 contain 留白导致框错位）
  const stage = img.parentElement;
  const ir = img.getBoundingClientRect(), sr = stage.getBoundingClientRect();
  canvas.style.left = (ir.left - sr.left) + 'px';
  canvas.style.top = (ir.top - sr.top) + 'px';
  canvas.style.width = ir.width + 'px';
  canvas.style.height = ir.height + 'px';
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0,0,w,h);
  if(!boxes || !boxes.length) return;
  // 字号按“显示尺寸”换算，缩略图上的中文标签才不会小到看不清
  const scale = w / Math.max(1, ir.width);
  const lw = Math.max(1.5, 2.2 * scale);
  const fs = Math.max(10, Math.round(11 * scale));
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'start';
  const placed = [];
  const hits = r => placed.some(q => r.x < q.x+q.w && r.x+r.w > q.x && r.y < q.y+q.h && r.y+r.h > q.y);

  boxes.forEach((b, idx)=>{
    const [cls,cx,cy,bw,bh] = b;
    const x = (cx-bw/2)*w, y = (cy-bh/2)*h, ww = bw*w, hh = bh*h;
    ctx.strokeStyle = COLORS[cls];
    ctx.lineWidth = lw;
    ctx.strokeRect(x,y,ww,hh);

    if(numbered){
      const r = Math.max(8, Math.round(9 * scale));
      const bx = Math.min(Math.max(x + r + 2, r), w - r);
      const by = Math.min(Math.max(y + r + 2, r), h - r);
      ctx.beginPath();
      ctx.arc(bx, by, r, 0, Math.PI * 2);
      ctx.fillStyle = COLORS[cls];
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,.9)';
      ctx.lineWidth = Math.max(1, 1.2 * scale);
      ctx.stroke();
      ctx.fillStyle = '#fff';
      ctx.font = `700 ${Math.round(r * 1.15)}px -apple-system,"PingFang SC","Microsoft YaHei",sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(String(idx + 1), bx, by + 0.5);
      ctx.textAlign = 'start';
      return;
    }

    ctx.font = `600 ${fs}px -apple-system,"PingFang SC","Microsoft YaHei",sans-serif`;
    const text = CLASSES[cls];
    const padX = fs*0.4, padY = fs*0.22;
    const tw = ctx.measureText(text).width;
    const th = fs + padY*2;
    const chipW = tw + padX*2;
    let tx = Math.min(Math.max(0, x), Math.max(0, w - chipW));
    let ty = y - th - 2;
    if(ty < 0) ty = y + 2;
    // 简单的标签避让：向下错开，避免多目标时文字叠在一起
    for(let k=0; k<6 && hits({x:tx,y:ty,w:chipW,h:th}); k++){
      ty += th + 2;
      if(ty + th > h){ ty = y + 2; tx = Math.min(Math.max(0, w - chipW), tx + chipW * 0.18); }
    }
    placed.push({x:tx,y:ty,w:chipW,h:th});
    ctx.fillStyle = COLORS[cls];
    ctx.fillRect(tx, ty, chipW, th);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, tx + padX, ty + th/2);
  });
}

function tag(label, color){
  return `<span class="tag" style="background:${color}">${label}</span>`;
}

function renderCards(items){
  grid.innerHTML = '';
  if(!items.length){
    grid.innerHTML = '<div class="empty" style="grid-column:1/-1">没有符合条件的图片，试试调整筛选或搜索词。</div>';
    return;
  }
  items.forEach(item=>{
    const card = document.createElement('div');
    card.className = 'card';
    const counts = {};
    item.boxes.forEach(b=>counts[b[0]]=(counts[b[0]]||0)+1);
    const tags = Object.keys(counts).slice(0,3).map(c=>tag(CLASSES[c], COLORS[c])).join('');
    const more = Object.keys(counts).length > 3 ? `<span class="tag" style="background:#94a3b8">+${Object.keys(counts).length-3}</span>` : '';
    card.innerHTML = `
      <div class="thumb"><img loading="lazy" src="${item.thumb}" alt=""><canvas></canvas></div>
      <div class="card-body">
        <div class="fname" title="${item.name}">${item.name}</div>
        <div class="tags">${tags}${more}<span class="count">${item.count} 框</span></div>
      </div>`;
    const img = card.querySelector('img');
    const canvas = card.querySelector('canvas');
    card._item = item;
    card._numbered = item.boxes.length > 3;
    img.addEventListener('load', ()=>drawBoxes(canvas, img, item.boxes, card._numbered));
    if(img.complete && img.naturalWidth) drawBoxes(canvas, img, item.boxes, card._numbered);
    card.addEventListener('click', ()=>openModal(item));
    grid.appendChild(card);
  });
}

function pageButtons(page, pages){
  const out = [];
  out.push(`<button data-page="${page-1}" ${page<=1?'disabled':''}>上一页</button>`);
  const start = Math.max(1, page-2), end = Math.min(pages, start+4);
  if(start > 1) out.push(`<button data-page="1">1</button>${start>2?'<button disabled>…</button>':''}`);
  for(let i=start;i<=end;i++) out.push(`<button data-page="${i}" class="${i===page?'on':''}">${i}</button>`);
  if(end < pages) out.push(`${end<pages-1?'<button disabled>…</button>':''}<button data-page="${pages}">${pages}</button>`);
  out.push(`<button data-page="${page+1}" ${page>=pages?'disabled':''}>下一页</button>`);
  return out.join('');
}

async function load(){
  summary.textContent = '加载中…';
  try{
    const res = await fetch('/api/items?' + query());
    const data = await res.json();
    if(data.error){ summary.textContent = data.error; return; }
    state.page = data.page;
    renderCards(data.items);
    summary.textContent = `共 ${data.total} 张 · 第 ${data.page} / ${data.pages} 页`;
    pager.innerHTML = pageButtons(data.page, data.pages);
  }catch(e){ summary.textContent = '加载失败：' + e.message; }
}

function updateClassCounts(){
  const counts = CLASS_COUNTS[state.split] || CLASS_COUNTS.all;
  document.querySelectorAll('[data-count]').forEach(el=>{
    el.textContent = counts[Number(el.dataset.count)] || 0;
  });
  const all = counts.reduce((a,b)=>a+b,0);
  document.getElementById('allCount').textContent = all;
}

document.getElementById('seg').addEventListener('click', e=>{
  const btn = e.target.closest('button'); if(!btn) return;
  state.split = btn.dataset.split; state.page = 1;
  document.querySelectorAll('#seg button').forEach(b=>b.classList.toggle('on', b===btn));
  updateClassCounts(); load();
});

document.getElementById('classList').addEventListener('click', e=>{
  const row = e.target.closest('.cls'); if(!row) return;
  state.cls = row.dataset.cls; state.page = 1;
  document.querySelectorAll('.cls').forEach(x=>x.classList.toggle('on', x===row));
  load();
});

let timer = null;
document.getElementById('q').addEventListener('input', e=>{
  clearTimeout(timer);
  timer = setTimeout(()=>{ state.q = e.target.value.trim(); state.page = 1; load(); }, 250);
});
document.getElementById('multi').addEventListener('change', e=>{
  state.multi = e.target.checked; state.page = 1; load();
});
document.getElementById('resetBtn').addEventListener('click', ()=>{
  state.split='all'; state.cls=''; state.q=''; state.multi=false; state.page=1;
  document.getElementById('q').value='';
  document.getElementById('multi').checked=false;
  document.querySelectorAll('#seg button').forEach(b=>b.classList.toggle('on', b.dataset.split==='all'));
  document.querySelectorAll('.cls').forEach(x=>x.classList.toggle('on', x.dataset.cls===''));
  updateClassCounts(); load();
});
document.getElementById('randomBtn').addEventListener('click', async ()=>{
  try{
    const res = await fetch('/api/random?' + query());
    const data = await res.json();
    if(data.error){ summary.textContent = data.error; return; }
    openModal(data);
  }catch(e){ summary.textContent = '加载失败：' + e.message; }
});
pager.addEventListener('click', e=>{
  const btn = e.target.closest('button'); if(!btn || btn.disabled) return;
  const p = Number(btn.dataset.page); if(!p) return;
  state.page = p; load();
});

const modal = document.getElementById('modal');
const modalImg = document.getElementById('modalImg');
const modalCanvas = document.getElementById('modalCanvas');
const boxList = document.getElementById('boxList');

function openModal(item){
  modal.classList.add('on');
  document.getElementById('modalName').textContent = item.name;
  document.getElementById('modalMeta').textContent = `${item.split==='train'?'训练集':'验证集'} · ${item.count} 个标注框`;
  modalImg._boxes = item.boxes;
  modalImg.onload = ()=>drawBoxes(modalCanvas, modalImg, item.boxes, true);
  modalImg.src = item.url;
  if(modalImg.complete && modalImg.naturalWidth) drawBoxes(modalCanvas, modalImg, item.boxes, true);
  boxList.innerHTML = item.boxes.map((b,i)=>`
    <div class="box-row"><span class="num-badge" style="background:${COLORS[b[0]]}">${i+1}</span><span>${CLASSES[b[0]]}</span><span class="coords">${(b[1]*100).toFixed(1)}%, ${(b[2]*100).toFixed(1)}%</span></div>
  `).join('') || '<div class="legend">这张图没有标注框</div>';
}
function closeModal(){ modal.classList.remove('on'); modalImg.src=''; }
document.getElementById('closeBtn').addEventListener('click', closeModal);
modal.addEventListener('click', e=>{ if(e.target===modal) closeModal(); });
document.addEventListener('keydown', e=>{ if(e.key==='Escape') closeModal(); });

let resizeTimer = null;
window.addEventListener('resize', ()=>{
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(()=>{
    document.querySelectorAll('.card').forEach(card=>{
      const img = card.querySelector('img'), canvas = card.querySelector('canvas');
      if(card._item && img.complete) drawBoxes(canvas, img, card._item.boxes, card._numbered);
    });
    if(modal.classList.contains('on') && modalImg.complete){
      drawBoxes(modalCanvas, modalImg, modalImg._boxes || [], true);
    }
  }, 160);
});

updateClassCounts();
load();

async function loadGallery(){
  const box = document.getElementById('gallery');
  try{
    const data = await (await fetch('/api/gallery?n=4')).json();
    box.innerHTML = data.map(c=>`
      <div class="gcard" data-cls="${c.cls}">
        <div class="gtop">
          <span class="dot" style="background:${c.color};width:8px;height:8px;border-radius:50%;display:inline-block"></span>
          <span class="gname">${c.name}</span>
          <span class="gcount">${c.count} 张</span>
        </div>
        <div class="gshots">
          ${c.images.map(im=>`<div class="gshot"><img loading="lazy" src="/thumb/${im.split}/${encodeURIComponent(im.name)}?w=200" alt=""></div>`).join('')}
        </div>
      </div>`).join('');
    box.querySelectorAll('.gcard').forEach(card=>{
      card.addEventListener('click', ()=>{
        const cid = card.dataset.cls;
        state.cls = cid; state.page = 1;
        document.querySelectorAll('.cls').forEach(x=>x.classList.toggle('on', x.dataset.cls===cid));
        load();
        document.querySelector('.layout').scrollIntoView({behavior:'smooth', block:'start'});
      });
    });
  }catch(e){ box.innerHTML = '<div style="color:#94a3b8;font-size:12px">画廊加载失败</div>'; }
}
loadGallery();

</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML,
        stats=STATS,
        class_cn=CLASS_CN,
        dataset_name=os.path.basename(DATA_DIR),
        colors=COLORS,
        class_counts=CLASS_COUNTS,
    )


if __name__ == "__main__":
    print(f"图片 {STATS['images']} 张，标注框 {STATS['boxes']} 个")
    app.run(host="127.0.0.1", port=5003, debug=False)
