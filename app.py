import io
import json
import os
import sqlite3
import time
from datetime import datetime
from functools import lru_cache

from flask import Flask, jsonify, render_template_string, request, send_from_directory
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename

from predict import predict_base64, predict_bytes, predict_path

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
SAMPLE_DIR = os.path.join(BASE, "samples")
MODEL_DIR = os.path.join(BASE, "models")
DATA_DIR = os.path.join(BASE, "dataset", "yolo16_final")
DB_PATH = os.path.join(BASE, "history.db")
ALLOWED = {"png", "jpg", "jpeg", "webp"}
GALLERY_N = 3          # 每类展示的实例数

os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__)

# ---------- 16 类数据集（实例总览用，只读） ----------
META = json.loads(open(os.path.join(DATA_DIR, "dataset_meta.json"), encoding="utf-8").read())
CLASS_NAMES = META["class_names"]
CLASS_CN = META["cn_names"]
COLORS = [
    "#f87171", "#ef4444", "#b91c1c",
    "#fbbf24", "#f59e0b", "#b45309",
    "#a78bfa", "#8b5cf6", "#6d28d9",
    "#fb923c", "#f97316", "#c2410c",
    "#fb7185", "#f43f5e", "#be123c",
    "#64748b",
]


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, filename TEXT,
        label TEXT, label_cn TEXT, confidence REAL, source TEXT)""")
    return conn


def load_eval():
    with open(os.path.join(MODEL_DIR, "evaluation_species.json"), encoding="utf-8") as fh:
        return json.load(fh)


EVAL = load_eval()
SAMPLES = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".png")))


def history(limit=8):
    conn = db()
    rows = conn.execute("SELECT ts,label_cn,confidence,source,filename FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows


def parse_boxes(label_path):
    boxes = []
    for line in open(label_path, encoding="utf-8").read().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        boxes.append((int(float(p[0])), *(float(v) for v in p[1:5])))
    return boxes


@lru_cache(maxsize=1)
def dataset_index():
    """扫描 dataset/yolo16_final，得到每张图的标注框与所属类别。"""
    items = []
    for split in ("train", "val"):
        img_dir = os.path.join(DATA_DIR, "images", split)
        lbl_dir = os.path.join(DATA_DIR, "labels", split)
        for fname in sorted(os.listdir(lbl_dir)):
            if not fname.endswith(".txt"):
                continue
            stem = fname[:-4]
            img = None
            for ext in (".jpg", ".jpeg", ".png"):
                p = os.path.join(img_dir, stem + ext)
                if os.path.exists(p):
                    img = stem + ext
                    break
            if img is None:
                continue
            boxes = parse_boxes(os.path.join(lbl_dir, fname))
            if not boxes:
                continue
            items.append({
                "split": split, "stem": stem, "file": img, "boxes": boxes,
                "classes": sorted({b[0] for b in boxes}),
            })
    return items


@lru_cache(maxsize=1)
def class_counts():
    counts = [0] * len(CLASS_NAMES)
    for it in dataset_index():
        for cls, *_ in it["boxes"]:
            counts[cls] += 1
    return counts


def gallery():
    """每类挑 3 张实例：优先用逐张核对过的清单，不足时按数据集顺序补。"""
    items = {f"{it['split']}/{it['stem']}": it for it in dataset_index()}
    picks = {}
    picks_path = os.path.join(MODEL_DIR, "gallery_picks.json")
    if os.path.exists(picks_path):
        picks = json.load(open(picks_path, encoding="utf-8"))

    groups = []
    for cid, (name, cn) in enumerate(zip(CLASS_NAMES, CLASS_CN)):
        chosen, seen = [], set()
        for key in picks.get(name, []):
            it = items.get(key)
            if it and cid in it["classes"] and key not in seen:
                chosen.append({"split": it["split"], "stem": it["stem"]})
                seen.add(key)
        if len(chosen) < GALLERY_N:
            rest = [it for it in dataset_index() if cid in it["classes"]
                    and f"{it['split']}/{it['stem']}" not in seen]
            rest.sort(key=lambda it: (it["split"] != "train", it["stem"]))
            if not chosen:  # 没有核对清单时才退回自动挑选，避免用未核对的图充数
                chosen = [{"split": it["split"], "stem": it["stem"]} for it in rest[:GALLERY_N]]
        groups.append({
            "cls": cid, "name": name, "cn": cn.replace("过熟", "完熟"), "color": COLORS[cid],
            "count": class_counts()[cid], "images": chosen,
        })
    return groups


def find_item(split, stem):
    for it in dataset_index():
        if it["split"] == split and it["stem"] == stem:
            return it
    return None


def full_path(split, stem):
    it = find_item(split, stem)
    if it is None:
        return None
    p = os.path.join(DATA_DIR, "images", split, it["file"])
    return p if os.path.exists(p) else None


@lru_cache(maxsize=256)
def render_thumb(split, stem, width):
    path = full_path(split, stem)
    if path is None:
        return None
    img = Image.open(path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    lw = max(2, W // 160)
    for cls, cx, cy, bw, bh in find_item(split, stem)["boxes"]:
        draw.rectangle(
            [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H],
            outline=COLORS[cls], width=lw,
        )
    height = max(1, round(H * width / W))
    return img.resize((width, height), Image.LANCZOS)


HTML = r"""
<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>水果成熟度分级系统</title>
<style>
:root{--bg:#f7f8fa;--card:#fff;--line:#e5e7eb;--text:#1f2329;--muted:#6b7280;--blue:#2563eb;--blue-bg:#eff6ff;--green:#0f9d75;--amber:#d97706;--red:#dc2626}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6}
.wrap{max-width:1180px;margin:0 auto;padding:32px 20px 56px}
header{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:24px}
h1{font-size:26px;margin:0;font-weight:600;letter-spacing:-.4px}
.sub{color:var(--muted);font-size:14px;margin-top:6px}
.badge{background:var(--blue-bg);color:var(--blue);border:1px solid #bfdbfe;border-radius:999px;padding:6px 14px;font-size:13px;white-space:nowrap}
.grid{display:grid;grid-template-columns:1.05fr .95fr;gap:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px}
.card h2{font-size:16px;margin:0 0 14px;font-weight:600}
.drop{border:1.5px dashed #cbd5e1;border-radius:12px;background:#fbfcfe;padding:34px 20px;text-align:center;cursor:pointer;transition:.15s}
.drop:hover,.drop.over{border-color:var(--blue);background:var(--blue-bg)}
.drop strong{display:block;font-size:15px;font-weight:600}
.drop span{color:var(--muted);font-size:13px}
input[type=file]{display:none}
.row{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}
button{font:inherit;font-size:13px;border:1px solid var(--line);background:#fff;color:var(--text);border-radius:8px;padding:8px 14px;cursor:pointer}
button:hover{border-color:var(--blue);color:var(--blue)}
button.primary{background:var(--blue);border-color:var(--blue);color:#fff}
video{width:100%;max-height:220px;border-radius:10px;background:#111;display:none;margin-top:12px}
.preview{width:100%;height:230px;object-fit:contain;background:#f3f4f6;border-radius:12px;display:block}
.result-top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:16px 0 10px}
.label{font-size:22px;font-weight:600}
.conf{font-size:14px;color:var(--muted)}
.conf b{font-size:20px;color:var(--green)}
.bar-row{margin-top:10px}
.bar-head{display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px}
.track{height:8px;background:#eef0f3;border-radius:999px;overflow:hidden}
.fill{height:100%;background:var(--blue);border-radius:999px;transition:width .35s}
.stats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}
.stat{background:#f9fafb;border-radius:10px;padding:12px 14px}
.stat .k{font-size:12px;color:var(--muted)}
.stat .v{font-size:22px;font-weight:600;margin-top:2px}
.section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin-top:18px}
table{width:100%;border-collapse:collapse;font-size:13px}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:500}
td.num{font-variant-numeric:tabular-nums}
.cm td{text-align:center}
.cm th{text-align:center}
.hint{color:var(--muted);font-size:12px;margin-top:8px}
.err{color:var(--red);font-size:13px;margin-top:8px;display:none}
.label.unsupported{color:var(--amber)}
.warn{display:none;margin-top:12px;padding:10px 12px;border-radius:8px;background:#fffbeb;border:1px solid #fde68a;color:#92400e;font-size:13px}
.bars-muted{opacity:.55}
.lvl{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:14px}
.lvl-item{border:1px solid var(--line);border-radius:9px;padding:8px 10px;background:#fbfcfe}
.lvl-item.on{border-color:#bfdbfe;background:var(--blue-bg)}
.lvl-name{font-size:12px;color:var(--muted)}
.lvl-val{font-size:15px;font-weight:600;margin-top:2px}
.lvl-bar{height:5px;background:#eef0f3;border-radius:999px;margin-top:6px;overflow:hidden}
.lvl-bar span{display:block;height:100%;background:var(--green);border-radius:999px;transition:width .35s}
.evidence{font-size:12px;color:var(--muted);margin-top:10px}
.lvl-sec{font-size:12px;color:var(--muted);margin-top:16px;letter-spacing:.4px}
.sec-head{display:flex;align-items:baseline;gap:10px;margin-bottom:16px}
.sec-head h2{font-size:16px;margin:0;font-weight:600}
.sec-head .hint{margin:0}
.gal{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
.gcard{border:1px solid var(--line);border-radius:11px;padding:9px;background:#fff;transition:.15s}
.gcard:hover{border-color:#cbd5e1;box-shadow:0 4px 14px rgba(15,23,42,.07)}
.gtop{display:flex;align-items:center;gap:6px;margin-bottom:8px}
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex:none}
.gname{font-size:13px;font-weight:600}
.gcount{margin-left:auto;font-size:11.5px;color:var(--muted);font-variant-numeric:tabular-nums}
.gshots{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}
.gshot{position:relative;aspect-ratio:1/1;background:#eef0f3;border-radius:6px;overflow:hidden;cursor:zoom-in}
.gshot img{width:100%;height:100%;object-fit:cover;display:block;transition:.2s}
.gshot:hover img{transform:scale(1.06)}
.modal{position:fixed;inset:0;background:rgba(15,23,42,.72);display:none;align-items:center;justify-content:center;padding:28px;z-index:50}
.modal.on{display:flex}
.modal-box{background:#fff;border-radius:14px;max-width:900px;width:100%;max-height:90vh;overflow:auto;padding:14px}
.modal-box img{width:100%;border-radius:10px;display:block;background:#f3f4f6}
.modal-cap{display:flex;align-items:center;gap:8px;font-size:14px;font-weight:600;padding:8px 4px 4px}
.modal-cap .meta{margin-left:auto;font-size:12px;color:var(--muted);font-weight:400}
@media(max-width:1000px){.gal{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div><h1>水果成熟度分级系统</h1><div class="sub">MobileNetV2 · 苹果 / 香蕉 / 葡萄 / 橙子 / 草莓 · 未熟 / 成熟 / 完熟 · 腐烂果单独识别</div></div>
  <div class="badge">端到端种类准确率 {{ '%.1f'|format(e2e*100) }}%</div>
</header>

<div class="grid">
  <div class="card">
    <h2>上传图片进行识别</h2>
    <div class="drop" id="drop"><strong>点击选择图片，或拖拽到此处</strong><span>支持 JPG / PNG / WEBP · 单张水果照片效果最佳</span></div>
    <input type="file" id="file" accept="image/*">
    <div class="hint">支持 5 种水果的成熟度分级；腐烂 / 霉变果实单独提示，不参与分级。</div>
    <div class="row">
      <button class="primary" id="camBtn">开启摄像头</button>
      <button id="shotBtn" style="display:none">拍摄并识别</button>
    </div>
    <video id="video" autoplay playsinline></video>
    <div class="err" id="err"></div>
  </div>

  <div class="card">
    <h2>识别结果</h2>
    <img class="preview" id="preview" src="/samples/{{ samples[0] if samples else '' }}" alt="预览">
    <div class="result-top"><div class="label" id="label">等待识别</div><div class="conf">置信度 <b id="conf">—</b></div></div>
    <div class="lvl-sec" id="barsTitle">种类概率</div>
    <div id="bars"></div>
    <div id="levelWrap" style="display:none">
      <div class="lvl-sec">成熟度判定（颜色语义）</div>
      <div class="lvl" id="lvl"></div>
      <div class="evidence" id="evidence"></div>
    </div>
    <div class="warn" id="warn"></div>
    <div class="stats">
      <div class="stat"><div class="k">种类准确率（端到端）</div><div class="v">{{ '%.1f'|format(e2e*100) }}%</div></div>
      <div class="stat"><div class="k">训练样本</div><div class="v">{{ train_n }}</div></div>
      <div class="stat"><div class="k">验证样本</div><div class="v">{{ val_n }}</div></div>
    </div>
  </div>
</div>

<div class="section">
  <div class="sec-head">
    <h2>16 类实例总览</h2>
    <div class="hint">训练数据 {{ total_images }} 张 · 每类均展示实例并标出标注框 · 点击可放大查看</div>
  </div>
  <div class="gal">
    {% for g in gallery %}
    <div class="gcard">
      <div class="gtop">
        <span class="dot" style="background:{{ g.color }}"></span>
        <span class="gname">{{ g.cn }}</span>
        <span class="gcount">{{ g.count }} 张</span>
      </div>
      <div class="gshots">
        {% for im in g.images %}
        <div class="gshot" data-split="{{ im.split }}" data-stem="{{ im.stem }}" data-cn="{{ g.cn }}" data-count="{{ g.count }}">
          <img loading="lazy" src="/thumb/{{ im.split }}/{{ im.stem }}?w=240" alt="{{ g.cn }}">
        </div>
        {% endfor %}
      </div>
    </div>
    {% endfor %}
  </div>
  <div class="hint">数据来自 dataset/yolo16_final：5 种水果各分未熟 / 成熟 / 完熟，外加腐烂水果；每类有多少张干净样本就用多少张。展示图均经逐张核对，个别类别（如葡萄-未熟）源数据干净样本较少，因此只展示 1–2 张。档位标签按颜色分位生成，是同档内的相对次序，不等同于农学成熟度真值。</div>
</div>

<div class="section">
  <h2>种类识别准确率与混淆矩阵</h2>
  <table class="cm">
    <tr><th>真实 \ 预测</th>{% for c in class_order %}<th>{{ class_cn[c] }}</th>{% endfor %}<th>召回率</th></tr>
    {% for row in cm %}<tr><th>{{ class_cn[class_order[loop.index0]] }}</th>{% for v in row %}<td class="num">{{ v }}</td>{% endfor %}<td class="num">{{ '%.1f'|format(per_class[class_order[loop.index0]].accuracy*100) }}%</td></tr>{% endfor %}
  </table>
  <div class="hint">数据集 dataset/yolo16_final：709 张实拍图，5 种水果三档成熟度合并为种类 + 腐烂水果；验证集 {{ val_n }} 张，带 TTA 准确率 {{ '%.1f'|format(tta*100) }}%。成熟度不靠这张表判定，而是按果实区域颜色语义给出（见上方“颜色证据”）。</div>
</div>

<div class="section">
  <h2>最近识别记录</h2>
  <table><thead><tr><th>时间</th><th>结果</th><th>置信度</th><th>来源</th></tr></thead><tbody id="hist">
    {% for h in history %}<tr><td class="num">{{ h[0] }}</td><td>{{ h[1] }}</td><td class="num">{% if h[1]=='未能确定' %}—{% else %}{{ '%.1f'|format(h[2]*100) }}%{% endif %}</td><td>{{ h[3] }}</td></tr>{% else %}<tr><td colspan="4">暂无记录</td></tr>{% endfor %}
  </tbody></table>
</div>
</div>

<div class="modal" id="modal">
  <div class="modal-box">
    <div class="modal-cap"><span class="dot" id="modalDot"></span><span id="modalName"></span><span class="meta" id="modalMeta"></span></div>
    <img id="modalImg" alt="">
  </div>
</div>

<script>
const file=document.getElementById('file'),drop=document.getElementById('drop'),err=document.getElementById('err');
const preview=document.getElementById('preview'),label=document.getElementById('label'),conf=document.getElementById('conf'),bars=document.getElementById('bars'),warn=document.getElementById('warn');
const video=document.getElementById('video'),camBtn=document.getElementById('camBtn'),shotBtn=document.getElementById('shotBtn');
const modal=document.getElementById('modal'),modalImg=document.getElementById('modalImg');
let stream=null;
function showErr(m){err.textContent=m;err.style.display='block'}
function render(data){
  err.style.display='none';
  preview.src=data.image_url;
  const unsupported=!!data.unsupported;
  const shownLabel=data.label_cn;
  const shownConf=(data.confidence*100).toFixed(1)+'%';
  label.textContent=shownLabel;
  label.className=unsupported?'label unsupported':'label';
  conf.textContent=shownConf;
  bars.className=unsupported?'bars-muted':'';
  bars.innerHTML=data.all.map(x=>`<div class="bar-row"><div class="bar-head"><span>${x.cn}</span><span>${(x.prob*100).toFixed(1)}%</span></div><div class="track"><div class="fill" style="width:${(x.prob*100).toFixed(1)}%"></div></div></div>`).join('');
  const lw=document.getElementById('levelWrap');
  if(data.level_probs){
    lw.style.display='block';
    const names={L:'未熟',M:'成熟',H:'完熟'};
    document.getElementById('lvl').innerHTML=['L','M','H'].map(k=>{
      const p=data.level_probs[k]*100;
      return `<div class="lvl-item${data.level===k?' on':''}"><div class="lvl-name">${names[k]}</div><div class="lvl-val">${p.toFixed(0)}%</div><div class="lvl-bar"><span style="width:${p.toFixed(0)}%"></span></div></div>`;
    }).join('');
    document.getElementById('evidence').textContent=data.note;
  }else{ lw.style.display='none'; }
  warn.style.display=(unsupported||data.note)?'block':'none';
  warn.textContent=unsupported?data.reason:(data.level_hint||data.note||'');
  const tbody=document.getElementById('hist');
  if(tbody.querySelector('td[colspan]')) tbody.innerHTML='';
  const tr=document.createElement('tr');
  tr.innerHTML=`<td class="num">${new Date().toLocaleString('zh-CN',{hour12:false})}</td><td>${shownLabel}</td><td class="num">${shownConf}</td><td>${data.source}</td>`;
  tbody.prepend(tr);
}
async function post(url,options){
  label.textContent='识别中…';conf.textContent='—';
  try{const r=await fetch(url,options);const d=await r.json();if(d.error)showErr(d.error);else render(d)}catch(e){showErr('请求失败：'+e.message)}
}
function upload(f){
  if(!f)return;
  const fd=new FormData();fd.append('image',f);
  post('/predict',{method:'POST',body:fd});
}
drop.onclick=()=>file.click();
file.onchange=()=>upload(file.files[0]);
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');upload(e.dataTransfer.files[0])};
document.querySelectorAll('.gshot').forEach(el=>el.onclick=()=>{
  const cls=el.closest('.gcard').querySelector('.dot').style.background;
  document.getElementById('modalDot').style.background=cls;
  document.getElementById('modalName').textContent=el.dataset.cn;
  document.getElementById('modalMeta').textContent='该类共 '+el.dataset.count+' 张 · '+el.dataset.split;
  modalImg.src='/img/'+el.dataset.split+'/'+encodeURIComponent(el.dataset.stem);
  modal.classList.add('on');
});
modal.onclick=e=>{ if(e.target===modal||e.target===modalImg) modal.classList.remove('on'); };
document.addEventListener('keydown',e=>{ if(e.key==='Escape') modal.classList.remove('on'); });
camBtn.onclick=async()=>{
  if(stream){stream.getTracks().forEach(t=>t.stop());stream=null;video.style.display='none';camBtn.textContent='开启摄像头';shotBtn.style.display='none';return}
  try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});video.srcObject=stream;video.style.display='block';camBtn.textContent='关闭摄像头';shotBtn.style.display='inline-block'}catch(e){showErr('无法访问摄像头：'+e.message)}
};
shotBtn.onclick=()=>{
  const c=document.createElement('canvas');c.width=video.videoWidth;c.height=video.videoHeight;
  c.getContext('2d').drawImage(video,0,0);
  post('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({image:c.toDataURL('image/jpeg')})});
};
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    gal = gallery()
    return render_template_string(
        HTML,
        accuracy=EVAL["accuracy"],
        e2e=EVAL.get("end_to_end_accuracy", EVAL["accuracy"]),
        tta=EVAL.get("tta_accuracy", EVAL["accuracy"]),
        train_n=EVAL["train_samples"],
        val_n=EVAL["val_samples"],
        class_order=EVAL["class_order"],
        class_cn=EVAL["class_cn"],
        per_class=EVAL["per_class"],
        cm=EVAL["confusion_matrix"],
        samples=SAMPLES,
        gallery=gal,
        total_images=len(dataset_index()),
        history=history(),
    )


@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        source = "上传"
        filename = None
        if request.content_type and "application/json" in request.content_type:
            payload = request.get_json(force=True) or {}
            if payload.get("sample"):
                name = secure_filename(payload["sample"])
                result = predict_path(os.path.join(SAMPLE_DIR, name))
                source = "示例"
                filename = name
            elif payload.get("image"):
                result = predict_base64(payload["image"])
                source = "摄像头"
            else:
                return jsonify(error="缺少 image 或 sample 字段"), 400
        else:
            f = request.files.get("image")
            if not f or not f.filename:
                return jsonify(error="请选择图片"), 400
            ext = f.filename.rsplit(".", 1)[-1].lower()
            if ext not in ALLOWED:
                return jsonify(error="仅支持 JPG / PNG / WEBP"), 400
            filename = f"{int(time.time()*1000)}_{secure_filename(f.filename)}"
            path = os.path.join(UPLOAD_DIR, filename)
            f.save(path)
            result = predict_path(path)

        conn = db()
        conn.execute(
            "INSERT INTO predictions(ts,filename,label,label_cn,confidence,source) VALUES(?,?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), filename or "", result["label"], result["label_cn"], result["confidence"], source),
        )
        conn.commit()
        conn.close()
        result["source"] = source
        if filename:
            folder = "samples" if source == "示例" else "uploads"
            result["image_url"] = f"/{folder}/{filename}"
        else:
            result["image_url"] = preview_url_from_result()
        return jsonify(result)
    except Exception as exc:
        return jsonify(error=str(exc)), 500


def preview_url_from_result():
    return f"/samples/{SAMPLES[0]}" if SAMPLES else ""


@app.route("/thumb/<split>/<stem>")
def thumb(split, stem):
    width = max(80, min(800, int(request.args.get("w", 240))))
    img = render_thumb(split, stem, width)
    if img is None:
        return ("", 404)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=86)
    buf.seek(0)
    return app.response_class(buf.read(), mimetype="image/jpeg")


@app.route("/img/<split>/<stem>")
def sample_image(split, stem):
    path = full_path(split, stem)
    if path is None:
        return ("", 404)
    return send_from_directory(os.path.dirname(path), os.path.basename(path))


@app.route("/uploads/<name>")
def uploaded(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.route("/samples/<name>")
def sampled(name):
    return send_from_directory(SAMPLE_DIR, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
