import json
import os
import sqlite3
import time
from datetime import datetime

from flask import Flask, jsonify, render_template_string, request, send_from_directory
from werkzeug.utils import secure_filename

from predict import predict_base64, predict_bytes, predict_path

BASE = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE, "uploads")
SAMPLE_DIR = os.path.join(BASE, "samples")
MODEL_DIR = os.path.join(BASE, "models")
DB_PATH = os.path.join(BASE, "history.db")
ALLOWED = {"png", "jpg", "jpeg", "webp"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
app = Flask(__name__)


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, filename TEXT,
        label TEXT, label_cn TEXT, confidence REAL, source TEXT)""")
    return conn


def load_eval():
    with open(os.path.join(MODEL_DIR, "evaluation.json"), encoding="utf-8") as fh:
        return json.load(fh)


EVAL = load_eval()
SAMPLES = sorted(f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith((".jpg", ".png")))


def history(limit=8):
    conn = db()
    rows = conn.execute("SELECT ts,label_cn,confidence,source,filename FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return rows


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
.wrap{max-width:1120px;margin:0 auto;padding:32px 20px 56px}
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
.samples{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}
.sample{border:1px solid var(--line);border-radius:10px;overflow:hidden;cursor:pointer;background:#fff;text-align:center;font-size:12px}
.sample:hover{border-color:var(--blue)}
.sample img{width:100%;height:92px;object-fit:cover;display:block;background:#f3f4f6}
.sample div{padding:6px 4px;color:var(--muted)}
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
@media(max-width:820px){.grid{grid-template-columns:1fr}.samples{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div><h1>水果成熟度分级系统</h1><div class="sub">MobileNetV2 迁移学习 · 香蕉与芒果成熟状态识别 · 实时演示</div></div>
  <div class="badge">验证集准确率 {{ '%.1f'|format(accuracy*100) }}%</div>
</header>

<div class="grid">
  <div class="card">
    <h2>上传图片进行识别</h2>
    <div class="drop" id="drop"><strong>点击选择图片，或拖拽到此处</strong><span>支持 JPG / PNG / WEBP · 单张水果照片效果最佳</span></div>
    <input type="file" id="file" accept="image/*">
    <div class="row">
      <button class="primary" id="camBtn">开启摄像头</button>
      <button id="shotBtn" style="display:none">拍摄并识别</button>
    </div>
    <video id="video" autoplay playsinline></video>
    <div class="err" id="err"></div>
    <h2 style="margin-top:22px">快速体验示例</h2>
    <div class="samples">
      {% for s in samples %}<div class="sample" data-file="{{ s }}"><img src="/samples/{{ s }}"><div>{{ s.replace('sample_','').replace('_1.jpg','').replace('Ripe_','成熟').replace('Raw_','未熟').replace('Banana','香蕉').replace('Mango','芒果') }}</div></div>{% endfor %}
    </div>
  </div>

  <div class="card">
    <h2>识别结果</h2>
    <img class="preview" id="preview" src="/samples/{{ samples[0] if samples else '' }}" alt="预览">
    <div class="result-top"><div class="label" id="label">等待识别</div><div class="conf">置信度 <b id="conf">—</b></div></div>
    <div id="bars"></div>
    <div class="stats">
      <div class="stat"><div class="k">验证准确率</div><div class="v">{{ '%.1f'|format(accuracy*100) }}%</div></div>
      <div class="stat"><div class="k">训练样本</div><div class="v">{{ train_n }}</div></div>
      <div class="stat"><div class="k">验证样本</div><div class="v">{{ val_n }}</div></div>
    </div>
  </div>
</div>

<div class="section">
  <h2>各类别准确率与混淆矩阵</h2>
  <table class="cm">
    <tr><th></th>{% for c in class_order %}<th>{{ c.replace('Raw_','未熟').replace('Ripe_','成熟').replace('Banana','香蕉').replace('Mango','芒果') }}</th>{% endfor %}<th>准确率</th></tr>
    {% for row in cm %}<tr><th>{{ class_order[loop.index0].replace('Raw_','未熟').replace('Ripe_','成熟').replace('Banana','香蕉').replace('Mango','芒果') }}</th>{% for v in row %}<td class="num">{{ v }}</td>{% endfor %}<td class="num">{{ '%.1f'|format(per_class[class_order[loop.index0]].accuracy*100) }}%</td></tr>{% endfor %}
  </table>
  <div class="hint">数据集：Hugging Face · darthraider/fruit-ripeness-detection-dataset（Apache-2.0）</div>
</div>

<div class="section">
  <h2>最近识别记录</h2>
  <table><thead><tr><th>时间</th><th>结果</th><th>置信度</th><th>来源</th></tr></thead><tbody id="hist">
    {% for h in history %}<tr><td class="num">{{ h[0] }}</td><td>{{ h[1] }}</td><td class="num">{{ '%.1f'|format(h[2]*100) }}%</td><td>{{ h[3] }}</td></tr>{% else %}<tr><td colspan="4">暂无记录</td></tr>{% endfor %}
  </tbody></table>
</div>
</div>

<script>
const file=document.getElementById('file'),drop=document.getElementById('drop'),err=document.getElementById('err');
const preview=document.getElementById('preview'),label=document.getElementById('label'),conf=document.getElementById('conf'),bars=document.getElementById('bars');
const video=document.getElementById('video'),camBtn=document.getElementById('camBtn'),shotBtn=document.getElementById('shotBtn');
let stream=null;
function showErr(m){err.textContent=m;err.style.display='block'}
function render(data){
  err.style.display='none';
  preview.src=data.image_url;
  label.textContent=data.label_cn;
  conf.textContent=(data.confidence*100).toFixed(1)+'%';
  bars.innerHTML=data.all.map(x=>`<div class="bar-row"><div class="bar-head"><span>${x.cn}</span><span>${(x.prob*100).toFixed(1)}%</span></div><div class="track"><div class="fill" style="width:${(x.prob*100).toFixed(1)}%"></div></div></div>`).join('');
  const tbody=document.getElementById('hist');
  if(tbody.querySelector('td[colspan]')) tbody.innerHTML='';
  const tr=document.createElement('tr');
  tr.innerHTML=`<td class="num">${new Date().toLocaleString('zh-CN',{hour12:false})}</td><td>${data.label_cn}</td><td class="num">${(data.confidence*100).toFixed(1)}%</td><td>${data.source}</td>`;
  tbody.prepend(tr);
}
async function post(url,options){
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
document.querySelectorAll('.sample').forEach(el=>el.onclick=()=>post('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sample:el.dataset.file})}));
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
    return render_template_string(
        HTML,
        accuracy=EVAL["accuracy"],
        train_n=EVAL["train_samples"],
        val_n=EVAL["val_samples"],
        class_order=EVAL["class_order"],
        per_class=EVAL["per_class"],
        cm=EVAL["confusion_matrix"],
        samples=SAMPLES,
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


@app.route("/uploads/<name>")
def uploaded(name):
    return send_from_directory(UPLOAD_DIR, name)


@app.route("/samples/<name>")
def sampled(name):
    return send_from_directory(SAMPLE_DIR, name)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
