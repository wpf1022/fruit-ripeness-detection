# 水果成熟度智能分级与检测系统

一个从「整图分类」升级到「多目标检测」的完整项目：用 YOLOv8n 在一张图里定位多个水果，并分别判断成熟度（高/中/低）与腐烂状态，配套 Flask 可视化演示网页。

## 效果

- **16 类**：苹果 / 香蕉 / 葡萄 / 草莓 / 柿子 × 高、中、低成熟，外加 `bad fruit`
- **模型指标**：mAP50 **0.832**，mAP50-95 **0.671**，Precision 0.810，Recall 0.788
- **速度**：CPU 约 19 ms/张；模型权重 6.2 MB
- **功能**：上传 / 拖拽 / 摄像头 / 示例图四种入口，实时画框 + 中文类别 + 置信度，支持多目标、置信度阈值调节、识别历史留痕

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动演示网页（Windows 可直接双击 start_detector.bat）
python detect_app.py
# 浏览器打开 http://127.0.0.1:5001

# 3. 命令行检测单张图片
python detect_photo.py path/to/image.jpg 0.25

# 4. 查看训练集标注（Windows 可双击 start_label_viewer.bat）
python label_viewer.py
# 浏览器打开 http://127.0.0.1:5003
```

> 说明：`detect_app.py` 需要 `dataset/yolo16/dataset_meta.json`（已随仓库提供）来读取类别名。模型权重在 `models/fruit_yolov8_detector.pt`。

## 项目结构

| 文件 | 作用 |
|---|---|
| `detect_app.py` / `static/detect_index.html` | 升级版检测演示网页（主入口，端口 5001） |
| `detect_photo.py` | 命令行单图检测 |
| `label_viewer.py` / `start_label_viewer.bat` | 训练集标注查看器（端口 5003）：筛选、搜索、分页、大图看框 |
| `train_detector.py` / `finetune_detector.py` | 16 类 YOLOv8 训练 / 微调 |
| `prepare_yolo16.py` / `prepare_persimmon.py` | 数据整理管线：把多个开源集映射成统一 16 类 YOLO 格式 |
| `models/fruit_yolov8_detector.pt` | 最终检测模型权重 |
| `models/detection_evaluation.json` | 评估报告与数据来源 |
| `demo_samples/` | 仓库内置的 16 张演示示例图，克隆后即可一键体验，无需下载数据集 |
| `app.py` / `train_species.py` / `predict.py` / `ripeness.py` | 成熟度分级系统（MobileNetV2 种类模型 + 颜色成熟度，端口 5000） |
| `calibrate_ood.py` / `models/ood_stats.npz` | 旧的“未知类别拒绝”校准脚本，已被新链路取代，留作对比 |

> 分级页支持 **苹果 / 香蕉 / 葡萄 / 橙子 / 草莓** 五种水果的三档成熟度（未熟 / 成熟 / 完熟），并单独识别腐烂水果。识别链路：YOLOv8 只负责**定位果实** → MobileNetV2 判**种类** → 在同一个框内按颜色语义判**成熟度**。不再使用“非支持类别”硬拒绝，低置信度时才提示“未能确定”。

> 演示网页开箱即用：模型权重、类别元数据和 16 张示例图都已随仓库提供，克隆后 `pip install -r requirements.txt` 再 `python detect_app.py` 即可，无需下载 2 GB 训练数据集。摄像头支持开启 / 拍摄 / 关闭，关闭时会真正释放设备。

## 数据来源

数据集由多个开源集合并整理而成，原始压缩包体积约 2 GB，未随仓库提交。按 `prepare_yolo16.py` 和 `prepare_persimmon.py` 中的说明重新下载并整理即可复现：

- 苹果 / 香蕉 / 葡萄 的各成熟度 + 腐烂类：Kaggle `fruit-ripeness-unjex/2`（CC BY 4.0）
- 草莓三类：Kaggle `samatsauranbek/multi-class-strawberry-ripeness-detection-v2`（CC BY 4.0）
- 柿子三类：Fruits-360 的 Kaki 样本经色相增强 + 贴背景弱监督生成（见 `prepare_persimmon.py`）

整理后共 2794 张训练图、734 张验证图。当前工作副本已下载并展开：`dataset/yolo16/images/{train,val}` 与同名 `labels/{train,val}`，共 7211 个标注框，无空标注。

## 局限

柿子三类为弱监督生成，且所有数据场景较单一；mAP 是在这套拼装数据上测得，真实光照下的泛化能力有限。下一步计划补充自采实拍样本和外部验证集。

## License

代码 MIT。数据集遵循各自原始许可证（见上）。

## 16 类数据集重建（yolo16_final）

原来的 `dataset/yolo16` 直接继承 Roboflow 源标注，核查后发现：橙子类里是芒果、苹果类里混入青椒和糖果树、"未熟"香蕉全是黄香蕉、葡萄框落在藤蔓上。因此重建为 `dataset/yolo16_final`。

**结构**：5 种水果 × 未熟/成熟/过熟 + 腐烂水果 = 16 类，每类 31~70 张，每类数量严格均等（同一水果三档等分）。

**重建流程**
```bash
python rebuild_dataset.py        # 从压缩包按几何规则清洗，产出 dataset/yolo16_clean
python relabel_dataset.py        # 种类闸门 + 颜色重标三档，产出 dataset/yolo16_final
python make_contact_sheet.py --data dataset/yolo16_final --out contacts_final
python label_viewer.py           # http://127.0.0.1:5003 按类查看实例
```

**标签口径（可复现）**
- 种类闸门：形状比 + 颜色构成 + ImageNet 分类器（香蕉/苹果/橙子三类可靠，草莓与葡萄改用颜色规则）
- 成熟度：框内颜色得分 `red·1.0 + orange·0.85 + yellow·0.55 + dark·0.40 − green·0.75 − white·0.30`，按每种水果内部三分位切档，保证 L < M < H 在视觉上单调
- 葡萄的框改为按果串紫色/深色像素重新定位

**已知局限**：橙子未熟档仍混有少量芒果（两者颜色、形状接近，自动判别无法完全分开）；葡萄框覆盖整簇藤蔓，精度低于其他类。若要用于正式训练，建议对这 830 张做一轮人工复核。

## 页面整合：分级系统 + 16 类实例总览

`app.py`（端口 5000）在原有"上传识别 / 识别结果 / 混淆矩阵 / 识别记录"基础上，把原来的"快速体验示例"整块换成了 **16 类实例总览**：

- 16 张卡片，每类显示实例缩略图（带标注框）与样本数；点击可放大看原图，不引入筛选侧栏、搜索、多目标、随机抽等浏览功能
- 展示图来自 `models/gallery_picks.json`（逐张人工核对过的清单），不足 3 张的类只展示实有张数，不用错标图凑数
- 缩略图由服务端 `/thumb/<split>/<stem>?w=` 实时生成并缓存，标注框颜色与类别一一对应

数据来源 `dataset/yolo16_final`，当前 **709 张**：

| 类别 | 张数 | 类别 | 张数 |
|---|---|---|---|
| 苹果-未熟/成熟/完熟 | 57/57/62 | 橙子-未熟/成熟/完熟 | 7/15/24 |
| 香蕉-未熟/成熟/完熟 | 45/58/53 | 草莓-未熟/成熟/完熟 | 34/39/37 |
| 葡萄-未熟/成熟/完熟 | 56/57/57 | 腐烂水果 | 51 |

### 种类清洗（curate_dataset.py）

对每张图取标注框内区域跑 ImageNet 分类器，按物种概率剔除错标：苹果 <0.30、香蕉 <0.50、橙子 <0.35、草莓 <0.20 移出；橙子再加长宽比 0.85–1.25（芒果偏长卵形）。共移出 **121 张**，存到 `dataset/rejected_species/`（含 `_log.json` 逐条原因），可回滚。

```bash
python pick_gallery.py     # 生成 contacts_pick/ 每类候选拼图，供逐张目视核对
python curate_dataset.py   # 按种类闸门清洗，移出错标样本
```

### 识别链路为什么改成“检测 + 种类 + 颜色”

`train16.py` 直接在 16 类上训练，验证集只有 68.7%，而且错误**几乎全部集中在同一水果的相邻档位**（苹果-M→苹果-H、草莓-M→草莓-H）。原因是这套三档标签本身就是按颜色三分位切出来的，相邻两档高度重叠；再加上真实照片里草莓/葡萄多为田间远景，整图缩放会让模型盯着叶片而不是果实。

现在的三段式链路把三个问题拆开解决：

| 环节 | 做法 | 实测 |
|---|---|---|
| 定位 | YOLOv8n（`models/fruit_yolov8_detector.pt`）只取框，不用它的类别 | 田间草莓可框出单个果实 |
| 种类 | MobileNetV2 6 类（5 种水果 + 腐烂），输入=检出的果实框，TTA = 原图+中心 0.8 裁剪+水平翻转，整图概率权重 0.25 | 验证集 **147/148 = 99.3%**，TTA 100% |
| 成熟度 | 框内颜色语义：绿多→未熟、褐斑多→完熟、其余→成熟，阈值按种类微调 | 红苹果→成熟、青苹果→未熟、带黑斑香蕉→成熟/完熟 |

```bash
python train_species.py        # 训练 6 类种类模型 → models/species_mobilenetv2.pth
python regression_check.py     # 三张真实照片回归（红苹果 / 带斑香蕉 / 田间草莓）
```

**为什么成熟度不用 CNN 判**：三档标签由颜色分位生成，让模型去学“相邻档之间的噪声”收益很低；而颜色规则可解释、可复核，也与标签定义同源。页面上的“颜色证据：绿 x% · 褐斑 y% · 着色度 z%”就是这条规则的输入，答辩时可以直接讲清依据。

**旧方案的坑（已修）**：原先 4 类闭集 + 马氏距离/近邻双判据做“未知类别拒绝”，导致带黑斑的成熟香蕉（马氏距离 77.4 > 阈值 71.5）被拦成“非支持类别”；苹果则被强行分到芒果。现在物种判断交给 6 类模型，只有种类置信度 < 0.40 才提示“未能确定”。


### 局限

- 葡萄-未熟、橙子-未熟、腐烂水果的干净样本很少，页面只展示 1–2 张
- 成熟度由颜色语义给出（绿→未熟、褐斑→完熟），是外观可解释判据，仍不是农学糖度/硬度指标；数据集里的 L/M/H 标签是按颜色分位生成的相对次序，两者口径已不同，页面文案已注明
