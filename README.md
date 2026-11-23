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
```

> 说明：`detect_app.py` 需要 `dataset/yolo16/dataset_meta.json`（已随仓库提供）来读取类别名。模型权重在 `models/fruit_yolov8_detector.pt`。

## 项目结构

| 文件 | 作用 |
|---|---|
| `detect_app.py` / `static/detect_index.html` | 升级版检测演示网页（主入口，端口 5001） |
| `detect_photo.py` | 命令行单图检测 |
| `train_detector.py` / `finetune_detector.py` | 16 类 YOLOv8 训练 / 微调 |
| `prepare_yolo16.py` / `prepare_persimmon.py` | 数据整理管线：把多个开源集映射成统一 16 类 YOLO 格式 |
| `models/fruit_yolov8_detector.pt` | 最终检测模型权重 |
| `models/detection_evaluation.json` | 评估报告与数据来源 |
| `app.py` / `train.py` / `predict.py` | 旧版整图分类（MobileNetV2，端口 5000） |

## 数据来源

数据集由多个开源集合并整理而成，原始压缩包体积约 2 GB，未随仓库提交。按 `prepare_yolo16.py` 和 `prepare_persimmon.py` 中的说明重新下载并整理即可复现：

- 苹果 / 香蕉 / 葡萄 的各成熟度 + 腐烂类：Kaggle `fruit-ripeness-unjex/2`（CC BY 4.0）
- 草莓三类：Kaggle `samatsauranbek/multi-class-strawberry-ripeness-detection-v2`（CC BY 4.0）
- 柿子三类：Fruits-360 的 Kaki 样本经色相增强 + 贴背景弱监督生成（见 `prepare_persimmon.py`）

整理后共 2794 张训练图、734 张验证图。

## 局限

柿子三类为弱监督生成，且所有数据场景较单一；mAP 是在这套拼装数据上测得，真实光照下的泛化能力有限。下一步计划补充自采实拍样本和外部验证集。

## License

代码 MIT。数据集遵循各自原始许可证（见上）。
