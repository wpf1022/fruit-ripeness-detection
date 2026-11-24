# 水果成熟度智能分级与检测系统 · 答辩PPT设计

## 全局视觉约定

### ① 画布与安全区
- 画布：1280×720，16:9。
- 内容页：左右边距 40px，内容宽度 1200px；A3 页眉 y=0–72，主标题置于页眉左侧；C 区 y=88–652；页脚 y=664–700。
- 封面 / 结束页：全出血，不显示常规页眉与页脚。
- 页脚固定：左侧“水果成熟度智能分级与检测系统”，右侧页码。

### ② 配色方案
- 主色：#0E3F8C（深蓝，页眉 / 深底）、#1E4FA8（宝蓝，编号 / 强调）。
- 辅色：#E8EFF8、#F0F5FC（浅蓝卡片底）、#D6DCE5（线框）。
- 强调色：#FFC107（数据黄，单页 ≤3%）、#D9534F（警示红，单页 ≤3%）。
- 底色：#FFFFFF、#F7F9FC。
- 正文色：#1A2230（主）、#4A5568（副）、#8B97A8（辅助）。
- 图表系列：#1E4FA8、#3D7BD9、#5B8FE6、#8AAEEA、#FFC107、#D9534F。

### ③ 字体系统
- 中文：思源黑 / 微软雅黑；英文：Arial。
- 字号阶梯：封面主标 58–68px；页面主标 34px；卡片标题 26px；正文 22–24px；表格 18–20px；KPI 数字 66–88px；注释 / 页脚 14–16px。

### ④ 视觉母题
- 内容页统一“深蓝顶条 + y=72 浅蓝细分隔线 + 左下项目名页脚”。
- 编号统一用深蓝方块 + 白色数字；结论条统一用浅蓝底 + 深蓝左侧文字。
- 全篇 L1 视觉统一为“真实检测效果图 / 真实实验样本图”，不使用卡通、3D、营销插画。

### ⑤ 配图清单
| 文件 | 用途 | 来源 | 使用页 |
|---|---|---|---|
| `detect_multi_strawberry.jpg` | 多目标检测主视觉，证明“一张图多个框” | 项目真实检测结果 | P1、P6 |
| `persimmon_high.jpg` | 柿子高成熟示例 | 项目真实检测结果 | P8 |
| `persimmon_mid.jpg` | 柿子中成熟示例 | 项目真实检测结果 | P8 |
| `persimmon_low.jpg` | 柿子低成熟示例 | 项目真实检测结果 | P8 |
| `sample_apple_h.jpg` | 原始样本示例 | demo_samples | P3、P4 |
| `sample_bad_fruit.jpg` | 腐烂类别示例 | demo_samples | P3、P4 |

---

## 逐页设计

### P1 封面
- title: 水果成熟度智能分级与检测系统
- type: cover
- role: hero
- rhythm: peak
- layout: 左侧深蓝标题区 + 右侧真实检测主视觉
- visual: L1 `detect_multi_strawberry.jpg` 占右侧 45%
- density: 字数约 80 / 图片 1 张 / 留白约 25%
- anti_pattern: 禁止装饰渐变、印章、金纹；禁止把英文副标做得比中文主标更抢眼
- description: 展示课题名称、英文副题、汇报人、时间，并用真实多目标检测图直接点明项目成果。

### P2 汇报提纲
- title: 汇报提纲
- type: catalog
- role: supporting
- rhythm: valley
- layout: 左侧深蓝编号栏 + 右侧 5 条提纲
- visual: 编号方块 + 提纲条
- density: 字数约 100 / 图片 0 张 / 留白约 20%
- anti_pattern: 禁止圆球编号；禁止英文比中文更醒目
- description: 依次介绍背景与问题、方案演进、三项创新点、实验对比、演示与展望。

### P3 背景与问题
- title: 为什么要做成熟度智能检测
- type: content
- role: supporting
- rhythm: valley
- layout: 左侧真实样本图 + 右侧三张痛点卡 + 底部结论条
- visual: L1 `sample_apple_h.jpg`、`sample_bad_fruit.jpg` 双图；三痛点卡
- density: 字数约 260 / 图片 2 张 / 留白约 14%
- anti_pattern: 禁止只写“效率低成本高”空话；禁止图片无说明
- description: 说明人工分级效率低且主观，整图分类不能定位多个目标，单一数据集难以覆盖多品类成熟度，因此需要一个多目标、细粒度、可演示的检测系统。

### P4 方案演进对比
- title: 从整图分类到多目标检测
- type: content
- role: supporting
- rhythm: valley
- layout: 双栏对照 + 中央箭头 + 底部结论
- visual: 左卡旧版 MobileNetV2 分类，右卡新版 YOLOv8n 检测，中央“升级”箭头
- density: 字数约 280 / 图片 0 张 / 留白约 12%
- anti_pattern: 禁止只堆功能词；禁止把旧版说得一无是处
- description: 对比输出形式、目标数量、定位能力、成熟度识别、演示能力，指出旧版是整图分类基线，新版是逐目标定位与分级。

### P5 技术路线与系统架构
- title: 技术路线：数据—模型—系统闭环
- type: content
- role: supporting
- rhythm: valley
- layout: 六步流程横向串联 + 下方数据规模与模型指标
- visual: SVG 流程：多源采集 → 标注统一 → 弱监督补齐 → YOLOv8n 训练/微调 → Flask 推理 → 可视化结果
- density: 字数约 240 / 图片 0 张 / 留白约 15%
- anti_pattern: 禁止把流程画成散点卡；禁止无箭头连接
- description: 展示项目完整链路，并标注 2794 张训练图、734 张验证图、16 类、8+4 epochs 等关键事实。

### P6 创新点一：16类细粒度多目标检测
- title: 创新点一：从“整图一个标签”到“逐目标分级”
- type: content
- role: hero
- rhythm: peak
- layout: 左侧真实检测大图 + 右侧三张机制卡 + 底部数据锚点
- visual: L1 `detect_multi_strawberry.jpg` 占左 52%；右侧机制卡
- density: 字数约 250 / 图片 1 张 / 留白约 12%
- anti_pattern: 禁止把 YOLOv8 说成自研算法；禁止用营销式形容词
- description: 说明 5 类水果 × 3 个成熟度 + 腐烂类共 16 类，单图多框输出中文类别和置信度，YOLOv8n 在 CPU 上约 19.1ms/张。

### P7 创新点二：多源数据重构
- title: 创新点二：多源异构数据的统一重构
- type: content
- role: supporting
- rhythm: valley
- layout: 上方三源输入箭头 → 中央映射规则 → 下方统一 YOLO 数据集
- visual: SVG 数据流 + 类别映射表
- density: 字数约 260 / 图片 0 张 / 留白约 14%
- anti_pattern: 禁止只列数据来源不写处理方式；禁止虚构类别数量
- description: 展示 Kaggle/Roboflow、草莓数据集、Fruits-360 三类来源，经过类别重映射、多边形转 bbox、坐标裁剪、类别均衡采样，形成统一 16 类 YOLO 数据。

### P8 创新点三：柿子弱监督样本生成
- title: 创新点三：用弱监督补齐缺失的柿子成熟度数据
- type: content
- role: supporting
- rhythm: valley
- layout: 上方五步流程 + 下方三张柿子成熟度结果图
- visual: `persimmon_high.jpg`、`persimmon_mid.jpg`、`persimmon_low.jpg`
- density: 字数约 240 / 图片 3 张 / 留白约 12%
- anti_pattern: 禁止把弱监督生成说成人工真实标注；禁止省略泛化局限
- description: 说明通过前景提取、HSV 色相扰动、随机背景粘贴、自动检测框生成，补齐柿子高/中/低三类样本，并诚实指出仍需真实实拍验证。

### P9 实验结果与对比
- title: 实验结果：精度与速度都能支撑演示
- type: content
- role: supporting
- rhythm: valley
- layout: 左侧指标条形图 + 右侧分类版/检测版对比表 + 底部判断
- visual: Chart 展示 mAP50、mAP50-95、Precision、Recall；表格展示方案对比
- density: 字数约 230 / 图片 0 张 / 留白约 10%
- anti_pattern: 禁止只摆数字不解释；禁止补造旧版不存在的指标
- description: 展示 mAP50 0.832、mAP50-95 0.671、Precision 0.810、Recall 0.788、19.1ms/张，并说明这些指标是在拼装数据集上取得，不等同于真实光照泛化能力。

### P10 系统功能演示
- title: 系统演示：从模型输出到可交互验证
- type: content
- role: supporting
- rhythm: valley
- layout: 左侧真实检测图 + 右侧功能清单 + 底部闭环结论
- visual: L1 `detect_multi_strawberry.jpg`；功能标签矩阵
- density: 字数约 250 / 图片 1 张 / 留白约 12%
- anti_pattern: 禁止把功能清单写成无解释的图标墙；禁止出现不存在的导出功能
- description: 展示上传/拖拽/示例图/摄像头开启与关闭/置信度调节/中文检测框/SQLite历史记录，强调形成了可训练、可评估、可演示的完整闭环。

### P11 局限与改进路线
- title: 局限与下一步：先把数据真实性补齐
- type: content
- role: supporting
- rhythm: valley
- layout: 左侧局限四卡 + 右侧改进路线三段
- visual: 2×2 局限卡 + 三阶段路线条
- density: 字数约 260 / 图片 0 张 / 留白约 12%
- anti_pattern: 禁止只写“未来可优化”；禁止回避弱监督与场景单一问题
- description: 说明柿子弱监督样本、公开数据场景单一、真实光照泛化未知、CPU 推理仍有提升空间；路线为自采实拍、外部验证集、轻量部署优化。

### P12 总结
- title: 项目结论
- type: ending
- role: hero
- rhythm: peak
- layout: 深蓝总结页 + 三条创新主张 + 一张检测图
- visual: `detect_multi_strawberry.jpg` 缩略图；三大关键词条
- density: 字数约 150 / 图片 1 张 / 留白约 30%
- anti_pattern: 禁止烟花、金色装饰；禁止长篇致谢
- description: 用三句话收束：细粒度多目标检测、多源数据重构、可交互系统闭环；明确创新定位是应用与工程创新，而非底层算法原创。
