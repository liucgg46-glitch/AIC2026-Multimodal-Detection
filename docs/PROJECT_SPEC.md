# PROJECT_SPEC

## 1. 项目名称

AIC2026 算法挑战赛——面向城市场景的视觉多模态目标检测

## 2. 任务定义

项目使用空间对齐的三模态图像完成 12 类目标检测：

- Visible / RGB
- Infrared
- Depth

类别定义：

```text
0  person
1  boat
2  animal
3  seat
4  sign
5  bicycle
6  car
7  ball
8  light
9  garbage can
10 uav
11 tricycle
```

其中：

- `sign` 包含路牌、标语、标志；
- `bicycle` 包含自行车和双轮电动车；
- `light` 包含路灯和室内照明灯。

## 3. 数据组织

项目统一采用以下逻辑目录：

```text
data/
├── raw/
│   ├── train/
│   │   ├── visible/
│   │   ├── infrared/
│   │   ├── depth/
│   │   └── labels/
│   └── test/
│       ├── visible/
│       ├── infrared/
│       └── depth/
├── splits/
│   ├── train.txt
│   └── val.txt
└── processed/
```

说明：

- `data/raw/train/`：正式训练数据；
- `data/raw/test/`：当前阶段官方测试数据；
- `data/splits/`：从正式训练集中固定划分的 train / val 样本列表；
- `data/processed/`：必要时保存派生数据、缓存或格式转换结果。

官方原始数据保持只读，不覆盖、不重新编码。

## 4. 阶段目标

### 4.1 初赛目标

在当前开发周期内完成：

1. 数据完整性与标签检查；
2. 类别、bbox 与三模态数据统计；
3. 固定 train / val 划分；
4. RGB-only YOLO baseline；
5. 初赛推理与合法提交；
6. 获得高于赛事 baseline 的有效成绩；
7. 建立第一版三模态 baseline。

### 4.2 长期目标

- 持续提高排行榜成绩；
- 进入复赛与半决赛；
- 形成可复现的完整训练与推理流程；
- 完成消融实验、技术报告与模型说明；
- 以国奖为目标进行后续优化。

## 5. 技术栈

优先使用：

- Python 3.8+
- PyTorch
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- Matplotlib

在没有明确实验依据的情况下，不随意更换整个训练框架。

## 6. 评价指标

主指标：

```text
mAP@50-95
```

辅助记录：

- mAP50
- Precision
- Recall
- per-class AP
- best epoch

模型选择与优化优先依据 `mAP@50-95`。

## 7. 赛事约束

### 7.1 数据

训练仅使用赛事官方训练数据。

允许使用赛事规则许可的公开预训练权重，例如：

- ImageNet
- COCO
- Objects365

测试数据：

- 仅用于推理与生成提交结果；
- 不参与训练；
- 不参与本地验证集划分；
- 不人工标注；
- 不修改；
- 不公开传播。

### 7.2 推理

最终训练与推理流程必须能够离线完成。

### 7.3 模型

不采用赛事规则禁止的简单多模型投票、平均等集成方式。

### 7.4 原创性与可复现性

正式参赛代码、模型和方法应保持原创、可解释、可复现。

## 8. 开发阶段

```text
Phase 1  数据检查与统计
Phase 2  固定 train / val
Phase 3  RGB-only baseline
Phase 4  IR-only / Depth-only 消融
Phase 5  简单多模态融合
Phase 6  Feature-level Fusion
Phase 7  Adaptive / Reliability Fusion
Phase 8  小目标、Depth、类别不均衡等专项优化
Phase 9  排行榜优化
Phase 10  复赛 / 半决赛材料整理
```

## 9. 项目原则

1. 正式训练集是唯一训练数据源。
2. 测试集与训练集严格隔离。
3. 固定 train / val 后，所有可比较实验统一使用同一划分。
4. 正式实验统一编号为 `E001、E002……`。
5. 一次实验尽量只改变一个主要变量。
6. 正式训练前记录 Git commit 和实验配置。
7. 先完成 RGB baseline，再进入三模态模型优化。
8. 先基于数据统计和实验结果确定优化方向。
9. `main` 保持稳定、可运行、可复现。
