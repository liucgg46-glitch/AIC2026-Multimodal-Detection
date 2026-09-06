# Experiment Rules

## Experiment ID

所有正式实验统一编号：

E001
E002
E003
...

## Baseline Sequence

E001 = RGB-only baseline

后续编号依次递增。

## Reproducibility

默认随机种子：

2026

所有可比较实验必须：

- 使用相同 train/val split
- 使用相同评价指标
- 记录完整配置
- 保存 Git commit
- 保存最佳 checkpoint 信息

## One Main Variable Rule

一次实验尽量只修改一个主要因素。

错误示例：

同时：

- 换 backbone
- 换 loss
- 加 attention
- 修改数据增强

然后比较成绩。

这样无法确定提升来源。

## Required Metrics

至少记录：

- mAP50-95
- mAP50
- Precision
- Recall
- per-class AP

## Required Record

每个实验记录：

- Experiment ID
- Date
- Git commit
- Model
- Modalities
- Image size
- Epoch
- Batch size
- Optimizer
- Learning rate
- Augmentation
- Validation score
- Leaderboard score
- Notes