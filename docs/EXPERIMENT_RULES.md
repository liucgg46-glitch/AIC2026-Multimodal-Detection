# EXPERIMENT_RULES

## 1. 实验编号

所有正式实验统一编号：

```text
E001
E002
E003
...
```

初始实验序列：

```text
E001 RGB-only baseline
E002 IR-only baseline
E003 Depth-only baseline
E004 RGB+IR
E005 RGB+Depth
E006 RGB+IR+Depth Early Fusion
```

后续实验依次递增编号。

## 2. 正式实验前置条件

正式 E001 启动前必须完成：

1. 数据完整性检查；
2. 标签合法性检查；
3. 固定 train / val；
4. 训练环境验证；
5. 对应代码已 commit；
6. 对应实验配置已保存。

## 3. 主评价指标

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

## 4. 固定数据划分

默认：

```text
train = 80%
val   = 20%
seed  = 2026
```

所有横向可比较实验必须使用同一 split。

## 5. 单变量原则

一次实验尽量只修改一个主要因素。

示例：

```text
E010 与 E009 相比，仅改变 Fusion 方式
```

其余主要配置保持一致。

## 6. 每个实验必须记录

至少记录：

- Experiment ID
- Date
- Branch
- Git commit
- Model
- Modalities
- Split version
- Image size
- Epochs
- Batch size
- Optimizer
- Learning rate
- Scheduler
- Pretrained weights
- Augmentation
- Seed
- GPU
- mAP50-95
- mAP50
- Precision
- Recall
- per-class AP
- Best epoch
- Leaderboard score（如提交）
- Notes
- Conclusion

## 7. 配置文件

每个正式实验建议对应一个配置文件：

```text
configs/experiments/Exxx_xxx.yaml
```

禁止只在命令行临时修改关键参数而不记录。

## 8. Git 与训练

正式训练前：

- 相关代码应已 commit；
- 记录当前 commit hash；
- 训练结果应能追溯到具体代码与配置。

## 9. 服务器调度

正式训练按实验编号统一管理。

多 GPU 并行实验时，每个实验必须明确：

- Experiment ID
- GPU
- 配置
- commit

## 10. 初赛提交目标

初赛提交按以下顺序推进：

1. 输出格式合法；
2. 成绩高于官方 baseline；
3. 获得有效成绩；
4. 再持续优化排行榜。

## 11. 数据与规则合规

所有正式实验应满足：

- 仅使用官方训练数据；
- 测试集不参与训练；
- 不人工修改测试结果；
- 最终推理不依赖在线 API；
- 不采用违规的简单多模型 ensemble。

## 12. 复赛与半决赛准备

从当前阶段开始保留：

- 数据预处理代码；
- 训练代码；
- 推理代码；
- 环境依赖；
- 权重说明；
- 实验配置；
- 消融实验记录；
- 创新点与局限性说明。

确保后续能够复现和整理技术报告。
