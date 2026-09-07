# TASKS

## 1. 当前状态

- [x] Private Git 仓库已建立
- [x] 项目规范已建立
- [x] 正式训练数据已获得
- [ ] 数据路径完成配置
- [ ] 三人功能分支建立
- [ ] 服务器训练环境确认

---

## 2. 第一阶段并行任务

### 成员 A：模型与实验

- [ ] PyTorch / CUDA / Ultralytics 环境验证
- [ ] RGB baseline 工程准备
- [ ] 训练入口
- [ ] 推理入口
- [ ] 提交格式工具
- [ ] `tests/test_submission_format.py`
- [ ] E001 配置准备

### 成员 B：数据分析

- [ ] 训练数据完整性检查
- [ ] 标签合法性检查
- [ ] 12 类统计
- [ ] bbox 分布统计
- [ ] 固定 80/20 train / val
- [ ] seed = 2026
- [ ] 生成 `train.txt`
- [ ] 生成 `val.txt`

### 成员 C：三模态数据分析

- [ ] RGB / IR / Depth 可视化
- [ ] 三模态尺寸检查
- [ ] Infrared 三通道差异统计
- [ ] Depth dtype / range / invalid ratio
- [ ] 人工检查代表性样本空间对应
- [ ] 生成三模态分析报告

三条任务并行执行。

---

## 3. Phase 2 - 固定数据划分

成员 B 完成数据检查后：

- [ ] 生成 `data/splits/train.txt`
- [ ] 生成 `data/splits/val.txt`
- [ ] 提交 `feature/data-analysis`
- [ ] Pull Request 合并到 `main`
- [ ] 全队确认 split 固定

该步骤是所有正式训练实验的共同前置条件。

---

## 4. Phase 3 - E001 RGB Baseline

前置条件：

- 数据检查完成；
- 固定 split 已进入 `main`；
- RGB baseline 工程可运行；
- 训练环境验证通过。

执行：

- [ ] `E001_rgb_baseline.yaml`
- [ ] RGB full training
- [ ] validation
- [ ] per-class AP
- [ ] inference
- [ ] 生成官方格式 TXT
- [ ] submission format check
- [ ] 初赛第一次提交
- [ ] 确认成绩高于官方 baseline

---

## 5. Phase 4 - 单模态消融

- [ ] E002 IR-only
- [ ] E003 Depth-only
- [ ] 比较 RGB / IR / Depth 的 per-class AP
- [ ] 结合三模态统计解释差异

---

## 6. Phase 5 - 简单多模态 Baseline

- [ ] E004 RGB+IR
- [ ] E005 RGB+Depth
- [ ] E006 RGB+IR+Depth Early Fusion
- [ ] 建立完整消融表

---

## 7. Phase 6 - Feature Fusion

根据 E001~E006 结果选择候选方法：

- [ ] Add
- [ ] Concat + 1x1 Conv
- [ ] Gated Fusion
- [ ] 轻量 Attention

一次实验只验证一个主要改动。

---

## 8. Phase 7 - Adaptive / Reliability Fusion

重点候选方向：

- [ ] Modality Quality
- [ ] Adaptive Weight
- [ ] Depth Invalid-region Reliability
- [ ] Low-light IR Utilization

---

## 9. Phase 8 - 专项优化

根据数据统计和验证结果选择：

- [ ] P2 Head / 小目标
- [ ] 输入分辨率
- [ ] 类别不均衡
- [ ] Depth normalization
- [ ] 模态专用增强

---

## 10. Phase 9 - Leaderboard

- [ ] 记录每次正式提交
- [ ] 以固定 val 为主要模型选择依据
- [ ] 排行榜用于阶段验证
- [ ] 保持泛化优先
- [ ] 保留提交对应的 commit、配置和权重

---

## 11. Phase 10 - 复赛 / 半决赛准备

- [ ] 完整训练代码
- [ ] 完整推理代码
- [ ] 环境说明
- [ ] 模型权重说明
- [ ] 消融实验表
- [ ] 模型架构图
- [ ] 创新点说明
- [ ] 局限性说明
- [ ] 技术报告
