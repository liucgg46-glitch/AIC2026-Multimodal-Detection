# AIC2026 项目统一开发规则

> 适用项目：AIC2026 面向城市场景的视觉多模态目标检测
> 用途：作为团队成员与 Codex 在新对话、新分支和新开发阶段中的统一约束。
> 原则：若本文件与赛事官方规则冲突，以赛事官方规则为最高优先级；若与仓库中更新后的正式规范冲突，以最新已合并到 `main` 的正式规范为准。

---

## 1. 基本原则

1. 开始任何任务前，先读取当前仓库中的项目规范、数据规范、团队协作规范和本文件。
2. 以当前 Git 仓库实际状态为准，不依赖旧对话中的记忆。
3. 不为“代码更漂亮”进行无必要重构。
4. 只修改完成当前任务所必需的文件。
5. 如果任务需要改变目录结构、公共接口、数据口径或其他成员负责的核心文件，先汇报原因和最小修改方案，不得擅自实施。
6. 遇到不确定的数据含义、赛事规则或实验口径时，先汇报，不得自行猜测后写入正式流程。

## 2. 路径与环境规则

1. 项目代码中禁止硬编码个人绝对路径，例如 `D:\Competition\...`、`C:\Users\...`。
2. 优先使用项目相对路径、`pathlib.Path`、配置文件参数或命令行参数。
3. 输出到 JSON、CSV、Markdown、YAML 等 Git 跟踪文件中的路径，必须使用项目相对路径，不得记录个人电脑绝对路径。
4. 本地开发环境与服务器环境可以不同，但核心代码不得依赖某一台个人电脑的固定路径或设备编号。
5. 从 Windows 切换到 Linux/服务器时，不应重写核心训练、推理或数据处理逻辑。

## 3. 官方数据规则

1. `data/raw/` 中的官方数据视为只读。
2. 禁止覆盖、删除、重命名、手工修改或重新编码官方原始图片和标签。
3. 所有清洗、缓存、兼容视图和派生数据必须写入 `data/processed/`。
4. 官方训练集三模态必须按同一 `stem` 对应：Visible / RGB、Infrared、Depth。
5. JPG/PNG 混合是官方数据特征，不得写死单一扩展名。
6. 同一 `stem` 若出现多个候选文件，应视为歧义并停止处理，不得静默选择其中一个。

## 4. 固定 train / val 规则

正式实验统一使用：

```text
data/splits/train.txt：1600
data/splits/val.txt：400
```

规则：

1. split 中统一记录无路径、无扩展名的 `stem`。
2. 禁止自行重新随机划分。
3. 所有正式实验必须使用同一固定 split，保证实验结果可比较。
4. `train.txt` 与 `val.txt` 各自必须唯一、不能有交叉，合计必须覆盖全部 2000 个正式训练样本。
5. Smoke test 可以使用单独的 smoke split，但不得覆盖或修改正式 `train.txt` / `val.txt`。

## 5. 标签规则

1. 官方原始标签位于：

```text
data/raw/train/labels/
```

2. 正式实验使用清洗标签：

```text
data/processed/train/labels_clean/
```

3. `labels_clean` 必须通过 Git 跟踪的确定性脚本生成，不依赖人工复制或某一成员本地私有文件。
4. 清洗脚本只能读取 `data/raw/`，不得修改 `data/raw/`。
5. 当前正式 clean-label 预期复检结果：

```text
标签文件：2000
有效目标：15194
errors：0
retained warnings：59
passed：true
```

6. 正式训练不得误用未清洗的 raw labels。
7. 如果发现新的标签异常，不得自行修改正式清洗规则；先汇报并由团队确认后再改。

## 6. 初赛测试集规则

`PHASE_1_1000` / `data/raw/test/` 只允许用于模型推理、生成提交 TXT 和提交格式检查。

禁止用于训练、本地验证、参数选择、人工标注、伪标签训练、根据测试结果反向修改 GT 或任何形式的数据泄漏。

无检测结果的测试图仍必须生成同名空 TXT。

## 7. Git 与分支规则

1. `main` 只保存已审查、已验证、可运行的稳定版本。
2. 不直接在 `main` 上进行日常开发。
3. 每项功能使用独立分支，功能完成后通过 PR 审查再合并 `main`。
4. 未经明确授权，不得自动 commit、push、merge、force push 或重写历史。
5. 如果 merge 出现 conflict：立即停止并汇报冲突文件和原因，不擅自选择 `ours` / `theirs`。
6. 已发布的 `main` 历史不要为了修改 Commit Summary 而 force push。
7. Commit Summary 统一使用中文，简洁描述本次改动。
8. 删除已完成分支前必须确认 PR 已合并、没有未合并独立提交、没有未 push 工作。

## 8. Git 跟踪范围

应进入 Git 的内容主要包括：源代码、小型配置、`data/splits/`、测试代码、必要的小型统计结果、项目文档和实验配置。

通常不应进入 Git：

```text
data/raw/
data/processed/
runs/
weights/
outputs/submissions/
outputs/visualization/
```

以及：

```text
*.pt
*.pth
*.ckpt
*.onnx
*.engine
```

每次完成任务后必须检查 `git status`，避免误跟踪大文件和派生数据。

## 9. 当前成员职责边界

### 成员 A：模型与实验

主要负责 YOLO 集成、RGB baseline、训练入口、推理入口、提交格式检查、实验配置与编号以及后续模型主线。

主要文件范围：

```text
scripts/train/
scripts/inference/
configs/experiments/
tests/test_submission_format.py
```

### 成员 B：数据分析

主要负责数据完整性检查、标签合法性检查、类别分布统计、bbox 统计、固定 train / val，以及后续实验的数据侧诊断与误差分析。

### 成员 C：多模态数据分析

主要负责 RGB / IR / Depth 可视化、三模态对应关系、IR 特性、Depth 特性、配准/黑边/有效视场分析，以及后续多模态预处理建议。

不同成员不得无必要修改其他成员负责的核心文件。

## 10. RGB 正式训练数据接口

正式 RGB YOLO 兼容视图：

```text
data/processed/rgb_yolo/
```

要求：

- 图像来源：`data/raw/train/visible/`
- 标签来源：`data/processed/train/labels_clean/`
- 使用固定 `train.txt` / `val.txt`
- 优先使用 hardlink
- 不复制或修改官方原始图片
- 兼容视图属于派生数据，不进入 Git

## 11. 多模态处理规则

1. 三模态尺寸一致不代表逐像素严格对齐。
2. Visible / IR / Depth 的 resize、crop、flip、affine、perspective 必须共享完全相同的几何增强参数。
3. Visible GT bbox 作为统一监督基准。
4. 不得根据 IR / Depth 残余偏移平移、修改或重写 Visible GT bbox。
5. IR / Depth 残余偏移只可用于掩码、鲁棒融合、后续配准实验或误差分析。
6. IR 应独立进行强度预处理，黑边应保留或显式掩码处理。
7. Depth 必须区分：
   - PNG Depth：`uint16` 单通道
   - JPG Depth：`uint8` 三通道
8. JPG Depth 的 `0~255` 不得解释为与 PNG 相同的物理深度。
9. PNG / JPG Depth 不得静默使用完全相同的数值归一化规则。
10. PNG Depth 应保留有效区域 / 零值掩码信息。

## 12. Smoke 与正式实验规则

Smoke test 的目的只有：验证数据读取、标签读取、训练能开始和结束、loss 为有限值、权重与日志能生成，以及推理与提交链路可用。

Smoke 结果不得作为正式模型性能结论。

正式实验统一编号，例如：

```text
E001_RGB_BASELINE
E002_...
E003_...
```

正式实验开始前必须先冻结配置并记录 Git commit。

## 13. 正式实验必须记录

每次正式实验至少记录：

- 实验编号
- Git commit SHA
- 模型结构
- 预训练权重及来源
- 数据版本
- split
- clean-label 状态
- imgsz
- epochs
- batch
- optimizer
- learning rate
- scheduler
- seed
- device
- workers
- AMP
- Ultralytics 版本
- Python / PyTorch / CUDA 环境
- 训练时间
- 峰值显存
- best epoch
- best checkpoint
- train / val loss
- mAP@50-95
- mAP50
- Precision
- Recall
- per-class AP

## 14. 服务器训练规则

1. 正式训练优先在 RTX 3090 服务器完成。
2. 第一次上服务器时先检查环境，不直接启动正式 E001。
3. 至少检查：`nvidia-smi`、Python、PyTorch、torchvision、CUDA、cuDNN、Ultralytics、GPU 型号与显存、磁盘空间。
4. 服务器上先运行 GPU smoke，再冻结 batch、workers、device、AMP 和 learning rate。
5. GPU smoke 通过后，才创建/冻结正式实验配置并启动 E001。

## 15. Codex 操作规则

每次 Codex 新对话开始时：

1. 先读取本文件和仓库当前正式规范。
2. 先检查当前 branch、HEAD 和 `git status`。
3. 以当前仓库为唯一代码事实来源，不依赖旧对话记忆。
4. 若任务只是“审查”，必须保持只读。
5. 若任务要求修改，只做最小必要修改；修改后运行相关测试、执行 `git diff --check`、检查 `git status`。
6. 未明确授权时，不自动 commit / push / merge。
7. 如发现需要修改公共架构、数据口径或其他成员核心文件，先汇报方案。
8. 完成后必须汇报新增/修改文件、实际执行内容、测试结果、是否存在阻塞问题以及最终 Git 状态。

## 16. 新 Codex 对话推荐开场

```text
这是 AIC2026 多模态目标检测项目的新 Codex 对话。

开始任务前，请先读取：
- docs/PROJECT_RULES.md
- 当前仓库中的项目规范
- 数据规范
- 团队协作规范

并检查当前 branch、HEAD 和 git status。

后续以仓库当前状态为准，不依赖旧对话记忆。
未明确授权时不要自动 commit、push 或 merge。
```
