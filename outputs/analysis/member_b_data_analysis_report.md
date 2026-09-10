# AIC2026 成员 B 数据分析报告

## 1. 文档说明

本报告随成员 B 的数据检查、标签分析和固定数据划分任务持续更新，用于最终交付与 Pull Request 说明。

当前状态：B1～B4 均已完成；seed=2026 的确定性 group-aware 多标签分层已通过数量、覆盖、分布、group 零交叉和重复运行校验。官方原始标签中的 5 行取值错误和 1 个重复框已在独立派生副本中最小修复，复检结果为 0 错误、59 个保留的边缘警告；官方 `data/raw` 未修改。

## 2. 环境与代码版本

- 项目路径：`D:\AIC\AIC2026-Multimodal-Detection`
- 开发分支：`feature/data-analysis`
- Conda 环境：`aic2026`
- Python：3.10.21
- Python 解释器：`D:\Miniconda3-latest\envs\aic2026\python.exe`
- PyTorch：2.14.0+cpu
- Torchvision：0.29.0+cpu
- CUDA：False

## 3. 数据来源与目录

正式训练压缩包：

```text
D:\AIC\已加速- AIC2026_Train_2000.zip
```

解压后的只读训练数据：

```text
data/raw/train/
├── visible/
├── infrared/
├── depth/
└── labels/
```

原始 ZIP 和解压后的官方数据均未修改。

## 4. B1 数据完整性检查

### 4.1 实现文件

```text
scripts/data/check_dataset.py
```

脚本执行以下只读检查：

- 四个目录的文件数量；
- 基于 filename stem 的一一对应关系；
- 缺失文件和重复 stem；
- 图像是否可由 OpenCV 正常解码；
- Visible、Infrared、Depth 的尺寸是否一致；
- 各模态 dtype、通道数、尺寸和扩展名分布；
- Depth 的最小值、最大值、均值、中位数摘要和零值比例。

运行命令：

```powershell
python scripts/data/check_dataset.py --train-root data/raw/train --output-dir outputs/analysis
```

### 4.2 文件与对应关系

| 模态 | 文件数 | 唯一 stem | 可读取文件 |
|---|---:|---:|---:|
| Visible | 2000 | 2000 | 2000 |
| Infrared | 2000 | 2000 | 2000 |
| Depth | 2000 | 2000 | 2000 |
| Labels | 2000 | 2000 | 不适用 |

- 四类文件的并集 stem 数：2000；
- 四类文件的共同 stem 数：2000；
- 缺失文件：0；
- 重复 stem：0；
- 无法读取图像：0；
- 同一样本三模态扩展名不一致：0；
- 三模态尺寸不一致：0。

### 4.3 图像格式与尺寸

| 数据批次 | 数量 | Visible | Infrared | Depth |
|---|---:|---|---|---|
| PNG 样本 | 1851 | uint8，3 通道，1080×1920 | uint8，3 通道，1080×1920 | uint16，单通道，1080×1920 |
| JPG 样本 | 149 | uint8，3 通道，360×640 | uint8，3 通道，360×640 | uint8，3 通道，360×640 |

149 个 JPG 样本在三个图像模态中数量相同、stem 对应且尺寸一致，说明它们构成数据集中的一组完整样本，并非随机缺失或单个文件损坏。比赛官方说明已明确：RGB、Infrared 和 Depth 图像允许 PNG 与 JPG 混合，但同一个样本的三模态图像格式保持一致。

### 4.4 Depth 初步统计

合并读取 2000 张 Depth 后的初步统计：

- 全局最小值：0；
- 全局最大值：19999；
- 像素加权均值：5745.8760；
- 每图中位数的中位数：4787；
- 全体像素零值比例：29.0589%。

注意：上述合并统计同时包含 1851 张 uint16 PNG 和 149 张 uint8 JPG。由于两组编码尺度不同，后续进行 Depth 分布分析或模型输入处理时必须分组统计，不能直接把 JPG 像素值解释为毫米。

### 4.5 JPG Depth 复核

对 149 张 JPG Depth 的通道进行额外检查：

- dtype：uint8；
- 通道数：3；
- 数值范围：0～255；
- 121 张三个通道完全相同；
- 其余图片的通道平均绝对差小于 0.1，符合灰度内容经 JPEG 压缩后产生的微小通道差异；
- 149 个 JPG stem 同时存在对应的 Visible、Infrared 和 Label。

结论：该格式差异属于比赛官方数据的正常混合编码，不是数据损坏。检查脚本按扩展名分别验证 Depth：PNG 预期为 uint16 单通道，JPG 预期为 uint8 灰度内容（OpenCV 可能读取为 1 或 3 通道）。同时检查同一样本的 RGB、Infrared 和 Depth 扩展名是否一致。

### 4.6 输出文件

```text
outputs/analysis/dataset_check_summary.json
outputs/analysis/dataset_issues.csv
```

脚本已根据官方说明更新，不再把符合预期的 JPG Depth 记为异常。重新全量检查结果为 `passed=true`、问题数 0。`dataset_issues.csv` 当前仅包含表头；后续若数据改变，该文件将记录真正的缺失、解码失败、同样本扩展名不一致、尺寸不一致或与对应文件格式不符的数据。

## 5. B2 标签合法性检查

### 5.1 实现与运行

实现文件：

```text
scripts/data/analyze_labels.py
```

运行命令：

```powershell
python scripts/data/analyze_labels.py
```

脚本只读取 `data/raw/train/labels`，不会修改标签。检查内容包括：5 字段格式、0～11 类别编号、数值及有限性、归一化中心坐标、正宽高、bbox 边界、空标签文件和完全重复标注。

### 5.2 检查结果

| 项目 | 结果 |
|---|---:|
| 标签文件 | 2000 |
| 空标签文件 | 1 |
| 非空标注行 | 15195 |
| 通过官方五字段取值检查的标注 | 15190 |
| 官方取值错误 | 5 |
| 附加质量警告 | 60 |
| 涉及问题的文件 | 57 |

空标签文件为 `shuming_102_00000228.txt`。空标签可能表示该图没有目标，因此仅记录，不作为错误，也不删除对应样本。

错误及警告分类：

| 级别 | 类型 | 数量 | 解释 |
|---|---|---:|---|
| 错误 | `invalid_center_coordinate` | 3 | `norm_center_x > 1`，违反官方 `[0,1]` 范围 |
| 错误 | `invalid_bbox_size` | 2 | `norm_h > 1`，违反官方 `(0,1]` 范围 |
| 警告 | `bbox_out_of_bounds` | 59 | 字段自身合法，但换算后的 bbox 边缘超出图像 |
| 警告 | `duplicate_annotation` | 1 | 同一标签文件存在完全相同的两行标注 |

字段数错误、非法类别编号、非数值 bbox、NaN/Infinity 均为 0。59 个边缘越界框中，归一化越界量中位数约 0.00573，最大约 0.11204；其中 24 个超过 0.01，不能全部解释为小数舍入误差。检查器将边缘越界列为质量警告，是因为赛题明文要求的是 `x/y/w/h` 各字段位于合法范围，而边缘检查是团队增加的更严格检查。

原始标签检查结论：B2 不能写成“官方原始标签完全无异常”。5 行确实违反官方坐标/宽高取值要求，另有 60 条值得团队复核的标注质量警告。后续仅在 `data/processed/train/labels_clean` 中按团队内部口径修复 5 个错误并移除 1 个重复框，始终未裁剪、删除或覆盖官方原始标签；派生副本的复检结果见第 11 节。

### 5.3 初步类别计数

以下计数包含全部 15190 行通过官方五字段取值检查的标注；边缘越界警告和重复行仍计入，5 行官方取值错误不计入：

| ID | 类别 | 目标数 | 出现图片数 |
|---:|---|---:|---:|
| 0 | person | 5545 | 1114 |
| 1 | boat | 135 | 62 |
| 2 | animal | 3020 | 798 |
| 3 | seat | 641 | 322 |
| 4 | sign | 788 | 424 |
| 5 | bicycle | 665 | 214 |
| 6 | car | 1949 | 459 |
| 7 | ball | 95 | 78 |
| 8 | light | 1835 | 542 |
| 9 | garbage can | 282 | 231 |
| 10 | uav | 208 | 150 |
| 11 | tricycle | 27 | 23 |

每张图片的有效目标数：平均 7.595，中位数 5，最少 0，最多 66。类别分布明显不均衡，`tricycle` 仅 27 个目标、23 张图片，后续固定验证集时必须特别检查稀有类别覆盖情况。

### 5.4 输出文件

```text
outputs/analysis/label_check_summary.json
outputs/analysis/label_issues.csv
outputs/analysis/class_counts.csv
```

## 6. B3 类别与 bbox 分布统计

### 6.1 统计口径

- 使用 15190 行通过官方五字段取值检查的标注；
- 读取每个 stem 对应 Visible 图像的真实宽高，将归一化 bbox 转换为像素宽、高和面积；
- 分别处理 1920×1080 与 640×360 图像，不假设统一原始分辨率；
- 59 个边缘越界警告和 1 个重复标注仍保留在数据分布中，5 行官方取值错误不纳入 bbox 统计；
- `bbox_stats.csv` 保存逐目标明细，包括归一化值、像素值、宽高比和四个方向的越界像素量。

### 6.2 bbox 全局统计

| 指标 | 最小 | P5 | P25 | 中位数 | 均值 | P75 | P95 | 最大 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 宽度（px） | 8.00 | 18.00 | 37.61 | 69.00 | 110.69 | 141.00 | 334.00 | 1814.00 |
| 高度（px） | 7.00 | 23.00 | 46.00 | 76.66 | 138.55 | 165.60 | 445.00 | 1080.00 |
| 面积（px²） | 100.00 | 527.00 | 2091.00 | 5550.00 | 28401.96 | 17639.98 | 127311.23 | 1912680.35 |
| 相对面积 | 0.000048 | 0.000301 | 0.001118 | 0.002876 | 0.013838 | 0.008705 | 0.061396 | 0.922396 |
| 宽高比 W/H | 0.084 | 0.292 | 0.460 | 0.775 | 1.107 | 1.393 | 3.177 | 17.800 |

分布明显右偏：像素面积均值约 28402 px²，而中位数仅约 5550 px²。5% 的目标面积不超过约 527 px²，说明数据中确实存在大量较小目标；后续模型评估应重点关注小目标召回率。

### 6.3 类别不均衡

- 最大类别：`person`，5545 个目标，占有效目标约 36.50%；
- 最小非零类别：`tricycle`，27 个目标，占约 0.18%；
- 最大/最小非零类别数量比约 205.37；
- `tricycle` 只出现在 23 张图，`boat` 出现在 62 张图，`ball` 出现在 78 张图；
- 后续生成验证集时必须检查这些稀有类别是否进入 val，不能只看 1600/400 数量正确。

### 6.4 团队内部目标尺寸分档

团队已确认统一使用 `imgsz=640` 的参考 letterbox 尺度。该标准已写入 `docs/DATA_SPEC.md`，属于团队内部分析标准，不是赛事官方标准。即使后续训练使用其他 `imgsz`，统计口径也保持不变。

```text
scale = min(640 / W, 640 / H)
bbox_width_ref  = norm_w * W * scale
bbox_height_ref = norm_h * H * scale
bbox_area_ref   = bbox_width_ref * bbox_height_ref
```

分档为：tiny `<256`，small `[256,1024)`，medium `[1024,9216)`，large `>=9216`。

| 尺寸档 | bbox 数量 | 比例 |
|---|---:|---:|
| tiny | 3776 | 24.86% |
| small | 5327 | 35.07% |
| medium | 4899 | 32.25% |
| large | 1188 | 7.82% |

tiny 与 small 合计 9103 个，占 59.93%，说明小目标是本赛题的重要难点。各类别四档数量与比例已保存至 `class_size_distribution.csv`；其中 `ball` 的 tiny 占 65.26%，`bicycle` 的 tiny 占 44.06%，应重点关注这些类别的小目标召回。

### 6.5 输出文件

```text
outputs/analysis/class_counts.csv
outputs/analysis/bbox_stats.csv
outputs/analysis/dataset_summary.json
outputs/analysis/size_distribution.csv
outputs/analysis/class_size_distribution.csv
```

## 7. B4 最终 group-aware train / val 划分

### 7.1 分组审计依据

普通随机候选 split 的类别最大占比差为 5.70 个百分点、尺寸最大占比差为 4.25 个百分点，并且存在场景泄漏证据：128 个多样本前缀组中有 46 个跨越 train/val；纯数字相邻编号中有 110 对灰度相关系数不低于 0.90，其中 46 对跨越随机候选 split，而随机图像对中没有相关系数不低于 0.90 的样本。

因此最终划分采用 group-aware 多标签分层，不再使用图像级普通随机划分。

### 7.2 固定分组规则

- 带下划线且末段为数字帧号的 stem：去掉最后帧号，前缀相同者归为一组；
- 纯数字 stem：相邻编号差不超过 5，且灰度相关系数不低于 0.90 或 dHash 汉明距离不大于 5 时连接为一组；
- 同一 group 必须全部进入 train 或全部进入 val；
- 共生成 1703 个 group，其中 196 个为多样本 group，最大 group 为 6 张；
- 最终跨 train/val 的 group 数为 0。

文件名和图像相似度属于团队推断的候选序列信息，不是赛事提供的官方序列元数据；此限制已保留在 `group_audit_summary.json`。

### 7.3 确定性分层方法

- seed：2026；
- 目标：train 1600、val 400；
- 同时平衡 12 类目标数量、12 类图片覆盖数量和 tiny/small/medium/large 四档数量；
- 固定执行 5000 次候选搜索及 50000 次同组大小局部交换，接受 126 次改进；
- 优化目标为各统计量相对 20% val 目标的均方相对偏差；
- 算法、阈值、迭代次数及 seed 已写入代码和 `DATA_SPEC.md`，避免人工反复挑选 split。

### 7.4 最终数量与一致性

- train：1600 张、12151 个有效标注；
- val：400 张、3039 个有效标注；
- stem 交集：0；并集：2000；
- 12 个类别均同时出现在 train 和 val；
- 四个尺寸档均同时出现在 train 和 val；
- 最大类别占比差：0.0415 个百分点（boat）；
- 最大尺寸占比差：0.0185 个百分点（tiny）；
- 团队内部一致性结论：`basically_consistent=true`。

| 尺寸档 | 全部 | Train | Val | Val 占该档比例 |
|---|---:|---:|---:|---:|
| tiny | 3776 | 3021 | 755 | 19.99% |
| small | 5327 | 4261 | 1066 | 20.01% |
| medium | 4899 | 3919 | 980 | 20.00% |
| large | 1188 | 950 | 238 | 20.03% |

稀有类别也得到覆盖：tricycle 为 train 22、val 5，boat 为 train 109、val 26，ball 为 train 76、val 19。

### 7.5 可复现性与输出

重复运行 `make_split.py` 后文件哈希保持不变：

```text
train.txt SHA256: F34D3EDAE8EBD182A258FC7A03D200E313B2635C18831360AE72BD9CB29CB64E
val.txt   SHA256: E6D7222DE1BD8EB1C1F63346D32FDBECCB0121B21FFCD0777E55961B3D2156D9
```

```text
scripts/data/analyze_groups.py
scripts/data/group_stratification.py
scripts/data/make_split.py
data/splits/train.txt
data/splits/val.txt
outputs/analysis/group_audit_summary.json
outputs/analysis/group_candidate_pairs.csv
outputs/analysis/split_group_assignments.csv
outputs/analysis/split_summary.json
outputs/analysis/split_class_distribution.csv
outputs/analysis/split_size_distribution.csv
```

## 8. 赛题原始文件要求摘要

- 任务为经过空间对齐的 RGB、Infrared、Depth 三模态 12 类目标检测；
- 训练集为 2000 组；初赛、复赛、半决赛测试集各 1000 组；
- 训练标签正式格式为 `class_id x_center y_center width height`，共 5 字段；赛题第 3 页有一句漏写 `norm_h`，但第 2 页及提交格式均证明应按 5 字段执行；
- 主指标为 `mAP@50-95`，IoU 阈值为 0.50～0.95、步长 0.05，AP 使用 101 点插值；
- 训练仅允许使用官方训练数据，可使用 ImageNet、COCO、Objects365 公开预训练权重；
- 测试集仅用于推理，不得修改、标注、训练或传播；训练和推理不得依赖在线服务/API；
- 三模态空间增强必须共享同一组几何参数，避免破坏官方对齐关系；
- B4 的 80/20、seed=2026 是团队统一规范，不是赛题 PDF 强制值。

## 9. 官方数据特征与影响

### F001：Depth 混合编码（官方确认正常）

- 影响范围：149 / 2000，约 7.45%；
- 表现：JPG Depth 为 uint8 三通道，PNG Depth 为 uint16 单通道；
- 数据属性：比赛官方确认 PNG/JPG 混合格式正常，同一样本的三模态扩展名保持一致；
- 对 B1 完整性结论的影响：不作为异常，不影响 stem、可读性和空间尺寸检查；
- 对后续分析的影响：Depth 统计应按编码类型分组；
- 对后续训练的影响：不得把 JPG 的 0～255 直接当作毫米深度与 PNG 混合归一化；
- 当前处理：按格式分组记录和验证，不转换、不覆盖官方数据。

## 10. 后续任务

- [x] B1 文件数量与 stem 对应检查
- [x] B1 图像可读性与尺寸检查
- [x] B1 图像 dtype、通道和 Depth 初步统计
- [x] 官方确认 JPG/PNG 混合格式属于正常数据特征
- [x] 完整阅读并核对赛题原始 PDF 要求
- [x] B2 标签合法性检查并输出逐行问题清单
- [x] B3 类别、像素 bbox、相对面积、宽高比和不均衡统计
- [x] B3 按团队内部 `imgsz=640` letterbox 标准生成目标尺寸分档
- [x] B4 审计场景/连续帧相似性与随机 split 泄漏风险
- [x] B4 生成 seed=2026 的确定性 group-aware 多标签分层
- [x] B4 验证 1600/400、全覆盖、零交集、group 零交叉及分布一致性
- [x] B4 重复运行并记录 train/val SHA256
- [x] 生成独立的最小清洗标签副本并确认官方原始标签未改变
- [x] 清洗标签独立复检：0 错误、0 重复框、59 个保留警告
- [x] 本地复核：脚本编译、重复生成、split 集合与 Git 忽略规则检查通过
- [ ] 提交 `feature/data-analysis`
- [ ] 创建 Pull Request 合并至 `main`

## 11. 训练标签最小清洗与独立复检

### 11.1 处理范围

- 官方原始目录 `data/raw/train/labels` 保持只读，未覆盖、移动或删除任何文件；
- 派生标签写入 `data/processed/train/labels_clean`；
- `PHASE_1_1000` 是测试集，不参与本次处理，也不参与 train/val 划分；
- 只处理 B2 已逐行定位并目视复核的 6 行，不批量改写其余标注。

### 11.2 处理结果

- 5 个官方字段范围错误：由 YOLO 中心点与宽高还原四角，裁剪至 `[0, 1]` 后重新计算；
- 1 个完全相同的重复框：仅在派生副本中删除后出现的一行；
- 空标签 `shuming_102_00000228.txt`：目视无明显比赛目标，保留；
- 其余 59 个字段合法但边缘轻微越界的 warning：暂不改写；
- 输入、输出标签文件均为 2000 个；修复 5 个框，删除 1 个重复框；
- 全部源标签的处理前后 SHA256 一致，确认 `data/raw` 未改变；
- 未列入复核清单的标注行改动数为 0。

### 11.3 独立复检

使用 `analyze_labels.py` 对 `labels_clean` 重新完整检查：

```text
标签文件：2000
空标签：1（允许）
非空行：15194
有效目标：15194
错误：0
警告：59（均为保留的 bbox_out_of_bounds）
重复框：0
passed：true
```

处理与审计文件：

```text
scripts/data/prepare_clean_labels.py
outputs/analysis/clean_label_changes.csv
outputs/analysis/clean_label_summary.json
outputs/analysis/clean_label_validation/
```

本处理规则是团队内部训练预处理标准，不是赛事官方要求。`data/processed/` 已被 Git 忽略，其他成员应从各自的官方训练集运行同一脚本生成标签副本，不能上传比赛数据。
