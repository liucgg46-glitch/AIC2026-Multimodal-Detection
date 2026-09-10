# DATA_SPEC

## 1. 数据目录

项目统一使用：

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

若官方压缩包实际目录结构不同：

- 保持官方原始数据结构不变；
- 通过配置文件或命令行参数记录实际路径；
- 不为适配代码而修改官方原始文件。

## 2. 三模态格式

### 2.1 Visible / RGB

- 三通道；
- `uint8`；
- 像素范围 `[0,255]`。

### 2.2 Infrared

- 三通道；
- `uint8`；
- 三通道视觉信息高度一致；
- 本质为热辐射灰度信息的三通道存储。

### 2.3 Depth

- 单通道；
- `uint16`；
- 单位：毫米；
- 理论范围 `[0,65535]`；
- 主要有效距离约 `300~20000 mm`；
- `0` 或过小值代表无效区域。

读取 Depth 必须保留原始位深，例如：

```python
cv2.imread(path, cv2.IMREAD_UNCHANGED)
```

任何归一化、裁剪、伪彩色等处理仅作用于运行时数据或派生结果，不覆盖原始文件。

## 3. 图像对应关系

三模态数据空间对齐。

程序通过 filename stem 建立：

```text
visible
infrared
depth
label
```

的一一对应关系。

程序不得仅硬编码 `.jpg` 或 `.png`，应兼容官方实际图像扩展名。

## 4. 训练标签格式

每个目标占一行：

```text
class_id norm_center_x norm_center_y norm_w norm_h
```

字段要求：

- `class_id`：0~11；
- `norm_center_x`：归一化中心横坐标；
- `norm_center_y`：归一化中心纵坐标；
- `norm_w`：归一化 bbox 宽度；
- `norm_h`：归一化 bbox 高度；
- 四个坐标字段范围应在 `[0,1]`；
- `norm_w > 0`；
- `norm_h > 0`。

## 5. 类别映射

```text
0: person
1: boat
2: animal
3: seat
4: sign
5: bicycle
6: car
7: ball
8: light
9: garbage can
10: uav
11: tricycle
```

## 6. 训练数据检查

必须检查：

- visible / infrared / depth / labels 文件数量；
- stem 是否完整对应；
- 是否存在缺失文件；
- 是否存在损坏图像；
- 三模态宽高是否一致；
- Visible dtype / channels；
- Infrared dtype / channels；
- Depth dtype / min / max / mean / median / zero ratio；
- 标签字段数量；
- class_id 合法性；
- bbox 坐标范围；
- `w / h > 0`；
- 空标签文件数量。

检查程序只报告问题，不直接修改官方原始数据。

## 6.1 目标尺寸分档（团队内部分析标准）

本节定义仅用于团队内部的数据统计、划分复核和实验分析，**不是赛事官方标准**。

所有 bbox 固定映射到 `imgsz=640` 的参考 letterbox 尺度。即使后续某次训练使用其他 `imgsz`，统计口径也不改变，以保证不同实验之间可比较。

对原图宽高 `W、H` 和归一化框宽高 `norm_w、norm_h`，计算：

```text
scale = min(640 / W, 640 / H)
bbox_width_ref  = norm_w * W * scale
bbox_height_ref = norm_h * H * scale
bbox_area_ref   = bbox_width_ref * bbox_height_ref
```

按照 `bbox_area_ref`（单位：参考尺度像素平方）分档：

```text
tiny   : area < 256
small  : 256 <= area < 1024
medium : 1024 <= area < 9216
large  : area >= 9216
```

其中 `small / medium / large` 的 32² 与 96² 边界借鉴 COCO 面积尺度，`tiny < 16²` 是本团队增加的分析档位。实现必须使用上述通用 letterbox 缩放公式，不得把当前数据的内容高度 `360` 写死。

## 7. 固定 train / val

完成训练数据检查后建立固定划分：

```text
train = 80%
val   = 20%
seed  = 2026
```

保存：

```text
data/splits/train.txt
data/splits/val.txt
```

要求：

- 仅从 `data/raw/train/` 中划分；
- `data/raw/test/` 不进入划分；
- 不复制、不移动原始图像；
- 所有正式可比较实验使用同一 split；
- 相同数据与 seed 应生成完全一致的划分。

为减少连续帧或同场景样本跨越 train / val 造成的验证泄漏，本项目固定采用 seed=2026 的确定性 group-aware 多标签分层，而不是单张图片级随机划分。该方法属于团队内部数据划分规范，不是赛事官方标准。

分组规则：

- 带下划线且末段为数字帧号的 stem：删除最后帧号后，前缀相同的样本归入同一组；
- 纯数字 stem：仅当相邻编号差不超过 5，且 64×36 灰度相关系数不低于 0.90 或 dHash 汉明距离不大于 5 时连接为同一候选序列组；
- 同一个 group 不得跨越 train / val；
- 分层同时考虑 12 类目标数、12 类图片覆盖数和四种目标尺寸数量；
- 固定使用 5000 次候选搜索和 50000 次同组大小局部交换，seed 固定为 2026；
- 最终必须检查 1600/400 数量、stem 全覆盖、零交集、group 零交叉、类别覆盖和尺寸覆盖，并记录 split 文件 SHA256。

## 7.1 训练标签最小清洗副本

官方原始标签始终保留在 `data/raw/train/labels/`，不得覆盖。训练如需使用已复核的最小修正版，统一生成到：

```text
data/processed/train/labels_clean/
```

当前团队内部处理口径：

- 只修复 B2 已定位并目视复核的 5 个官方字段范围错误；
- 将 YOLO 中心点和宽高还原为四角坐标，裁剪到 `[0, 1]` 后重新计算中心点和宽高；
- 只删除 1 个完全相同的重复框；
- 保留无目标场景的空标签文件；
- 其余字段合法但边缘轻微越界的 warning 暂不批量改写；
- 必须输出逐行改动清单，并复检至 `error_count=0`；
- 该处理是团队内部训练预处理标准，不是赛事官方要求。

生成命令：

```powershell
python scripts/data/prepare_clean_labels.py
```

`data/processed/` 已被 Git 忽略；各成员需使用相同脚本从各自的官方训练集本地生成，不能上传比赛数据。

## 8. 测试数据使用

`data/raw/test/` 仅用于：

- 模型推理；
- 生成预测 TXT；
- 打包比赛提交文件。

不得用于：

- 模型训练；
- 验证集选择；
- 人工标注；
- 数据增强来源；
- 伪标签训练。

## 9. 官方预测格式

每张测试图对应一个同名 TXT。

每个预测目标占一行：

```text
class_id norm_center_x norm_center_y norm_w norm_h confidence
```

要求：

- `class_id`：0~11；
- bbox 坐标合法；
- `confidence ∈ [0,1]`；
- 每张图最多 100 个预测框；
- 无检测结果时仍生成空 TXT。

最终将全部预测 TXT 打包提交。

## 10. 提交前自动检查

建议实现：

```text
tests/test_submission_format.py
```

至少检查：

- 测试图与预测 TXT 一一对应；
- 文件名；
- 每行字段数；
- class_id；
- bbox 范围；
- confidence；
- 每图预测框数量；
- 空 TXT 是否存在。
