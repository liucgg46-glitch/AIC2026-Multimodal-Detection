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
