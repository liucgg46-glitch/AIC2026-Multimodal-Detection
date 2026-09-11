# AIC2026 多模态数据分析报告

本报告汇总成员 C 对 Visible、Infrared 与 Depth 数据的 C1～C4 检查结果。数值以 `outputs/analysis/modalities_stats.json` 的正式全量统计为准。实测结果、工程解释与后续实验假设在文中分别说明；任何预处理或融合方案均需通过固定划分上的受控实验验证，本文不对 mAP 提升作预判。

## 1. 数据范围与约束

**实测结果**

- Train 数据共 2000 组；Visible、Infrared、Depth 与 `labels_clean` 的 filename stem 一一对应，共同 stem 和并集均为 2000，未发现缺失或重复 stem。
- 官方原始标签位于 `data/raw/train/labels/`，共 2000 个 TXT；旧路径 `data/raw/train/labels/labels/` 当前不使用。
- Bbox、near/far 和目标级分析统一使用 `data/processed/train/labels_clean/`。官方原始标签未被修改。
- 固定划分为 `data/splits/train.txt` 1600 组、`data/splits/val.txt` 400 组，二者无重叠，不重新随机划分。
- `PHASE_1_1000` 不参与训练、验证、调参或人工标注。

**工程约束**

- `data/raw/` 始终只读；分析派生文件写入 `outputs/analysis/` 或 `outputs/visualization/`。
- 所有后续 E001～E006 实验必须沿用同一固定 train/val split，保证比较口径一致。
- Visible GT bbox 的坐标基准始终是官方 Visible 图像，不得依据 IR 或 Depth 的残余偏移修改标注。

## 2. 三模态基础对应关系

| 模态 | 文件数 | 扩展名 | dtype | shape | 通道数 |
|---|---:|---|---|---|---:|
| Visible | 2000 | PNG 1851，JPG 149 | uint8 | 1080×1920×3：1851；360×640×3：149 | 3 |
| Infrared | 2000 | PNG 1851，JPG 149 | uint8 | 1080×1920×3：1851；360×640×3：149 | 3 |
| Depth | 2000 | PNG 1851，JPG 149 | PNG uint16；JPG uint8 | PNG 1080×1920；JPG 360×640×3 | PNG 1；JPG 3 |

三模态 width 和 height 一致率均为 100%。该结果仅说明同一 stem 的数组宽高一致，不能证明 RGB/Visible、Infrared 与 Depth 已达到逐像素严格配准。黑边、有效视场差异、残余位移以及有限的自动配准可靠率均表明：**尺寸一致不等于像素级严格对齐**。

## 3. Infrared 数据特性

全部 2000 张 Infrared 均为 uint8 三通道图像，全局数值范围为 0～255，全局均值为 113.7516，标准差为 59.9968。

| 通道对 | 平均绝对差 mean | median | P95 | max |
|---|---:|---:|---:|---:|
| B/G | 0.9368 | 0.2591 | 2.9000 | 5.9921 |
| B/R | 1.2018 | 0.3974 | 3.4380 | 8.1935 |
| G/R | 0.7550 | 0.2255 | 2.3996 | 5.0824 |

**结论：** Infrared 三通道高度相似，但并非逐像素完全相同。正式统计记录了 38 个 Tukey 通道差异异常样本；这些样本是后续人工复核对象，不能据此直接判定为错误数据或自动删除。

## 4. Infrared 黑边与有效视场

- 黑边出现率：59.30%。
- 有效视场比例：均值 92.45%，中位数 94.24%。
- 右侧连续低值边带宽度：中位数 42 px。

黑边测量基于图像边缘连续灰度低值区域，普通 uint8 图像的低值阈值为灰度不高于 8；该指标用于描述边缘带，不代表所有暗像素均无效。较高的黑边出现率和有效视场变化会影响直接像素级融合。原图不应被独立裁切或覆盖；后续可将边缘 mask 作为受控实验输入，并保证涉及几何变换时三模态参数完全同步。

## 5. PNG Depth 数据特性

1851 张 PNG Depth 均以 `cv2.IMREAD_UNCHANGED` 读取，实测为 shape `1080×1920`、单通道 uint16，数值范围 0～19999。全像素均值为 5899.2233，中位数为 5287。

| 指标 | 结果 |
|---|---:|
| 全局 zero ratio | 27.7472% |
| 每图 zero ratio median | 25.0948% |
| 每图 zero ratio P90 | 54.9888% |
| 每图 zero ratio P95 | 62.6916% |
| 每图 zero ratio max | 99.8979% |
| 全像素 `<300`，包含零值 | 27.7555% |
| 非零有效像素 `<300` | 0.0116% |
| 全像素 `300～20000`，含边界 | 72.2445% |
| 全像素 `>20000` | 0.0000% |

Depth 中的零值是**无效深度候选**，不能解释为真实 0 mm，也不能把包含大量零值的 `<300` 全像素比例解释为真实近距离比例。物理范围分析必须将 zero ratio 与非零有效像素中的 `<300` 比例分开报告。

## 6. JPG Depth 数据特性

149 张 JPG Depth 均为 shape `360×640×3`、uint8 三通道表示，全局范围 0～255，全局均值 30.8624，标准差 67.6604。

| 指标 | 结果 |
|---|---:|
| 每图 zero ratio mean | 77.9446% |
| 每图 zero ratio median | 77.0330% |
| 动态范围 median | 255 |
| 灰度熵 mean | 2.3222 bits |
| 灰度熵 median | 2.4910 bits |

JPG Depth 的物理深度映射无法仅从当前数据确认。JPG 的 0～255 不得解释为毫米，不适用 `<300 mm`、`300～20000 mm` 或 `>20000 mm` 阈值。PNG uint16 与 JPG uint8 是两种显著不同的数据表示，不能直接共享同一套物理深度解释或归一化流程。

## 7. C4 Depth 专项可视化

C4 已对代表性 PNG uint16 Depth 生成 2×3 分析图，输出位于 `outputs/visualization/depth_analysis/`。布局包括 Visible 参考图、Raw/Normalized Depth、Valid Mask、Percentile-Clipped Depth、Log Depth 与 Inverse Depth。

| 可视化 | 处理方式与用途 |
|---|---|
| Valid Mask | 使用 `depth > 0` 标记有效区域，直接检查无效区域分布和 zero ratio。 |
| Percentile-Clipped | 仅用非零像素计算 P2～P98，裁剪后归一化；用于改善显示对比度，并作为候选归一化方案。 |
| Log Depth | 仅对有效像素计算 `log1p(depth)` 并独立归一化；用于压缩远距离动态范围。 |
| Inverse Depth | 仅对有效像素计算 `1/depth`，避免除零后独立归一化；用于突出近距离结构变化。 |

四类派生结果均保持无效零值区域为黑色，仅用于分析或候选预处理，不改变原始 Depth，也不是已经验证的最优训练方案。代表性输出覆盖指定高 zero ratio 样本及按 zero ratio 分位确定性选取的 val 样本。

## 8. 三模态空间配准分析

配准统计采用 seed 42 的 200 个确定性、格式平衡样本。方法为受约束的局部边缘相关，仅估计有限搜索窗口内的图像级平移；低置信、峰值模糊或边界结果不进入可靠位移汇总。

| 配准对 | 可靠结果 | 可靠率 | 位移 mean | median | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| Visible ↔ IR | 107/200 | 53.50% | 4.5222 px | 2.9814 px | 9.8920 px | 20.2773 px |
| Visible ↔ Depth | 45/200 | 22.50% | 16.1495 px | 16.4924 px | 24.3311 px | 28.0000 px |

自动配准可靠率有限，这些结果只能作为残余偏移线索，不能外推为全部样本都存在某个固定平移，也不能证明逐像素严格对齐。可靠率本身还受到跨模态外观差异、黑边和有效视场的影响。任何估计偏移均不得用于移动或修改 Visible GT bbox、重写 `labels_clean` 或重新生成官方标注。

## 9. Near/Far 目标分析

Bbox 和目标级统计使用 `labels_clean`。clean-label 目标共 15194 个，其中 9587 个获得可用的 PNG 框内物理深度中位数，覆盖率为 63.10%。Near/Far 阈值来自有效 PNG 目标深度的 33.3% 与 66.7% 分位数：Near 不高于 7789.60 mm，Far 不低于 12152.70 mm，两组各 3196 个目标；JPG 样本不参与物理 Near/Far 分组。

| 分组 | 目标数 | IR 位移 median / P90 | Depth 位移 median / P90 |
|---|---:|---:|---:|
| Near | 3196 | 8.94 / 20.00 px | 16.00 / 28.00 px |
| Far | 3196 | 4.00 / 20.00 px | 16.49 / 28.00 px |

这些位移是目标继承的**可靠图像级平移估计**，不是 bbox 内的局部配准或目标视差测量。当前结果不足以断言距离越近或越远必然导致更大的配准误差。

## 10. 代表性异常样本

- PNG near-all-zero：`003127`，zero ratio 约 99.90%。
- 其他高 zero ratio 代表样本：`shuming_343_00000288`、`shuming_342_00000275`、`003125`。
- 最小动态范围 PNG：`000011_004_00000086`，动态范围为 2255。
- 全黑 PNG：未发现。
- 非 uint16、非单通道或 shape 异常 PNG：未发现。
- 最大值低于 300 或高于 20000 的 PNG：未发现。
- Infrared 通道差异 Tukey 异常样本：38 个。

这些记录用于后续人工复核和鲁棒性实验，不构成自动删除、填充、转换或修复数据的依据。

## 11. IR-only 后续实验假设

E002 建议在固定 split 上分别验证：原始三通道 IR、灰度单通道 IR、灰度复制三通道、IR 独立归一化，以及 CLAHE 受控消融。三通道高度相似使单通道方案具有验证价值，但仍需与保留三通道的兼容基线直接比较。

**实验假设：** 在低照度、Visible 对比度不足、阴影或强光干扰等场景中，IR 可能提供额外的轮廓和目标响应信息。该判断仅用于提出分场景实验，不代表灰度化、CLAHE 或 IR-only 一定提升 mAP。

## 12. Depth-only 后续实验假设

E003 对 PNG 建议比较 percentile normalization、log-depth、inverse-depth 与 Depth + valid mask。输入形式可分别验证单通道、复制三通道、附加 valid-mask 通道、归一化到 `[0,1]` 以及固定无效值处理。

JPG Depth 应作为独立的 uint8 representation，采用独立 normalization，不按毫米解释。如果训练代码把 PNG uint16 和 JPG uint8 直接纳入同一套物理尺度归一化，会造成明确的数据表示不一致风险。

**实验假设：** 在前后景外观相似、尺度变化、遮挡或视觉纹理不足的场景中，有效 Depth 可能提供几何和距离线索；大面积无效区域则可能削弱这一作用，因此必须同时验证 valid mask。上述方案均需通过 E003 实验评估，不能预先认定会提升 mAP。

## 13. Fusion 实验前安全约束

后续 E004～E006 的 resize、crop、flip、affine 和 perspective 等几何增强必须在 Visible、IR 和 Depth 上共享完全相同的参数。光度增强与数值归一化可以按模态独立设计，但不得破坏三模态的几何对应关系。

融合实验应显式考虑 IR black-border mask、Depth valid mask、残余错位和模态专用归一化。IR/Depth 相对 Visible 的残余偏移只可用于数据质量分析、有效区域 mask、鲁棒融合设计、显式配准实验和模态不确定性处理；不得用于移动或修改 Visible bbox、重写 `labels_clean`、依据 IR/Depth 偏移重新生成标注或擅自修正官方 Visible 标注。

## 14. E002～E006 建议实验顺序

```text
E001 RGB baseline
→ E002 IR-only
→ E003 Depth-only
→ E004 RGB + IR
→ E005 RGB + Depth
→ E006 RGB + IR + Depth
```

该顺序先建立单模态基线，再评估双模态与三模态增益，有利于区分额外信息来自哪一模态。成员 C 的报告只提供可验证假设和数据边界，不实现 Fusion。

## 15. 当前不能下结论的事项

当前数据分析不能证明：

- JPG Depth 的物理深度映射；
- 三模态已达到逐像素严格配准；
- 低置信或不可靠样本具有精确可信的 dx/dy；
- 图像级平移等价于目标级局部视差；
- Near 或 Far 必然对应更大的配准误差；
- CLAHE、percentile、log、inverse 或 valid mask 一定提升 mAP；
- Fusion 一定优于 RGB baseline。

## 16. 成员 C 最终结论

1. Infrared 三通道高度相似但并非完全相同，存在 38 个值得人工复核的通道差异 Tukey 异常样本。
2. PNG Depth 为单通道 uint16，范围 0～19999；JPG Depth 为三通道 uint8，范围 0～255，二者的数据表示和物理解释不可混用。
3. PNG Depth 全局 zero ratio 为 27.7472%，零值应作为无效深度候选单独处理；JPG Depth 的物理映射仍无法确认。
4. 三模态尺寸和 stem 完全对应，但现有证据不支持逐像素严格配准；IR 黑边、Depth 无效区及残余偏移需要在后续管线中显式处理。
5. IR 在低照度或 Visible 对比度不足场景、Depth 在可能受益于几何与距离线索的场景中，具有提供额外信息的实验价值；这仍是待验证假设。
6. 后续正式实验必须沿用固定 split，通过 E002～E006 逐项验证模态专用预处理和融合方案，不得根据 IR/Depth 偏移修改 Visible GT。
