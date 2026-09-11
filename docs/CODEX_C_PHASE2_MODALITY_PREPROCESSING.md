# CODEX_C_PHASE2_MODALITY_PREPROCESSING

负责人：成员 C

未来分支：`feature/modality-preprocessing`

## 1. 阶段定位

成员 C 从第一阶段 multimodal inspection 进入 trainable modality preprocessing，但不负责正式 E002/E003 full training。

已知 Infrared：

- uint8；
- 三通道高度一致；
- 本质为热辐射灰度信息的三通道存储。

已知 Depth：

- 1851 个 PNG：uint16、single-channel；
- 149 个 JPG：uint8、3-channel。

绝对禁止把 JPG 的 0～255 当成与 PNG 相同的毫米物理深度。

第一阶段 registration 抽样统计：

- Visible↔IR：107 / 200 reliable；
- Visible↔Depth：45 / 200 reliable。

三模态尺寸一致不代表 pixel-perfect alignment。Visible GT 始终作为监督坐标基准，不得因为 IR/Depth residual offset 移动、修改或重写标签。

## 2. C1：IR-only Trainable Interface

为 E002 提供 deterministic 输入接口，要求：

- fixed train/val；
- `labels_clean`；
- raw read-only；
- mixed file extension safe；
- image/label stem 一一对应；
- 默认预处理简单。

如果提供 grayscale、normalization、CLAHE，必须分别作为独立、可配置、可单独实验的开关，不能默认全部叠加。

## 3. C2：Depth-only Trainable Interface

Depth 接口必须 format-aware，并明确区分：

- PNG uint16；
- JPG uint8 3-channel。

两者不能静默合并为同一种物理数值或使用相同的数值归一化假设。

## 4. C3：Depth Preprocessing Candidates

将第一阶段假设实现为独立候选：

- valid mask；
- percentile clipping；
- log depth；
- inverse depth。

每一项都必须：

- 可独立开启或关闭；
- deterministic；
- 不覆盖 raw；
- 可测试。

不得全部默认开启。PNG 的 invalid region 和 valid mask 必须保留；log depth 必须避免 `log(0)`，inverse depth 必须避免除零，并保证输出无 NaN/Inf。

## 5. C4：Model-compatible Representation

如果需要转换为 YOLO 或 pretrained backbone 可接受的 3-channel uint8 或 3-channel float，必须明确记录：

- conversion formula；
- invalid-value policy；
- normalization/range；
- channel mapping；
- deterministic behavior。

## 6. C5：Geometry Safety

未来多模态几何增强必须共享完全相同的参数，包括：

- resize；
- crop；
- flip；
- affine parameters。

当前阶段不实现复杂 Fusion augmentation。IR/Depth residual offset 只能用于掩码、鲁棒融合、误差分析或后续独立配准实验，不能用于移动 Visible GT。

## 7. C6：Smoke Tests

至少检查：

- stem mapping；
- train/val counts；
- shape；
- dtype；
- finite values；
- range；
- label mapping；
- PNG/JPG branch；
- deterministic output。

## 8. 最终输出

- IR default candidate；
- Depth default candidate；
- PNG/JPG policy；
- optional preprocessing modes；
- smoke-test result；
- E002/E003 recommendation。

## 9. 禁止事项

- full E002 training；
- full E003 training；
- RGB+IR Fusion；
- RGB+Depth Fusion；
- tri-modal Fusion；
- 修改 `labels_clean`；
- 修改 fixed split；
- 修改 raw。
