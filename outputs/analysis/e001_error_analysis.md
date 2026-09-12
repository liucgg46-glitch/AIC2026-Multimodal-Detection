# E001 CLEAN 固定验证集错误分析

## 1. 范围与口径

- 对象：`E001_RGB_YOLO11N_CLEAN`；
- 数据：固定 `val.txt` 的 400 张图，未使用 prelim_test；
- GT：3041 个；低阈值预测：38885 个；
- CLEAN 校验：导出 GT 与本地 `labels_clean` 的 400 张、3041 个框逐项一致；
- provenance 详见 `e001_error_analysis/provenance.json`；无法由现有证据恢复的字段统一记为 `unavailable`；
- size bin 使用 `imgsz=640` letterbox 参考面积：tiny < 256，small [256,1024)，medium [1024,9216)，large >= 9216；
- 未修改 labels_clean、fixed split 或任何数据文件。

## 2. B1：两套指标与来源

两套指标必须分开理解，本次 analysis export / re-validation 不覆盖 canonical historical training `results.csv`。

| 指标来源 | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| E001 historical training `results.csv`（best epoch=78） | 0.81605 | 0.55820 | 0.61119 | 0.37346 |
| B analysis export / re-validation | 0.81725 | 0.55934 | 0.61144 | 0.37329 |

Historical result 直接取 export 中 `results.csv` 的最高 mAP50-95 行；checkpoint 记录为 `runs/E001_RGB_YOLO11N_CLEAN/weights/best.pt`，但 export 未包含 checkpoint/hash。re-validation 的指标来自 `validation_metrics.json`，逐类指标来自 `per_class_metrics.csv`。其 validation/export 命令、imgsz、conf、iou、max_det、device 及 Python/torch/Ultralytics 版本均无法由现有 export 确认，记为 `unavailable`。`args.yaml` 是历史训练配置，不能当作独立 re-validation 命令证据。

### Re-validation per-class metrics

| class | GT | Precision | Recall | AP50 | AP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| person | 1111 | 0.7564 | 0.5563 | 0.6405 | 0.3619 |
| boat | 26 | 0.8428 | 0.4615 | 0.5277 | 0.3162 |
| animal | 604 | 0.8133 | 0.6109 | 0.7064 | 0.3996 |
| seat | 128 | 0.9245 | 0.6641 | 0.7246 | 0.5477 |
| sign | 151 | 0.7432 | 0.4967 | 0.5559 | 0.3669 |
| bicycle | 134 | 0.6073 | 0.4616 | 0.5145 | 0.2602 |
| car | 390 | 0.7931 | 0.8179 | 0.8406 | 0.4465 |
| ball | 19 | 0.9916 | 0.5263 | 0.5367 | 0.3601 |
| light | 374 | 0.8841 | 0.7834 | 0.8253 | 0.4534 |
| garbage can | 57 | 0.8824 | 0.3952 | 0.4573 | 0.3212 |
| uav | 42 | 0.9136 | 0.7381 | 0.8070 | 0.4868 |
| tricycle | 5 | 0.6545 | 0.2000 | 0.2007 | 0.1590 |

AP50-95 最低的类别为：tricycle (0.159, GT=5)、bicycle (0.260, GT=134)、boat (0.316, GT=26)、garbage can (0.321, GT=57)、ball (0.360, GT=19)。

## 3. B2：Confusion 与内部 FP/FN 的不同口径

### Imported confusion

- 来源：analysis export 中的 `confusion_matrix.csv`；
- threshold：`unavailable`；matching 设置：`unavailable`；
- diagonal TP = 2086；background-row misses = 920。

### Member B internal FP/FN matcher

- 实现：当前 `scripts/analysis/analyze_e001_errors.py`；prediction confidence >= 0.25；matching IoU >= 0.5；
- TP = 2040；FN = 1001；
- high-confidence FP 定义为 confidence >= 0.5，count = 223。

**Imported confusion 与内部 matcher 不是同一个 evaluator / operating point 的可直接比较结果，不应把两组 TP/FN 数字写成等式或要求一致。**

主要非背景类别混淆：seat -> person (6)；bicycle -> person (5)；boat -> person (4)；bicycle -> car (3)；person -> animal (3)。
完整证据见 `confusion_pairs.csv`、`high_confidence_fp.csv` 和 `representative_fn.csv`。

## 4. B3：BBox-size Performance

尺寸定义固定为 `imgsz=640` 的 letterbox 参考像素面积：tiny < 256 px²、small [256,1024) px²、medium [1024,9216) px²、large >= 9216 px²。以下是自定义内部工作点统计，不是标准 COCO size AP/Precision：TP/FN 按 GT size 归档，FP 按 prediction size 归档，因此表中 Precision 只用于内部诊断。

| size | GT | TP | FN | FP(pred-size) | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 757 | 261 | 496 | 290 | 0.4737 | 0.3448 |
| small | 1066 | 730 | 336 | 359 | 0.6703 | 0.6848 |
| medium | 980 | 840 | 140 | 215 | 0.7962 | 0.8571 |
| large | 238 | 209 | 29 | 38 | 0.8462 | 0.8782 |

## 5. B4：Hard Cases

`hard_cases.csv` 是定向错误案例清单，不代表固定 val 的总体分布。每类最多选择 3 个候选；派生 JPG 仅在本地按需生成且不纳入 Git。occlusion 只是 GT overlap proxy，不报告或声称正式 occlusion Recall。low-light 仅定义为 Visible 灰度均值 P10 候选子集（40 张），目前没有正式 low-light Recall。

## 6. B5：Data-performance Relation

类别在固定 val 中的 GT support 与 AP50-95 的 Spearman 相关系数为 0.413；log10(val GT support) 与 AP50-95 的 Pearson 相关系数为 0.496。这里的 frequency 仅指 val GT support，不是 training frequency；仅报告探索性相关，不解释为因果。

## 7. B6：Multimodal Hypotheses

1. 主要瓶颈是整体 Recall 明显低于 Precision、极低频类别不稳定、tiny/small 目标漏检、背景方向漏检多，以及 AP50 到 AP50-95 的定位性能下降。
2. 假设：Visible-defined 低照度和低对比度候选中的漏检可能与 RGB 信息不足有关，尚未证明。
3. E002 IR 假设：优先验证低照度、低对比度下的 person、animal 和其他高 FN 类别；若白天正常场景也同样漏检，则不能归因于 RGB 光照不足。
4. E003 Depth 假设：优先验证拥挤、GT-overlap proxy 和前后景重叠样本；边界截断和 tiny 目标不应预设可由 Depth 自动解决。
5. 重点观察低 AP/低 Recall 类别，以及 `hard_cases.csv` 中同时出现 FN 和高置信度 FP 的样本。
6. Fusion 假设：首先做针对 Visible-defined low-light 与 GT-overlap proxy 子集的可证伪比较；若单模态 IR/Depth 未改善对应子集，不应直接增加 Fusion 复杂度。

## 8. 限制

- size Precision/Recall 是自定义固定工作点内部统计，不是标准 COCO size 指标或官方榜单指标；
- hard cases 是定向错误候选，不代表总体分布；场景标签由图像统计和框几何筛选，仍需人工复核；
- 12 类相关分析样本量小，只能形成假设；
- 本报告没有使用 prelim_test，也没有启动新训练。
