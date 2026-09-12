# E001 CLEAN 固定验证集错误分析

## 1. 范围与口径

- 对象：`E001_RGB_YOLO11N_CLEAN`；
- 数据：固定 `val.txt` 的 400 张图，未使用 prelim_test；
- GT：3041 个；低阈值预测：38885 个；
- CLEAN 校验：导出 GT 与本地 `labels_clean` 的 400 张、3041 个框逐项一致；
- FP/FN 工作点：`confidence >= 0.25`、`IoU >= 0.5`；
- size bin 使用 `imgsz=640` letterbox 参考面积：tiny < 256，small [256,1024)，medium [1024,9216)，large >= 9216；
- 未修改 labels_clean、fixed split 或任何数据文件。

## 2. B1：Per-class Metrics

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

## 3. B2：Confusion、FP 与 FN

在内部工作点共有 1001 个 FN，confidence >= 0.5 的 FP 共 223 个。
主要非背景类别混淆：seat -> person (6)；bicycle -> person (5)；boat -> person (4)；bicycle -> car (3)；person -> animal (3)。
完整证据见 `confusion_pairs.csv`、`high_confidence_fp.csv` 和 `representative_fn.csv`。

## 4. B3：BBox-size Performance

以下是内部工作点统计，不等同于赛事官方 AP：

| size | GT | TP | FN | FP(pred-size) | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 757 | 261 | 496 | 290 | 0.4737 | 0.3448 |
| small | 1066 | 730 | 336 | 359 | 0.6703 | 0.6848 |
| medium | 980 | 840 | 140 | 215 | 0.7962 | 0.8571 |
| large | 238 | 209 | 29 | 38 | 0.8462 | 0.8782 |

## 5. B4：Hard Cases

已按 tiny、遮挡代理、低照度、低对比度、密集场景、相似类别和边界目标各筛选最多 3 个候选，并生成 GT/预测叠加图。遮挡代理基于 GT 框重叠，必须由人复核。清单见 `hard_cases.csv`，图片见 `hard_cases/`。

## 6. B5：Data-performance Relation

类别 GT 数量与 AP50-95 的 Spearman 相关系数为 0.413；log10(GT) 与 AP50-95 的 Pearson 相关系数为 0.496。仅报告探索性相关，不解释为因果。

## 7. B6：Multimodal Hypotheses

1. 主要瓶颈是整体 Recall 明显低于 Precision、极低频类别不稳定、tiny/small 目标漏检、背景方向漏检多，以及 AP50 到 AP50-95 的定位性能下降。
2. 低照度和低对比度候选中的漏检可能与 RGB 信息不足有关，应结合叠加图人工确认。
3. E002 IR 优先验证低照度、低对比度下的 person、animal 和其他高 FN 类别；若白天正常场景也同样漏检，则不能归因于 RGB 光照不足。
4. E003 Depth 优先验证拥挤、遮挡和前后景重叠样本；边界截断和 tiny 目标不应预设可由 Depth 自动解决。
5. 重点观察低 AP/低 Recall 类别，以及 `hard_cases.csv` 中同时出现 FN 和高置信度 FP 的样本。
6. Fusion 首先做针对低照度与遮挡子集的可证伪比较；若单模态 IR/Depth 未改善对应子集，不应直接增加 Fusion 复杂度。

## 8. 限制

- size 性能是固定工作点内部统计，不是官方榜单指标；
- hard-case 场景标签由图像统计和框几何筛选，仍需人工复核；
- 12 类相关分析样本量小，只能形成假设；
- 本报告没有使用 prelim_test，也没有启动新训练。
