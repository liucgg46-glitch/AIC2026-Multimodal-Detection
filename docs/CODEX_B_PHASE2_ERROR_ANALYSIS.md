# CODEX_B_PHASE2_ERROR_ANALYSIS

负责人：成员 B

未来分支：`feature/e001-error-analysis`

## 1. 阶段定位

成员 B 从第一阶段 dataset analysis 转入 canonical RGB baseline error analysis。分析对象为 `E001_RGB_YOLO11N_CLEAN`，只使用固定 val = 400，禁止使用 prelim_test。

## 2. B1：Per-class Metrics

整理全部 12 类的：

- Precision；
- Recall；
- AP50；
- AP50-95；
- GT support / object count。

## 3. B2：Confusion Analysis

基于固定 val 分析：

- confusion matrix；
- 主要类别混淆；
- high-confidence FP；
- representative FN。

## 4. B3：BBox-size Performance

沿用项目已经定义的 size bin：

- tiny；
- small；
- medium；
- large。

不得为了结果重新定义 size bin。

## 5. B4：Hard Cases

整理少量有代表性的固定 val 样本：

- tiny target；
- occlusion；
- low light；
- low contrast；
- crowded scene；
- similar classes；
- boundary object。

## 6. B5：Data-performance Relation

可分析：

- class frequency vs AP；
- bbox size vs performance。

只能报告 correlation 或 hypothesis，不能仅凭统计直接声称因果。

## 7. B6：Multimodal Hypotheses

最终必须回答：

1. E001 CLEAN 最大的 3～5 个瓶颈；
2. 哪些问题可能来自 RGB 信息不足；
3. 哪些问题值得 E002 IR 验证；
4. 哪些问题值得 E003 Depth 验证；
5. 哪些类别或场景需要重点观察；
6. 后续 Fusion 值得优先验证什么。

建议输出位置：

- `outputs/analysis/e001_error_analysis/`
- `outputs/analysis/e001_error_analysis.md`

## 8. 禁止事项

- 修改 `labels_clean`；
- 修改 fixed split；
- 分析 prelim_test；
- 启动正式新训练；
- 实现 Fusion；
- 人工修改 submission。
