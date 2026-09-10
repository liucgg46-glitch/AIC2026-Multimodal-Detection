# AIC2026 Multimodal Modalities Summary

> Measurement results describe the selected local dataset. Engineering suggestions are hypotheses for later experiments and are not claims of mAP improvement.

Official raw labels: `data/raw/train/labels/`

Clean labels used by this analysis: `data/processed/train/labels_clean/`

Old path `data/raw/train/labels/labels/` is not used.

## 1. Data scope and fixed split

- Split: `all`; analyzed stems: 2000.
- Fixed train/val counts: 1600/400; overlap: 0.
- Registration is a deterministic format-balanced sample: 200 stems (seed 42).
- No new train/val partition is created.

## 2. Label paths

- Raw label files: 2000.
- Clean label files: 2000.
- Bbox and target-depth analysis use only `labels_clean`.

## 3. Basic modality statistics

| Modality | Files | Suffixes | Dtypes | Shapes | Channels |
|---|---:|---|---|---|---|
| visible | 2000 | `.jpg`: 149, `.png`: 1851 | `uint8`: 2000 | `1080x1920x3`: 1851, `360x640x3`: 149 | `3`: 2000 |
| infrared | 2000 | `.jpg`: 149, `.png`: 1851 | `uint8`: 2000 | `1080x1920x3`: 1851, `360x640x3`: 149 | `3`: 2000 |
| depth | 2000 | `.jpg`: 149, `.png`: 1851 | `uint16`: 1851, `uint8`: 149 | `1080x1920`: 1851, `360x640x3`: 149 | `1`: 1851, `3`: 149 |

- Common Visible/Infrared/Depth/clean-label stems: 2000.
- Three-modality width agreement: 100.00%.
- Three-modality height agreement: 100.00%.
- Safety conclusion: equal width and height establish dimension agreement only; they do not establish strict pixel-level RGB/Visible, Infrared and Depth registration. Residual offsets, black borders, effective-field-of-view differences and automatic-registration reliability show that size agreement does not imply direct pixel correspondence.

## 4. Infrared dtype, channels and distribution

- Three-channel ratio: 100.00%; single-channel ratio: 0.00%.
- Global pixel range/mean/std: 0.0000 / 255.0000 / 113.7516 / 59.9968.
- B/G mean absolute difference: mean=0.9368, median=0.2591, p90=2.2887, p95=2.9000, min=0.0000, max=5.9921.
- B/R mean absolute difference: mean=1.2018, median=0.3974, p90=2.8662, p95=3.4380, min=0.0000, max=8.1935.
- G/R mean absolute difference: mean=0.7550, median=0.2255, p90=1.9539, p95=2.3996, min=0.0000, max=5.0824.
- Tukey channel-difference outliers: 38 total; up to 20 are stored in JSON.

## 5. Black borders and effective field of view

Black-border measurement uses median continuous low-value runs from each image edge; fully low scan lines are excluded from the perpendicular side estimate. For ordinary uint8 imagery, low means grayscale <= 8. For PNG uint16 Depth, it means zero. This is an engineering measurement of edge bands, not proof that every dark pixel is invalid.

| Modality | Border occurrence | Effective FOV ratio | Low/zero pixel ratio |
|---|---:|---|---|
| visible | 0.65% | mean=99.99%, median=100.00%, p90=100.00%, p95=100.00%, min=97.87%, max=100.00% | mean=0.36%, median=0.06%, p90=0.40%, p95=0.96%, min=0.00%, max=35.77% |
| infrared | 59.30% | mean=92.45%, median=94.24%, p90=100.00%, p95=100.00%, min=52.76%, max=100.00% | mean=8.62%, median=7.10%, p90=18.30%, p95=26.44%, min=0.00%, max=46.27% |
| depth PNG | 100.00% | mean=82.08%, median=96.09%, p90=98.54%, p95=98.80%, min=0.05%, max=99.17% | mean=27.75%, median=25.09%, p90=54.99%, p95=62.69%, min=1.18%, max=99.90% |
| depth JPG | 100.00% | mean=24.10%, median=22.52%, p90=58.26%, p95=84.90%, min=0.00%, max=96.72% | mean=78.87%, median=77.72%, p90=91.80%, p95=100.00%, min=56.28%, max=100.00% |

## 6. PNG Depth statistics

- Files: 1851; representation: single-channel uint16.
- All pixels: min=0.0000, max=19999.0000, mean=5899.2233, median=5287.0000.
- Nonzero pixels: count=2773232562, mean=8164.6946, median=7376.0000.
- Global zero ratio: 27.75%.
- Per-image zero ratio: mean=27.75%, median=25.09%, p90=54.99%, p95=62.69%, min=1.18%, max=99.90%.
- All-pixel `<300` ratio (includes zero): 27.76%.
- All-pixel `300..20000` inclusive ratio: 72.24%.
- All-pixel `>20000` ratio: 0.00%.
- Nonzero `<300` ratio: 0.01%.

## 7. JPG Depth statistics

- Files: 149; observed dtype/channel distributions are shown in the basic table.
- Global pixel range/mean/std: 0.0000 / 255.0000 / 30.8624 / 67.6604.
- Per-image zero ratio: mean=77.94%, median=77.03%, p90=89.98%, p95=99.90%, min=52.93%, max=100.00%.
- Per-image dynamic range: mean=229.7450, median=255.0000, p90=255.0000, p95=255.0000, min=0.0000, max=255.0000.
- Grayscale entropy: mean=2.3222, median=2.4910, p90=3.2437, p95=3.5240, min=-0.0000, max=4.2635 bits.
- B/G difference: mean=0.0885, median=0.0000, p90=0.4463, p95=0.4632, min=0.0000, max=0.9253.
- B/R difference: mean=0.0293, median=0.0000, p90=0.1479, p95=0.1536, min=0.0000, max=0.3076.
- G/R difference: mean=0.0592, median=0.0000, p90=0.2985, p95=0.3096, min=0.0000, max=0.6180.
- JPG Depth physical mapping cannot be confirmed directly from the current data.

## 8. PNG/JPG encoding differences

PNG Depth and JPG Depth are separate representations. PNG files are evaluated as uint16 single-channel depth under the project specification. JPG files are compressed uint8 imagery with unknown physical mapping; their values are not merged with PNG statistics and are never subjected to millimeter thresholds.

## 9. Spatial alignment analysis

Pilot comparison found ORB-RANSAC unstable across modalities (implausible scale/rotation/translation) and global phase correlation reliable only for some JPG Visible/IR samples. The selected method is constrained local edge correlation. It searches only a small translation window and rejects weak, ambiguous, or boundary peaks.

`dx,dy` describe detected modality-content translation relative to Visible in original-image pixels. Reliable flags are mandatory; unreliable estimates are excluded from displacement summaries.
Residual Infrared/Depth offsets may be used only for data-quality analysis, valid-region masks, robust Fusion design, explicit future registration experiments and modality-uncertainty handling. They must never be used to shift or modify Visible GT bboxes, rewrite `labels_clean`, regenerate annotations from IR/Depth offsets, or alter official Visible labels. The official Visible image remains the coordinate basis for every Visible GT bbox.

### Visible ↔ Infrared

- Attempted: 200
- Reliable: 107 (53.50%)
- Reliable displacement: mean=4.5222, median=2.9814, p90=9.8920, p95=20.2773, min=0.0000, max=34.4093 px
- dx: mean=-1.3458, median=-1.3333, p90=0.0000, p95=6.8000, min=-24.0000, max=20.0000 px
- dy: mean=1.0467, median=1.3333, p90=2.6667, p95=4.0000, min=-28.0000, max=20.0000 px
- Unreliable reasons: `ambiguous_peak`: 20, `low_edge_correlation`: 11, `low_edge_correlation,ambiguous_peak`: 24, `ambiguous_peak,weak_peak_ratio`: 8, `weak_peak_ratio`: 5, `ambiguous_peak,weak_peak_ratio,search_boundary`: 1, `low_edge_correlation,ambiguous_peak,search_boundary`: 14, `ambiguous_peak,search_boundary`: 5, `low_edge_correlation,search_boundary`: 3, `low_edge_correlation,ambiguous_peak,weak_peak_ratio,search_boundary`: 1, `low_edge_correlation,ambiguous_peak,weak_peak_ratio`: 1
- `jpg_uint8`: 78/100 reliable (78.00%); displacement mean=2.4855, median=2.9814, p90=2.9814, p95=2.9814, min=1.3333, max=4.2164 px.
- `png_uint16_single`: 29/100 reliable (29.00%); displacement mean=10.0000, median=5.6569, p90=25.3611, p95=28.9896, min=0.0000, max=34.4093 px.

### Visible ↔ Depth

- Attempted: 200
- Reliable: 45 (22.50%)
- Reliable displacement: mean=16.1495, median=16.4924, p90=24.3311, p95=28.0000, min=0.0000, max=29.1204 px
- dx: mean=14.4296, median=16.0000, p90=24.0000, p95=28.0000, min=-8.0000, max=28.0000 px
- dy: mean=1.4222, median=0.0000, p90=4.0000, p95=7.7333, min=-4.0000, max=20.0000 px
- Unreliable reasons: `low_edge_correlation,ambiguous_peak`: 18, `low_edge_correlation`: 5, `low_edge_correlation,ambiguous_peak,search_boundary`: 24, `ambiguous_peak`: 26, `ambiguous_peak,search_boundary`: 17, `low_edge_correlation,search_boundary`: 6, `insufficient_edges`: 8, `low_edge_correlation,ambiguous_peak,weak_peak_ratio,search_boundary`: 7, `low_edge_correlation,ambiguous_peak,weak_peak_ratio`: 6, `search_boundary`: 22, `weak_peak_ratio`: 10, `ambiguous_peak,weak_peak_ratio,search_boundary`: 2, `ambiguous_peak,weak_peak_ratio`: 3, `weak_peak_ratio,search_boundary`: 1
- `jpg_uint8`: 5/100 reliable (5.00%); displacement mean=3.8891, median=1.3333, p90=9.5077, p95=10.4107, min=0.0000, max=11.3137 px.
- `png_uint16_single`: 40/100 reliable (40.00%); displacement mean=17.6821, median=16.4924, p90=24.6979, p95=28.0000, min=0.0000, max=29.1204 px.

## 10. Near/far and bbox-scale alignment

- Targets with usable PNG physical-depth medians: 9587 (63.10% of all labels).
- Physical target-depth distribution: mean=10055.9874, median=8670.0000, p90=17153.9000, p95=18534.4000, min=654.0000, max=19816.0000.
- Bbox area-ratio distribution: mean=0.0140, median=0.0030, p90=0.0317, p95=0.0617, min=0.0001, max=0.9229.
- Near/far thresholds are the 33.3% and 66.7% quantiles of valid PNG target-depth medians. JPG targets are excluded from physical near/far grouping.
- Small/large bbox groups use bbox area-ratio quantiles and represent apparent target scale, not physical distance.
- Limitation: Registration values are reliable image-level translations inherited by targets; they are not local per-bbox registrations.

Physical thresholds: near <= 7789.6028, far >= 12152.6986.

### Physical depth groups

| Group | Targets | Visible↔IR magnitude | Visible↔Depth magnitude |
|---|---:|---|---|
| near | 3196 | n=71, mean=8.7789, median=8.9443, p90=20.0000, p95=20.3961, min=0.0000, max=25.6125 px | n=62, mean=16.0386, median=16.0000, p90=28.0000, p95=28.0000, min=0.0000, max=29.1204 px |
| far | 3196 | n=45, mean=7.1486, median=4.0000, p90=20.0000, p95=24.3178, min=0.0000, max=31.2410 px | n=90, mean=19.3044, median=16.4924, p90=28.0000, p95=28.6162, min=12.0000, max=29.1204 px |

### BBox scale groups

| Group | Targets | Visible↔IR magnitude | Visible↔Depth magnitude |
|---|---:|---|---|
| small_bbox | 5065 | n=149, mean=7.7414, median=2.9814, p90=20.0000, p95=20.0000, min=0.0000, max=25.6125 px | n=116, mean=21.0415, median=24.0000, p90=29.1204, p95=29.1204, min=0.0000, max=29.1204 px |
| large_bbox | 5065 | n=185, mean=5.5412, median=2.9814, p90=17.8885, p95=20.3961, min=0.0000, max=25.6125 px | n=109, mean=15.8520, median=16.4924, p90=28.0000, p95=28.0000, min=0.0000, max=29.1204 px |

## 11. Representative anomalies

- IR largest channel differences: 000003_069_00000175, 000003_015_00000077, hehe_234_000002_044_00000055, 000003_069_00000088, hehe_237_000002_044_00000217, hehe_235_000002_044_00000109, 000019_029_00000205, 000003_015_00000001, shuming_829_00000328, 000011_004_00000086.
- PNG Depth highest zero ratios: 003127, shuming_343_00000288, shuming_342_00000275, 003125, 000157, 001075, 001077, hehe_69_00000070, 000242, shuming_780_00000323.
- Lowest IR effective FOV: 00000529, 00000519, 00000535, 00000526, 00000282, 003056, 00000248, 003127, 00000524, 00000253.
- Lowest PNG Depth effective FOV: 003127, 000157, 001038, shuming_780_00000323, 001075, 000708, 003125, 000707, shuming_779_00000254, shuming_536_00000023.
- Lowest JPG Depth visual FOV: 00000212, 00000214, 00000220, 00000373, 00000394, 00000396, 00000399, 00000401, 00000410, 00000414.

## 12. IR-only preprocessing suggestions

- Treat grayscale conversion or single-channel training as an ablation only if channel-difference statistics confirm strong redundancy; retaining the stored three channels is the compatibility baseline.
- Normalize IR independently from RGB. Consider CLAHE or contrast enhancement only as controlled experiments.
- Preserve or explicitly mask measured edge bands; do not crop them independently from other modalities in fusion training.

## 13. Depth-only preprocessing suggestions

- PNG: preserve uint16 on read, maintain a valid mask, and compare percentile, log-depth, or inverse-depth display/input transforms without modifying source files.
- PNG zero regions should remain distinguishable; an additional mask channel is an experiment candidate.
- JPG: treat as an independent uint8 encoded representation with its own normalization. Do not interpret values as millimeters.
- A training pipeline that silently mixes PNG uint16 and JPG uint8 Depth under one normalization is an engineering risk.

## 14. Fusion preprocessing suggestions

- Resize, crop, flip, affine and perspective parameters must be shared exactly across Visible, IR and Depth.
- Modality-specific photometric normalization may differ, but geometry must stay synchronized.
- Carry IR edge-band masks and PNG invalid-depth masks where useful, and evaluate robustness to residual misalignment.

## 15. Current limitations and unsupported conclusions

- Automatic registration is a translation-only estimate on a deterministic sample, not a dense calibration or proof of pixel-perfect alignment.
- Unreliable matches are not converted into precise offsets.
- Target-group registration inherits image-level shifts and cannot establish target-local parallax.
- JPG Depth physical units and mapping remain unknown.
- These statistics motivate experiments; they do not establish that any preprocessing choice improves mAP.
