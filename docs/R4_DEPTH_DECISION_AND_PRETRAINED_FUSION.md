# R4 Depth decision and next experiments

## Evidence

All three runs used the same fixed 1600/400 split, `labels_clean`, log Depth view,
and output identity.

| run | changed variable | best mAP50-95 | late stability |
|---|---|---:|---|
| R3 baseline | none | 0.21404 | epochs 171-200 SD 0.00458 |
| R4 mosaic off | `mosaic=0` | 0.12288 | SD 0.01184 |
| R4 low LR | `lr0=0.0003` | **0.22420** | SD **0.00130** |

The early jumps were optimization instability rather than corrupt conversion:
lowering the learning rate removed the large late jumps and improved the best
metric. Mosaic is essential on this small dataset; disabling it loses 0.0912
mAP50-95. Further single-modality tuning has lower expected value than testing
whether the learned modality features help the stronger RGB detector.

## Round 5 decision

Run these in parallel from one reviewed commit:

1. `F003_M960_RGB_IR_PRETRAINED_P45`: M960 RGB plus the R3 IR encoder.
2. `F004_M960_RGB_IR_DEPTH_PRETRAINED_P45`: the same graph plus the R4 low-LR
   Depth encoder.

Both freeze the three pretrained paths and train only zero-start P4/P5 residual
fusion. A 3x3 auxiliary projection can learn a one-cell correction at P4, which
matches the observed median 16-pixel offset at input scale. P3 is excluded because
that offset is two cells at P3 and the alignment audit found few reliable pairs.
The RGB path therefore starts as an exact M960 fallback.

The Depth input is generated in memory with the exact frozen R4 contract: PNG is
the fixed `log1p` mapping repeated over three channels, while JPG remains its
original three-channel image. Validation uses YOLO-style pre-resize and rectangular
batches of 16 so the RGB fallback is comparable with the original M960 protocol.

## Decision thresholds

- First verify epoch-zero/initial validation is approximately the M960 baseline.
  Stop if it is below 0.47; that indicates a protocol or transfer failure.
- Continue F003/F004 only while the fused metric is recovering toward 0.479.
- Treat F004 minus F003 as the causal Depth contribution. Keep Depth only if the
  best-checkpoint revalidation improves by at least 0.003 mAP50-95 and modality-off
  validation confirms that the gain depends on Depth.
- If neither run exceeds the RGB baseline, stop feature fusion and move to
  prediction-level calibration/ensemble experiments rather than adding capacity.

Run `train_pretrained_fusion.py --audit-init --config <config>` before the smoke
and formal run. It performs a read-only val400 pass and writes the exact metrics
to `outputs/analysis/<experiment>_init_audit.json` without taking an optimizer step.
