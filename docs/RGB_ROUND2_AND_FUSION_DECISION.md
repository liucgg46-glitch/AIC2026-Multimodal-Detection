# RGB Round 2 and fusion decision

## Evidence frozen on 2026-09-14

All RGB scaling runs used commit `41f2f8fb3e9ed0d75bcb4e2e4180cd3aeb98360a`, the fixed
1600/400 split and `labels_clean`.

| run | local mAP50-95 | leaderboard |
| --- | ---: | ---: |
| AUDIT_N640 | 0.38033 | not submitted |
| AUDIT_S640 | 0.40617 | not submitted |
| AUDIT_S960 | 0.45883 | 47.484 |

The S960 leaderboard result confirms that the local ranking is useful. Local PDF-rule reconstruction
gave 0.43484 at `imgsz=960, conf=0.001, iou=0.7, max_det=100`. Inference-only 1280 gave 0.43433 and
IoU 0.5 gave 0.43158, so those inference changes are rejected.

All three scaling runs peaked before `close_mosaic` disabled mosaic for the last 20 epochs. S960
peaked at epoch 178 and ended at 0.44817. Round 2 keeps mosaic enabled through epoch 200.

## F001 decision

The fixed-val ablations are:

| mode | mAP50-95 | delta from NORMAL |
| --- | ---: | ---: |
| NORMAL | 0.32480 | 0 |
| RESIDUAL_OFF | 0.31364 | -0.01116 |
| IR_ZERO | 0.30557 | -0.01923 |
| IR_SHUFFLED | 0.30857 | -0.01624 |

IR contains useful paired information, but F001 does not gate it by quality. P4 has mean gate 0.6907
and residual/RGB norm 0.2925. The mean P4 residual ratio is 0.2910 for reliably aligned samples and
0.2933 for unreliable samples. Thus the gate injects nearly the same amount in both groups.

Only 101/400 pairs were classified as reliably aligned. IR borders occur in 55.5% of validation images.
F001 improves `light` by 0.112 AP and `car` by 0.024 AP relative to residual-off, while several classes
decline. F001 is retired as a score candidate. Fusion V2 must start from the strongest RGB model, fuse
P3/P4/P5, carry an IR valid-region mask, use modality dropout, and preserve an exact RGB fallback.

## Round 2 experiments

Run in this order:

1. `RGB_R2_S960_CONTROL`: same S960 architecture at batch 8 with mosaic kept through training.
2. `RGB_R2_M960`: capacity test against the control.
3. `RGB_R2_S960_P2`: stride-4 detection head for tiny objects.

The P2 model initializes from the frozen AUDIT_S960 best checkpoint, SHA-256
`7fa00453124ff0ebfe2a3e7b4bce0316a751985acbb43d4f1a984b24c68b8486`. A semantic mapper copies the
unchanged backbone/top-down path, maps the old P3/P4/P5 downsampling path into its shifted layer
positions, and maps the three regression branches to P3/P4/P5. It covers more than 95% of target
parameters. The new P2 path and incompatible four-scale classification towers retain seeded
initialization.

Every experiment remains check-only by default. On the server, run check-only and smoke before a
formal detached-HEAD invocation:

```bash
python scripts/train/train_rgb.py --config configs/experiments/RGB_R2_S960_CONTROL.yaml
python scripts/train/train_rgb.py --config configs/experiments/RGB_R2_S960_CONTROL.yaml --smoke
python scripts/train/train_rgb.py --config configs/experiments/RGB_R2_S960_CONTROL.yaml --train --expected-sha <SHA>
```

Repeat with `RGB_R2_M960.yaml` and `RGB_R2_S960_P2.yaml`. M960 requires the public `yolo11m.pt` at
`weights/yolo11m.pt`. P2 requires the existing `runs/AUDIT_S960/weights/best.pt` with the exact hash
above. Do not run P2 if its initialization report is absent or reports less than 95% parameter coverage.
