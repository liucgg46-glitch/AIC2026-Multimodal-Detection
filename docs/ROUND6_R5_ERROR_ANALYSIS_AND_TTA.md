# Round 6: diagnose the R5 plateau before another formal training run

## Evidence and decision

- `RGB_R5_X1280` best.pt reval mAP50-95 is 0.48117; history best is 0.48239 at epoch 132. Round 2 M960 reval was 0.47901. A much larger model and 1280 input bought only 0.00216 on this fixed val, so further detector scaling is low priority.
- `DEIMV2_S_RGB_FORMAL` best mAP50-95 is 0.45900, with COCO AP_small 0.18685, AP_medium 0.38186, AP_large 0.62664. This supports a small-object problem, although it does not prove YOLO11x has the identical per-size profile.
- Existing E001 error analysis (older YOLO11n, not R5) found 496 misses among 757 tiny val GT at conf=0.25/IoU=0.5. This is a hypothesis generator, not a current R5 metric.
- `labels_clean` has 27 tricycle objects in all 2000 images; the 400-image val has only 5. The scorer averages 12 classes. One rare class can move aggregate mAP by up to 1/12, while its five val examples make class-specific tuning noisy. Do not change labels or oversample based only on these five validation examples.
- IR heat audit was REJECT. No IR heat fusion experiment is scheduled in this round.

The first experiment is an **inference-only**, paired standard-versus-Ultralytics-TTA comparison of the same R5 best.pt on the same 400 val images. It tests whether the existing model has recoverable recall before spending a GPU-day on crop training. The AIC PDF scorer reconstructed in this repository is not the official scorer; use paired deltas and inspect per-class changes, then confirm a meaningful gain on the leaderboard.

## GPU-A, fixed checkout and paired prediction exports

After the local commit is pushed by the user, set `R6_SHA` to its exact 40-character value. Use the same 24 GB GPU-A workspace that contains `runs/RGB_R5_X1280/weights/best.pt`. These commands do not retrain or alter the fixed 1600/400 split.

```bash
cd /root/data1/AIC2026/AIC2026-Multimodal-Detection
git fetch origin
git checkout --detach "$R6_SHA"
test "$(git rev-parse HEAD)" = "$R6_SHA"
test -z "$(git status --porcelain)"
source /root/data1/AIC2026/aic2026_env/bin/activate

sha256sum runs/RGB_R5_X1280/weights/best.pt
# Must equal 8864bf704a0a2c84bcb7fcc9ea9a9b72fef1ac5f362863b02300730865c257b2.
python -m pytest tests/test_predict_rgb.py tests/test_compare_r5_predictions.py -q

python scripts/inference/predict_rgb.py \
  --model runs/RGB_R5_X1280/weights/best.pt \
  --source data/processed/rgb_yolo_clean/images/val \
  --output outputs/analysis/R6_R5_STANDARD \
  --device 0 --imgsz 1280 --conf 0.001 --iou 0.7 --max-det 100

python scripts/inference/predict_rgb.py \
  --model runs/RGB_R5_X1280/weights/best.pt \
  --source data/processed/rgb_yolo_clean/images/val \
  --output outputs/analysis/R6_R5_TTA \
  --device 0 --imgsz 1280 --conf 0.001 --iou 0.7 --max-det 100 --augment

python scripts/analysis/compare_r5_predictions.py \
  --baseline outputs/analysis/R6_R5_STANDARD \
  --candidate outputs/analysis/R6_R5_TTA \
  --output outputs/analysis/R6_R5_STANDARD_VS_TTA.json
```

Each inference directory must contain exactly 400 TXT files, including empty files. Keep both directories and the JSON report; send the compact JSON plus both prediction directories for review. The report includes 12 per-class AP deltas and original-image COCO size-bin recall at conf=0.25/IoU=0.5. The size-bin recall is diagnostic and is **not** COCO AP_small/AP_medium/AP_large.

## Decision rule

1. If TTA improves reconstructed macro mAP50-95 by at least 0.005 with no obvious failure mode in the per-class table, make one leaderboard submission with the same 1280/0.001/0.7/100 settings. Retain standard inference if the leaderboard does not improve.
2. If TTA is flat or worse, do not spend time sweeping many confidence/NMS combinations against only 400 val images. Use the exported predictions to inspect current R5 class AP and size recall. The next training experiment should then target the confirmed loss mode: controlled high-resolution crops for small-object recall if small recall is poor; rare-class image sampling only if the relevant class AP and false-negative evidence justify it; label audit if errors cluster around ambiguous class definitions or boxes.
3. Do not interpret a single low-support class delta as a reliable hidden-test gain. `tricycle` has only 5 val GT. The overall 12-class delta and leaderboard confirmation matter more than an isolated jump on those examples.

GPU-B remains available for a second experiment after this scorecard identifies the bottleneck. Do not spend its time on another unmeasured detector family or rejected IR-heat fusion.
