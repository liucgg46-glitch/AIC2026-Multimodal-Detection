# Round 7: fixed sliced-inference diagnostic

## Why this experiment

The paired R6 exports contain 400 TXT files each. With the local PDF-reconstructed scorer, standard YOLO11x-1280 gives 0.45327355 and Ultralytics TTA gives 0.45657499, a delta of +0.00330144. This is below the predeclared +0.005 leaderboard gate. At confidence 0.25 and IoU 0.5, TTA matches 100 more GT objects but adds 456 more unmatched predictions; it is a poor precision/recall trade for the current model. The standard model recalls only 213/379 (56.2%) small objects versus 1158/1478 (78.3%) medium and 1018/1184 (86.0%) large objects, using original-image COCO area bins. These are operating-point diagnostics, not AP_small values.

The official SAHI work motivates testing local high-resolution crops before fine-tuning: https://arxiv.org/abs/2202.06934 . This repository's experiment is simpler and fixed in advance. It does not use or claim SAHI's postprocessor or benchmark gains.

## Frozen inference policy

- Keep the existing full-image R6 standard predictions as the baseline. Do not use R6 TTA predictions.
- For images wider than 1152 and taller than 648 pixels, predict four 1152x648 overlapping views (a 1920x1080 image gets a 384x216 overlap); skip 640x360 images.
- Run the same R5 best.pt at `imgsz=1280`, tile `conf=0.01`, tile NMS IoU `0.7`, at most 100 boxes per tile. Drop a tile box whose center falls within 16 px of an **internal** crop edge; actual image borders are exempt.
- Project tile boxes into full-image normalized coordinates, multiply tile confidence by 0.8, append only boxes without same-class IoU >0.55 against the baseline or already accepted tiles, then keep the global top 100. Baseline boxes are never removed by local merging except for the required global top-100 cap.
- No parameter sweep against the 400-image val. This is one diagnostic candidate.

## GPU-A execution

After the user pushes the new code commit, set `R7_SHA` to its full 40-character SHA. The R6 standard export must remain at `outputs/analysis/R6_R5_STANDARD` in GPU-A's AIC workspace.

```bash
cd /root/data1/AIC2026/AIC2026-Multimodal-Detection
git fetch origin
git checkout --detach "$R7_SHA"
test "$(git rev-parse HEAD)" = "$R7_SHA"
test -z "$(git status --porcelain)"
source /root/data1/AIC2026/aic2026_env/bin/activate
echo '8864bf704a0a2c84bcb7fcc9ea9a9b72fef1ac5f362863b02300730865c257b2  runs/RGB_R5_X1280/weights/best.pt' | sha256sum -c -
python -m pytest tests/test_predict_rgb_sliced.py tests/test_compare_r5_predictions.py -q

python scripts/inference/predict_rgb_sliced.py \
  --model runs/RGB_R5_X1280/weights/best.pt \
  --images data/processed/rgb_yolo_clean/images/val \
  --baseline outputs/analysis/R6_R5_STANDARD \
  --output outputs/analysis/R7_R5_SLICED \
  --device 0

python scripts/analysis/compare_r5_predictions.py \
  --baseline outputs/analysis/R6_R5_STANDARD \
  --candidate outputs/analysis/R7_R5_SLICED \
  --output outputs/analysis/R7_R5_STANDARD_VS_SLICED.json
```

Archive the prediction directory and comparison JSON locally. Verify 400 prediction files, finite normalized boxes and no image exceeding 100 boxes. Do not submit to the leaderboard before reviewing the comparison.

## Gate after the run

- **Proceed to one leaderboard check** if local reconstructed mAP50-95 rises by at least +0.010 and small-object recall rises by at least five percentage points, without a per-class collapse that explains the gain as five validation tricycles.
- **Reject crop-focused training** if gain is below +0.005 or small-object recall is flat. Move to class/box annotation review, confidence ranking, and localization analysis using the now-available R6/R7 predictions.
- Between +0.005 and +0.010, inspect per-class and hard-case evidence before committing GPU time. Do not tune the tile parameters on this same 400-image val until it passes.

The 400-image reconstructed score may differ from Ultralytics `val()` because its inference and matching protocols differ. Compare standard and sliced **within this paired protocol** only.
