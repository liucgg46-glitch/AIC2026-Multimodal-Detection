# Round 3 final audit and Round 4 Depth controls

The fixed Round 3 commit was `836287775da47b03e2c25ab43e782ee28f6ec0ed`. IR raw3
YOLO11m completed 200 epochs with training-history best mAP50-95 0.27269 at epoch 174 and
best.pt revalidation 0.27253049. Depth log YOLO11m completed 200 epochs with history best
0.21404 at epoch 153, final history 0.21014 and best.pt revalidation 0.21061410. These
revalidations were reported by the server operator; this local artifact contains curves,
args and checkpoint hashes, but not the checkpoint bytes or final revalidation console logs.

The Depth mAP jumps reflect changing predictions on the fixed validation set. For example,
epoch 72 scored 0.16187 with recall 0.26384 and val/cls_loss 1.98169; epoch 73 scored 0.01155
with recall 0.04128 and val/cls_loss 4.70699; epoch 74 recovered to 0.14279. Epochs 105–107
show the same pattern: 0.00921 → 0.19414 → 0.01254. Training losses decline smoothly,
and the cosine learning rate changes only slightly across each jump. There is no evidence of
missing labels, wrong class count, NaN or a different COCO source checkpoint in the live logs:
both runs loaded 643/649 source items, scanned 1600/400 with zero corrupt, and used the same
source SHA256 `d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95`.

Depth had 28 adjacent jumps larger than 0.05 mAP during epochs 101–150. During epochs
171–200 it had zero such jumps, mean mAP 0.2071 and standard deviation 0.0046, while median
learning rate fell to approximately 2.46e-5. Learning-rate reduction and model maturation
occur together, so the curve alone cannot prove learning rate caused the instability.

The data remain intrinsically hard. In the fixed val set, 903/2961 PNG Visible-label boxes
cover less than 10% nonzero raw Depth pixels. In the 27 JPG images, 60/80 label boxes cover
less than 10% grayscale-above-8 pixels (a signal proxy, not a physical-validity test).
Among PNG boxes with at least 10% valid Depth, the valid-pixel log-view P95−P5 contrast has
median about five 8-bit levels. Visible↔Depth registration was reliable in only 45/200 of an
earlier sampled audit, with a 16-pixel median offset in those reliable pairs; this is not a
universal fixed shift. The frozen `log` view also copies 149 physically uninterpretable JPGs
unchanged and carries no explicit validity/format channels. These are representation and
supervision limits, not proof of a code-level conversion bug.

## Two single-variable experiments

Run both from the same public `yolo11m.pt` and the same canonical log view, fixed 1600/400
split, labels_clean and 200-epoch schedule. Use identical inference/validation settings.

| Workspace | Config | Only training parameter changed from R3 | Hypothesis |
|---|---|---|---|
| GPU-A | `DEPTH_R4_M960_LOG_MOSAIC_OFF.yaml` | mosaic 1.0 → 0.0 | Mosaic creates too many sparse/artificial Depth contexts. |
| GPU-B | `DEPTH_R4_M960_LOG_LR3E4.yaml` | AdamW initial LR 0.001 → 0.0003 | Parameter updates are too large for low-information Depth. |

Judge both **stability and AP**. Record history best, best.pt same-protocol revalidation,
mean mAP for epochs 171–200, and number/max magnitude of adjacent mAP jumps over 0.05 in
epochs 1–50, 51–100 and 101–150. A run that never reaches competitive AP can appear stable
simply because predictions never improve. Also compare recall and val losses at jump epochs.

After pushing the new commit, detach each independent workspace at the exact new 40-character
SHA and confirm `git status --porcelain` is empty. GPU-A may need to generate its canonical
Depth log view; GPU-B already used one. Do not regenerate an existing view during a run. Check
the source checkpoint hash against the value above before training.

GPU-A, `/root/data1/AIC2026/AIC2026-Multimodal-Detection`:

```bash
set -euo pipefail
test -f data/processed/depth_trainable/log/manifest.json || python scripts/data/prepare_depth_candidate_yolo.py --candidate log
test "$(sha256sum weights/yolo11m.pt | cut -d' ' -f1)" = d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R4_M960_LOG_MOSAIC_OFF.yaml
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R4_M960_LOG_MOSAIC_OFF.yaml --smoke
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R4_M960_LOG_MOSAIC_OFF.yaml --train --expected-sha <NEW_40_CHARACTER_SHA> 2>&1 | tee DEPTH_R4_M960_LOG_MOSAIC_OFF.log
```

GPU-B, `/root/data1/AIC2026/AIC2026-Multimodal-Detection-B`:

```bash
set -euo pipefail
test -f data/processed/depth_trainable/log/manifest.json
test "$(sha256sum weights/yolo11m.pt | cut -d' ' -f1)" = d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R4_M960_LOG_LR3E4.yaml
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R4_M960_LOG_LR3E4.yaml --smoke
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R4_M960_LOG_LR3E4.yaml --train --expected-sha <NEW_40_CHARACTER_SHA> 2>&1 | tee DEPTH_R4_M960_LOG_LR3E4.log
```

Keep the formal YAML unchanged if smoke fails, and version any batch-size adjustment separately.
Save results.csv, args.yaml, console log, data manifest, best.pt SHA256 and best.pt revalidation.
Round 4 can establish whether LR or mosaic contributes to the oscillation. Regardless of the
outcome, the next score-oriented fusion model should use explicit Depth validity and JPG-format
information, source the improved IR encoder from R3, and retain an exact M960 RGB fallback.
