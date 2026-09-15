# F002 protocol audit and Round 3 modality controls

The completed `F002_quality_protocol_audit_9257cbe.json` was produced from the fixed 400-image
validation split. The RGB fallback with the original preprocessing measured 0.45138; matching
Ultralytics' long-side pre-resize and rectangular validation measured 0.47733. The original M960
revalidation was 0.479013. The 0.02595 AP recovery identifies preprocessing as the dominant
cause of the apparent RGB initialization collapse; a 0.00168 AP gap remains. Do not compare
scores produced by different resize/LetterBox protocols.

On the corrected rectangular protocol, F002 full RGB+IR+Depth measured 0.47240, RGB-only fallback
0.47733, IR-only auxiliary contribution (Depth masked) 0.48005, and Depth-only auxiliary
contribution (IR masked) 0.47210. The IR gain is only 0.00272 AP versus RGB fallback; the Depth
branch loses 0.00523 AP. No submission claim follows from these validation numbers. In particular,
the current final detector needs a useful Depth branch for the three-modality repechage task.

The F002 inference script now defaults to single-image Ultralytics-style `auto_rect` LetterBox and
can reproduce the old square transport with `--letterbox-mode square`. The audit's rectangular
validation batches of 16 are not the same as single-image prediction. Before submitting F002,
generate two 400-image validation predictions using the same checkpoint, NMS settings and labels,
then score them with `scripts/analysis/evaluate_submission.py`. Keep both prediction outputs and
the predictor mode in the artifact. The reconstructed evaluator is not the official leaderboard.

## Round 3: two independent modality controls

Run `IR_R3_M960_RAW3_CLEAN` on GPU-A and `DEPTH_R3_M960_LOG_CLEAN` on GPU-B. Both use YOLO11m,
960 pixels, the fixed 1600/400 split, COCO initialization and labels_clean. HSV augmentation is
disabled because physical IR and Depth intensities are not RGB hue/saturation. Existing clean-view
contracts now check the exact view path, manifest, split and staged labels before training.

IR raw3 is a byte-preserving three-channel transport. Depth log transforms uint16 PNG into an
8-bit log view, while the 149 JPG images (122 train, 27 val) remain byte-preserved and have
unknown physical scale. The Depth run is a controlled *representation diagnostic*, not evidence
that one unified physical depth representation has been achieved. Preserve separate PNG/JPG AP
and per-class errors for the subsequent Depth redesign.

On both servers, first fetch the user-pushed bundle/commit into the independent workspace, detach
at the **new 40-character SHA supplied with this runbook**, and verify `git status --porcelain` is
empty. Use the existing shared Python environment. Keep the two output names from the YAML files;
the workspaces must remain independent.

GPU-A, `/root/data1/AIC2026/AIC2026-Multimodal-Detection`:

```bash
set -euo pipefail
test -f weights/yolo11m.pt
test -f data/processed/ir_trainable/raw3/manifest.json || python scripts/data/prepare_ir_yolo.py --representation raw3 --link-mode copy
python scripts/train/train_rgb.py --config configs/experiments/IR_R3_M960_RAW3_CLEAN.yaml
python scripts/train/train_rgb.py --config configs/experiments/IR_R3_M960_RAW3_CLEAN.yaml --smoke
python scripts/train/train_rgb.py --config configs/experiments/IR_R3_M960_RAW3_CLEAN.yaml --train --expected-sha <NEW_40_CHARACTER_SHA> 2>&1 | tee IR_R3_M960_RAW3_CLEAN.log
```

GPU-B, `/root/data1/AIC2026/AIC2026-Multimodal-Detection-B`:

```bash
set -euo pipefail
test -f weights/yolo11m.pt
test -f data/processed/depth_trainable/log/manifest.json || python scripts/data/prepare_depth_candidate_yolo.py --candidate log
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R3_M960_LOG_CLEAN.yaml
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R3_M960_LOG_CLEAN.yaml --smoke
python scripts/train/train_rgb.py --config configs/experiments/DEPTH_R3_M960_LOG_CLEAN.yaml --train --expected-sha <NEW_40_CHARACTER_SHA> 2>&1 | tee DEPTH_R3_M960_LOG_CLEAN.log
```

If the preflight finds a wrong or stale generated view, regenerate only the exact canonical view
using its generator's `--force` after checking the target directory. If a 3090 cannot fit batch 8,
record the smoke OOM and create a separate committed batch-size experiment; do not mutate the
formal YAML on a detached server checkout. Save `results.csv`, `args.yaml`, `best.pt`, training
console output and data `manifest.json` from each run.

Revalidate both best checkpoints with the same 960-pixel validation protocol as RGB M960. If IR
improves sharply over the old n640 result, use it to initialize or distill the IR fusion encoder.
If Depth remains weak, compare performance on the PNG and JPG subsets before designing the next
format-aware Depth encoder. A stronger RGB baseline can be trained after these modality limits
are measured; fusion should be judged against its *own* same-protocol RGB fallback.
