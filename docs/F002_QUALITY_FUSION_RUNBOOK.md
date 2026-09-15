# F002: M960 RGB + IR + Depth quality fusion

F002 is one detector, not an average of separately trained predictions. The 24-layer YOLO11m RGB
path is initialized from the verified Round 2 M960 `best.pt`; stage 1 freezes its weights **and RGB
BatchNorm running statistics** and trains
small IR/Depth pyramids and P3/P4/P5 residual adapters. The initialized model returns exactly the
M960 RGB output whenever both modality-validity masks are zero. This protects the known RGB baseline
while measuring whether either additional modality actually adds information.

## Fixed inputs and representation

- Use the original 2000 labeled RGB/IR/Depth triplets, fixed `1600/400` train/val stems and
  `data/processed/train/labels_clean`. The trainer checks full stem sets, source extensions,
  dimensions, class range and clean-label identity before training.
- RGB and IR are raw three-channel `uint8` images. The IR validity mask excludes dark regions
  connected to the image border (grayscale `<=8`), including near-black JPEG border noise.
- Depth PNG is original single-channel `uint16`: encode physical millimeters with a capped
  0–20 m linear channel, inverse-distance channel and valid mask. Depth JPG is original
  three-channel `uint8`: its grayscale and complement are treated as an unknown-scale encoding,
  with a separate format token and first adapter. Its near-black no-data border is masked. JPG
  gray levels are **not** interpreted as
  millimeters. The representation does not invent missing physical scale.
- The 11 transport channels are `[RGB3, IR3, depth_linear, depth_inverse, depth_valid,
  ir_valid, depth_jpg_format]`. Letterbox, translation, scale and horizontal flip act on the
  whole container. HSV applies to RGB only. IR and Depth are dropped independently at 20% each
  during training to constrain over-reliance on poorly aligned modalities.

## Server execution

The frozen initial checkpoint must be at
`/root/data1/AIC2026/AIC2026-Multimodal-Detection/runs/RGB_R2_M960/weights/best.pt`, with SHA-256
`097852eb7357dc8b7db72fef82e0485e5542a67b051977451edd4305be70ac85`.
Check the exact F002 Git commit after fetching the bundle, then use detached HEAD and a clean tree.

```bash
cd /root/data1/AIC2026/AIC2026-Multimodal-Detection
source /root/data1/AIC2026/aic2026_env/bin/activate
git checkout --detach <F002_40_CHARACTER_COMMIT_SHA>
git rev-parse HEAD
git status --porcelain
sha256sum runs/RGB_R2_M960/weights/best.pt
python scripts/train/train_quality_fusion.py
python scripts/train/train_quality_fusion.py --smoke
python scripts/train/train_quality_fusion.py --train --expected-sha <F002_40_CHARACTER_COMMIT_SHA>
```

The first command checks pairing, model graph and actual trainer-side RGB weight transfer without
training. The smoke command uses 32 train and 16 val examples, 320 pixels, batch 2 and one real GPU
epoch; its result is a runtime check, not a score. Formal F002 uses 960 pixels, batch 4, 100 epochs,
patience 30, AdamW and the frozen RGB path. Preserve `args.yaml`, `results.csv`, `best.pt`, complete
console output and the full Git SHA. If the RTX 3090 runs out of memory, do not silently alter the
formal config: record the failure and make a separately versioned batch-size experiment.

## Decision after F002

The comparison baseline is the **same M960 checkpoint** revalidated at `mAP50-95=0.479013` on
the fixed val set. Evaluate F002 `best.pt` with the same split and also zero the IR mask, zero the
Depth mask, and zero both masks. Report macro and per-class AP, plus aligned/unreliable subsets.
An unpaired/shuffled IR check is useful because only 101/400 earlier F001 val pairs were classified
as reliably aligned. Modality ablations should use the same trained checkpoint, so a drop or gain
has an interpretable meaning.

If F002 does not beat M960, use those ablations to choose the next work: fix IR masking/registration
if IR helps only on aligned examples; investigate PNG/JPG Depth branches separately if Depth is
neutral or harmful; or train one controlled 960-pixel single-modality IR/Depth reference to isolate
representation from model size. A broad single-modality hyperparameter sweep has lower priority
before the repechage dataset opens on September 21 at 11:00.

## Prediction ZIP once the repechage test is released

Hash the trained F002 `best.pt` and use its actual RGB/IR/Depth test directories. The default
contract expects 1000 identical stems with matching extensions; change `--expected-count` only if
the official released test count differs. The ZIP contains exactly one `<stem>.txt` per image,
including empty files, with class and normalized `xywh` plus confidence. It is written atomically.

```bash
sha256sum runs/F002_M960_RGB_IR_DEPTH_QUALITY_P345/weights/best.pt
python scripts/inference/predict_quality_fusion.py \
  --model runs/F002_M960_RGB_IR_DEPTH_QUALITY_P345/weights/best.pt \
  --model-sha256 <F002_BEST_PT_SHA256> \
  --rgb-dir data/raw/test/visible --ir-dir data/raw/test/infrared \
  --depth-dir data/raw/test/depth \
  --output-zip outputs/submissions/F002_predictions.zip \
  --imgsz 960 --conf 0.001 --iou 0.7 --device 0
```

The [official schedule](official/repechage_2026/附件1：第八届AIC算法大赛算法挑战赛道复赛赛事日程表.pdf)
opens repechage result submissions September 22 at 09:00 and closes them October 5 at 20:00.
The [official submission attachment](official/repechage_2026/附件2：第八届AIC算法大赛算法挑战赛道复赛作品提交要求.pdf)
sets the separate prediction result and October 7 at 23:59 code/materials link deadline. Treat
these files as competition requirements; the score-2 rules control the specific deliverable contents.
