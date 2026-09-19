# F002 result and protocol audit (2026-09-15)

The synchronized `results.csv`, `args.yaml`, `MANIFEST.txt`, `best.pt`, `last.pt` and `formal.log`
for F002 identify formal commit
`03ea0521e7f477c68f4a00541dadf5f37a1679ea`, the expected M960 source SHA-256 and a
46-epoch run with best history score at epoch 16. The `best.pt` hash matches its manifest.

| observation | value |
| --- | ---: |
| M960 best.pt original revalidation mAP50-95 | 0.479013 |
| F002 first-epoch mAP50-95 | 0.44874 |
| F002 best history mAP50-95, epoch 16 | 0.45223 |
| F002 final history mAP50-95, epoch 46 | 0.44661 |
| F002 best minus original M960 reval | -0.02678 |

F002 was weak from its first epoch. Train box loss went from `0.84055` to `0.83332`, and val box
loss from `1.34007` to `1.33765`; neither curve shows a strong learning trend. This is not the
late severe overfitting pattern seen in F001. The result **does not yet identify** whether the
frozen adapters failed or whether the two reported mAP values used different validation geometry.

The formal log confirms the expected 649 M960 RGB tensor transfer, `freeze=24`, AdamW and a passed
AMP check. Direct comparison of local `best.pt` state dictionaries finds **all 649 RGB tensors
byte-identical** to the original M960 `best.pt`, including every RGB BatchNorm running statistic.
All six fusion projections and gates have nonzero learned weights. With both modality masks zero,
the saved F002 detector's raw prediction is exactly equal to M960 on the same synthetic tensor.
Thus the score gap is not caused by a discarded initialization or RGB weight drift; it must come
from input/validation geometry, actual fusion effects, or both.

In Ultralytics 8.3.253, `DetectionTrainer.build_dataset()` sets `rect=mode == "val"` for
**training-time** RGB validation, which groups images by aspect ratio. A standalone
`DetectionValidator.build_dataset()` does not force rectangle; with `rect=False` it uses a square.
The original M960 `best.pt` revalidation command was not preserved in the local artifacts, so its
exact geometry cannot be asserted from the `0.479013` number alone. F002's custom validation
dataset in the frozen commit used a full `960×960` square for every val image. Its prediction
script also used square letterboxing. A zero-residual RGB fallback can be mathematically identical
for the **same tensor**, while obtaining a different AP when its letterboxed tensor differs from
the original M960 validation path. The original `0.479013` must therefore not be used as proof
of modality damage until a same-protocol control is measured.

The original M960 formal log validates val400 in 25 batches, so its **training-time rectangular**
validation batch is 16. The audit fixes this value explicitly; using F002's validation batch 8
would create different aspect-ratio groups and would not reproduce that geometry exactly. The
corrected square control is compared to the standalone `best.pt` reval `0.479013`; the corrected
rectangle control can be compared to the training-history peak `0.47923`.

The first server audit measured `M960_INIT_rectangle=0.45138` despite batch 16, then crashed when
F002 `best.pt` reached validation loss: the stripped checkpoint stores `model.args` as a `dict`,
while the loss expects attribute access such as `.box`. This was an **audit script crash**, not a
training failure. The patched audit restores the saved args with Ultralytics `get_cfg()` in memory.
It does not change weights or predictions. A local loaded-checkpoint forward/loss probe passes.

The remaining preprocessing mismatch is more consequential than the batch shape. Native
`BaseDataset.load_image()` first resizes the image so its long side is `imgsz` using
`cv2.INTER_LINEAR` and `ceil()` on the short side. Its validation `LetterBox(scaleup=False)` then
pads that already resized image. F002's frozen custom dataset sent the original raw image directly
to `LetterBox(scaleup=False)`, so smaller source images stayed small within the 960 canvas.
The patched read-only dataset offers an explicit `rgb_protocol_resize` control and preserves the
prior resize gain in `ratio_pad`. Synthetic JPEG integration tests confirm that its RGB pixels and
`ratio_pad` are exactly equal to a native `YOLODataset` validation sample in both square and
rectangular modes. This is a proven code difference; its exact mAP effect on val400 is still to be
measured.

The new audit measures four M960 initialized RGB-fallback controls on val400:
`legacy_square`, `legacy_rectangle`, `rgb_square`, and `rgb_rectangle`. It then measures F002 best
on its original legacy square protocol, the corrected RGB square protocol and the corrected RGB
rectangle protocol, including IR-off, Depth-off and both-off on the corrected rectangle.

Run on the server after fetching the new audit commit. The best checkpoint must still match
SHA-256 `ede0c7507d9d0cdf4e13323105a6d0d7ee06988a492ea210f7f2fbcd12aeeb62`.
This reads existing weights and labels; it makes **no optimizer step**.

```bash
cd /root/data1/AIC2026/AIC2026-Multimodal-Detection
source /root/data1/AIC2026/aic2026_env/bin/activate
git checkout --detach <F002_AUDIT_40_CHARACTER_COMMIT_SHA>
git status --porcelain
sha256sum runs/F002_M960_RGB_IR_DEPTH_QUALITY_P345/weights/best.pt
python scripts/analysis/audit_f002_quality.py
```

The script writes the JSON after each completed mode with `completed=false`, then marks it true at
the end; a later crash therefore preserves the valid partial results. Use a new `--output` name
when rerunning rather than overwriting an earlier audit.

Sync `outputs/analysis/F002_quality_protocol_audit.json` after the audit. The legacy rectangle
should approximately reproduce the already observed `0.45138`. If `M960_INIT_rgb_square`
approaches `0.479013` and/or `M960_INIT_rgb_rectangle` approaches `0.47923`, the main validation
gap is explained by missing prescaling. Compare
`F002_BEST_rgb_rectangle_NORMAL` against `F002_BEST_rgb_rectangle_BOTH_OFF` to isolate actual
fusion gain; the IR-off and Depth-off controls identify which branch helps. If corrected M960 still
misses the corresponding original benchmark, inspect source image byte/color parity and
clean-label geometry next. If the
corrected protocol agrees but F002 fusion is neutral or harmful, use the ablations to choose a
controlled IR/Depth representation or trainable-head experiment. Do not launch another long fusion
run until this check resolves the gap.
