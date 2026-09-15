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

In Ultralytics 8.3.253, `DetectionTrainer.build_dataset()` sets `rect=mode == "val"`, and the
ordinary RGB YOLO validation dataset groups images by aspect ratio. F002's custom validation
dataset in the frozen commit used a full `960×960` square for every val image. Its prediction
script also used square letterboxing. A zero-residual RGB fallback can be mathematically identical
for the **same tensor**, while obtaining a different AP when its letterboxed tensor differs from
the original M960 validation path. The original `0.479013` must therefore not be used as proof
of modality damage until a same-protocol control is measured.

The read-only audit script now reproduces rectangular batching, fixes the rectangular
`ratio_pad` gain, and measures on the fixed val400:

1. M960 initialized inside the fusion model with both modality masks zero, square and rectangle;
2. F002 `best.pt` normal and both-off, square and rectangle;
3. F002 `best.pt` IR-off and Depth-off under the rectangle protocol.

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

Sync `outputs/analysis/F002_quality_protocol_audit.json` after the audit. If both-off initialized
rectangle matches `0.479013` but initialized square is about
`0.45`, the large apparent gap is primarily an evaluation-protocol difference. Compare F002 best
rectangle normal against initialized rectangle and its rectangle ablations before a leaderboard
submit. If the initialized rectangle is also far below `0.479013`, investigate image byte/color
parity and clean-label geometry before any more training. If the protocols agree but F002 best is
neutral or worse, the fixed RGB head and low-amplitude masked residuals are the next hypotheses;
then choose a controlled IR/Depth representation or trainable-head experiment based on which
ablation actually helps. Do not launch another long fusion run until this check resolves the gap.
