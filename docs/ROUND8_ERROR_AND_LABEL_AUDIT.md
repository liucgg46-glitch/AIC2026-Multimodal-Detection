# Round 8: reject inference additions, audit the training annotations and localization

## Decisions already made

The fixed 400-image R7 comparison gives standard reconstructed mAP50-95 **0.45327355** and sliced **0.43820762**, delta **-0.01506594**. Small-object recall at conf=0.25/IoU=0.5 moves from 213/379 (56.2%) to 223/379 (58.8%), only +2.64 percentage points. Eleven of twelve class APs fall; the +0.010 AP/recall leaderboard gate is not met. Stop sliced inference and do not launch crop-focused training from this evidence.

R6 TTA also missed its gate (+0.00330 AP vs +0.005 required). A single offline, class-aware IoU-0.5 NMS pass on the R6 standard TXT was tested after the R7 failure: reconstructed mAP50-95 fell to **0.44474411** (delta **-0.00852945**). The existing same-class overlaps cannot simply be deleted; no NMS sweep is scheduled.

At conf=0.25/IoU=0.5, R7 sliced predicts 3954 boxes, matches 2416 GT and leaves 1538 unmatched predictions; standard predicts 3210, matches 2389 and leaves 821 unmatched. The 27 extra matches cost 717 extra unmatched predictions. This explains the AP loss more directly than the small-recall increase alone.

## Read-only R5 error audit

The reproducible command below reads the **R6 standard** predictions, the fixed `labels_clean` val labels and `val.txt`. It does not read or mutate test data, labels, or images. The output is a review queue, not a claim that every heuristic category is an actual model or annotation error.

```powershell
python scripts/analysis/audit_r5_prediction_failures.py `
  --predictions artifacts/outputs/analysis/R6_R5_STANDARD `
  --output outputs/analysis/R8_R5_FAILURE_AUDIT
```

The generated `summary.json`, `gt_misses_at_conf_0p001.csv` and `false_positives_at_conf_0p5.csv` are already present locally. The summary's 2659 TP and 382 FN at the minimum export confidence match an independent greedy-matcher calculation; 2209 TP and 380 FP at conf=0.5 match the same independent calculation.

At conf=0.001/IoU=0.5, the 382 unmatched GT split into 191 with a same-class candidate at IoU 0.1–0.5, 159 with no candidate even at IoU 0.1, 24 with an overlapping other-class candidate, and 8 matching conflicts. At conf=0.5, 380 unmatched predictions include 152 near a same-class GT at IoU 0.1–0.5 and 188 with no same-class overlap at IoU 0.1 (the latter may include unlabelled valid objects). These numbers point first to localization/annotation geometry and missing candidates. An image review is necessary before changing labels or losses.

Two directly inspected examples show that the *raw* labels already contain obvious semantic errors:

- `001139.png`: three visible pedestrians align with class-1 (`boat`) raw/clean GT boxes. The R5 model predicts class-0 (`person`) with confidence 0.938, 0.882 and 0.876, at IoU 0.892, 0.784 and 0.646 against those class-1 GT boxes. A fourth class-1 box in the image also needs review.
- `000575.png`: a peacock is assigned class-1 (`boat`) in both raw and clean labels. The model's class-2 (`animal`) prediction overlaps it at IoU 0.634.

These examples were found by model disagreement and are **not a prevalence estimate**. The `background_or_unlabeled` category does not prove a missing annotation: some visible signs or fixtures may be outside the competition's intended class definition.

## Next work, in order

1. Review all 62 images containing class-1 `boat` GT (135 objects over all 2000 images), plus the 24 val GT with overlapping wrong-class candidates. Start with the two confirmed examples. Compare each proposal to the official category definitions and record image stem, row index, original class/box, proposed class/box and visual rationale. Review a stratified sample of high-confidence `background_or_unlabeled` and `localization_overlap_0p1_to_0p5` rows, especially `person`, `animal`, `sign`, `bicycle` and `light`.
2. Keep `data/raw/train/labels` immutable. Do not silently change `labels_clean` or the fixed 400-image validation GT. Put adjudicated training-only corrections in a separate, versioned patch manifest. Preserve baseline validation against the original GT and report any separately adjudicated diagnostic metric under a different name.
3. If review finds systematic training-label errors (for example, >=10% of `boat` training instances or a repeatable box-geometry pattern), prepare two matched fine-tunes from the same R5 checkpoint: unchanged labels control on GPU-B and training-only corrected labels on GPU-A, with identical seed, schedule and validation protocol. If review finds only isolated mistakes, do not spend two full GPU runs on them; prioritize a controlled localization/recall training change based on the reviewed misses.

The 12-class macro score means rare-class errors can matter, but a jump from roughly 0.48 to the leader's reported 0.66 requires improvements across several classes. Do not infer that fixing the two examples alone can close that gap.
