# S003 Depth-only submission tools

E003 historical training commit remains
`1e1040af0dc6160d415c1b02cc9406f60a42ccd4`. The commit introducing these tools is
a separate preprocessing/inference commit, never a replacement training commit.
Resolve and record that full SHA, sync it and verify server HEAD before execution.
This local implementation task does not execute server staging, inference or a
real submission ZIP, and does not submit to the leaderboard.

## Artifact gate

Use `runs/E003_DEPTH_YOLO11N_CLEAN/weights/best.pt`, SHA256:
`b3ba23370e1370ff7831c3ac923c0b26e6da63bf4ff7438f7fe502d9fc711a4d`.
Check file existence and this hash before inference; the generic inference entry
does not enforce the artifact hash. Never use a model name that could download weights.

## Staging (future server execution)

```bash
python scripts/data/prepare_depth_inference_yolo.py \
  --candidate inverse \
  --source data/raw/prelim_test/depth \
  --output data/processed/depth_inference/inverse/prelim_test
```

The local layout may use `data/raw/test/depth`; the current official server layout
uses `data/raw/prelim_test/depth`. These are the only two allowed source aliases and
use identical conversion semantics. Always explicitly pass the canonical test-depth
directory that actually exists. The manifest records that actual project-relative
source path, including each file record.

Only these canonical source aliases, the fixed output path and inverse candidate are accepted. No force
option is provided. Existing output fails before image processing. Failed builds
may leave a private `.prelim_test-*` temporary sibling for inspection; no recursive
deletion occurs. Nothing reads labels or split, or fits statistics.

PNG uses C4 `write_candidate_png(..., "inverse", None)` directly; metadata uses
`candidate_conversion_metadata("inverse", None)`. No inverse formula is reimplemented.
The frozen PNG rules are valid=depth>0, near=300, far=19999, invalid=0,
valid=[1,255], uint8 x3. JPG uses C4 isolated byte-preserving copy with physical unit
unknown. Test JPEG values are never interpreted as mm. Original stems are retained
and output extensions lowercased. Source inventory is hashed before and after.

Output contains `images/` and deterministic `manifest.json`. The manifest records
purpose, representation, relative paths, counts, unique casefold stems, source and
output SHA summaries, per-file SHA records, C4 conversion metadata, JPG policy,
dtype/channels and source_unchanged. No labels or machine absolute paths are recorded.
The CLI enforces 1000 images but derives PNG/JPG counts from source. Local inspection
previously found 845 PNG and 155 JPG; verify the server source independently.

## Inference (after artifact and staging gates)

```bash
python scripts/inference/predict_rgb.py \
  --model runs/E003_DEPTH_YOLO11N_CLEAN/weights/best.pt \
  --source data/processed/depth_inference/inverse/prelim_test/images \
  --output outputs/submissions/E003_DEPTH_YOLO11N_CLEAN_PRELIM_001 \
  --imgsz 640 --conf 0.001 --iou 0.7 --max-det 100 --device 0
```

The existing generic directory inference entry remains unchanged. Use the same
S001/S002 policy; no test-based threshold tuning or training.

## Validation, statistics and optional ZIP (not executed here)

```bash
python tests/test_submission_format.py \
  --images data/raw/prelim_test/depth \
  --predictions outputs/submissions/E003_DEPTH_YOLO11N_CLEAN_PRELIM_001 \
  --max-det 100
python scripts/inference/prepare_s003_submission.py --images data/raw/prelim_test/depth
```

The second command reuses the existing format validator, enforces 1000 inputs/TXTs,
exact stems and TXT-only directory contents, and prints missing/extra, empty/nonempty,
total/mean/max boxes, hit_max_det_100 and validity results. Invalid class IDs,
coordinates, confidence, NaN/Inf or more than 100 boxes cause failure.

To create the final ZIP only when separately authorized:

```bash
python scripts/inference/prepare_s003_submission.py \
  --images data/raw/prelim_test/depth \
  --zip-output outputs/submissions/E003_DEPTH_YOLO11N_CLEAN_PRELIM_001.zip
```

ZIP creation is exclusive (no overwrite), with sorted TXT-only root members and
fixed timestamps. It verifies member names/count, CRC and byte contents, then prints
ZIP SHA256. No manifest, images or subdirectories enter the archive. Statistics
without `--zip-output` do not create a ZIP. Unit tests use synthetic temporary ZIPs
only, not S003 artifacts.
