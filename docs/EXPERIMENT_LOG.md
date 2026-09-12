# EXPERIMENT_LOG

## 1. 记录规范

- 所有正式实验使用 `E001`、`E002`、`E003`……编号。
- Submission 可使用 `S001`、`S002`……辅助编号。
- 正式可比较实验必须使用固定的 train/val split。
- 当前固定 split 为：train = 1600，val = 400，seed = 2026。
- 后续正式实验统一使用 `data/processed/train/labels_clean`。
- `prelim_test` 只允许用于 inference 和 submission，不得用于训练、验证集选择、调参、人工标注或伪标签训练。
- Leaderboard 不能替代固定 val；模型选择与实验结论以固定 val 为主要依据。
- 一次实验尽量只修改一个主要变量。
- 不知道的数据必须标记为 `unavailable` 或 `pending`，禁止猜测。

每次正式实验必须记录：

- Experiment ID
- Date
- Branch
- Git commit
- Model
- Modalities
- Dataset / label version
- Split
- Training config
- GPU
- Best epoch
- Precision
- Recall
- mAP50
- mAP50-95
- per-class AP（获得后）
- Checkpoint
- Submission
- Leaderboard score
- Notes
- Conclusion

## 2. E001_RAW_RGB_YOLO11N

| Field | Value |
| --- | --- |
| Experiment ID | `E001_RAW_RGB_YOLO11N` |
| Status | Completed / historical raw-label baseline |
| Date | unavailable |
| Branch | unavailable; experiment executed from a non-Git server snapshot |
| Git commit | Exact training commit unavailable because the server experiment was executed from a non-Git code snapshot. |
| Model | YOLO11n pretrained |
| Modalities | Visible / RGB only |
| Dataset / label version | official train = 2000; `data/raw/train/labels` |
| Split | train = 1600, val = 400, seed = 2026 |
| GPU | NVIDIA GeForce RTX 3090 24GB |
| Runtime | approximately 1063 s |
| Best epoch | 88 |
| Per-class AP | unavailable |
| Checkpoint | `runs/E001_RGB_YOLO11N/weights/best.pt` |
| Submission | unavailable |
| Leaderboard score | unavailable |

Training config：

| Parameter | Value |
| --- | --- |
| epochs | 100 |
| imgsz | 640 |
| batch | 32 |
| device | 0 |
| workers | 8 |
| optimizer | auto |
| learning rate | unavailable |
| scheduler | unavailable |
| seed | 2026 |
| deterministic | true |
| pretrained | true |
| amp | true |
| cache | false |

`results.csv` best-epoch metrics：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.74593 | 0.55689 | 0.60855 | 0.37340 |

Independent re-validation of `best.pt`：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| approximately 0.744727 | approximately 0.557472 | approximately 0.606733 | approximately 0.373421 |

Run directory：`runs/E001_RGB_YOLO11N`

Notes：这是早期正式训练时误用 raw labels 得到的 baseline。该结果保留为历史对照，但不是后续 canonical baseline。Phase 1 repository finalization commit after PR #6 为 `560d9a92c78af0b392bf8678dd9772d5945ba3e6`；该 commit 不是本实验训练时的 commit。

Conclusion：这是历史 raw-label 对照；后续正式实验不得继续使用 raw labels。

## 3. E001_RGB_YOLO11N_CLEAN

| Field | Value |
| --- | --- |
| Experiment ID | `E001_RGB_YOLO11N_CLEAN` |
| Status | Completed / canonical RGB baseline |
| Date | unavailable |
| Branch | unavailable; experiment executed from a non-Git server snapshot |
| Git commit | Exact training commit unavailable because the server experiment was executed from a non-Git code snapshot. |
| Model | YOLO11n pretrained |
| Modalities | Visible / RGB only |
| Dataset / label version | official train = 2000; `data/processed/train/labels_clean` |
| Split | train = 1600, val = 400, seed = 2026 |
| GPU | NVIDIA GeForce RTX 3090 24GB |
| Runtime | 1064.0 s |
| Best epoch | 78 |
| Per-class AP | unavailable |
| Checkpoint | `runs/E001_RGB_YOLO11N_CLEAN/weights/best.pt` |
| Submission | `S001` |
| Leaderboard score | pending |

Clean-label summary：

| Check | Result |
| --- | ---: |
| source label files | 2000 |
| output label files | 2000 |
| objects | 15194 |
| clipped invalid boxes | 5 |
| removed exact duplicates | 1 |
| reclassified boxes | 12 |
| source files unchanged | true |
| empty label preserved | true |
| unreviewed rows changed | 0 |
| format errors | 0 |

RGB YOLO clean view：

| Check | Result |
| --- | ---: |
| train images | 1600 |
| val images | 400 |
| train labels | 1600 |
| val labels | 400 |
| YOLO clean labels vs `labels_clean` SHA mismatch | 0 |

Training config：

| Parameter | Value |
| --- | --- |
| epochs | 100 |
| imgsz | 640 |
| batch | 32 |
| device | 0 |
| workers | 8 |
| optimizer | auto |
| learning rate | unavailable |
| scheduler | unavailable |
| seed | 2026 |
| deterministic | true |
| pretrained | true |
| amp | true |
| cache | false |
| plots | true |
| save | true |
| val | true |

`results.csv` best-epoch metrics：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.81605 | 0.55820 | 0.61119 | 0.37346 |

Independent re-validation of `best.pt`：

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.8159998894 | 0.5594892323 | 0.6115705829 | 0.3729277459 |

Run directory：`runs/E001_RGB_YOLO11N_CLEAN`

RAW → CLEAN comparison（基于两次训练各自 `results.csv` 的 best epoch）：

| Metric | Difference |
| --- | ---: |
| Precision | +0.07012 |
| Recall | +0.00131 |
| mAP50 | +0.00264 |
| mAP50-95 | +0.00006 |

Notes：该实验修复了已经确认的标签异常。Exact training commit unavailable because the server experiment was executed from a non-Git code snapshot.

Conclusion：clean labels 对本次 mAP50-95 没有显著提升，但 Precision 明显提高。从第二阶段开始，`E001_RGB_YOLO11N_CLEAN` 是 canonical RGB baseline，所有 `E002+` 正式实验必须使用 clean labels。

## 4. Submission S001

| Field | Value |
| --- | --- |
| Submission ID | `S001` |
| Internal name | `E001_RGB_YOLO11N_CLEAN_PRELIM_001` |
| Platform work name | `E001_RGB_CLEAN` |
| Source experiment | `E001_RGB_YOLO11N_CLEAN` |
| Checkpoint | `runs/E001_RGB_YOLO11N_CLEAN/weights/best.pt` |
| Test | 1000 official preliminary-test Visible images |
| Inference runtime | 77.4 s |
| Submission directory | `outputs/submissions/E001_RGB_YOLO11N_CLEAN_PRELIM_001` |
| ZIP | `outputs/submissions/E001_RGB_YOLO11N_CLEAN_PRELIM_001.zip` |
| ZIP size | approximately 1.4 MB |
| ZIP entries | 1000 TXT files at ZIP root |
| ZIP integrity | No errors detected |
| SHA-256 | `8f18a6e87cb6eee83f90b9bd4ef1125b04e0f4e13ace51d2df7938d9c1913eee` |
| Platform status | DONE |
| Platform scoring time | 2026-09-11 18:55:00 |
| Leaderboard score | pending |
| Rank | pending |

Inference config：

| Parameter | Value |
| --- | ---: |
| imgsz | 640 |
| conf | 0.001 |
| iou | 0.7 |
| max_det | 100 |
| device | 0 |

Prediction statistics：

| Check | Result |
| --- | ---: |
| test images | 1000 |
| TXT files | 1000 |
| missing TXT | 0 |
| extra TXT | 0 |
| non-empty TXT | 999 |
| empty TXT | 1 |
| total predictions | 49720 |
| mean predictions/image | 49.72 |
| max predictions/image | 100 |
| images reaching 100-box cap | 124 |
| format errors | 0 |

Important reproduction note：正式 submission 推理时，服务器已应用与 [PR #6](https://github.com/liucgg46-glitch/AIC2026-Multimodal-Detection/pull/6) 相同的两个 hotfix：

1. directory source + `batch=1` + `stream=True`；
2. 跳过 width/height `<= 0` 的退化预测框。

PR #6 已进入 `main`。Phase 1 final main 为 `560d9a92c78af0b392bf8678dd9772d5945ba3e6`，但该 commit 不是 E001 训练时的 commit。

## 5. E002_IR_YOLO11N_CLEAN — completed

| Field | Value |
| --- | --- |
| Status | Completed; results supplied by the user on 2026-09-12 |
| Date | 2026-09-12 |
| Branch | `feature/phase2-experiments` |
| Preparation base | `6eaa6f054bef3b3b8f35290f040d20fb3ff81f0c` (not the E002 training commit) |
| E002 training commit | `62abdac3eb638ac09e2da815cff9687175ff3f6f` |
| Config | `configs/experiments/E002_IR_YOLO11N_CLEAN.yaml` |
| Model / initial weights | YOLO11n pretrained; same E001 initial `yolo11n.pt` artifact, placed at `weights/yolo11n.pt`; SHA-256 confirmed below |
| Input | IR-only, C1 `raw3`, preprocessing=none, staging=copy, isolation=true |
| Dataset YAML | `data/processed/ir_trainable/raw3/data.yaml` |
| Labels | `data/processed/train/labels_clean` |
| Split | Existing 1600/400; seed=2026; no split regeneration |
| Run name / checkpoint | `E002_IR_YOLO11N_CLEAN` / `runs/E002_IR_YOLO11N_CLEAN/weights/best.pt` |
| Runtime / best epoch / per-class AP | unavailable / 88 / unavailable |
| Submission / leaderboard | S002 / 20.7870 |
| Conclusion | Fixed-val IR raw3 mAP50-95=0.20181, below E001 CLEAN 0.37346; this does not establish a general conclusion about all IR representations. |

### E001 historical evidence recovered for E002

On 2026-09-12 the user confirmed values recovered from the server's
`runs/E001_RGB_YOLO11N_CLEAN/args.yaml` and `results.csv`. These files were not
independently opened in this local task. This supplements, rather than fabricates,
the historical record above. The E001 dataset argument was
`data/processed/rgb_yolo_clean/data.yaml`; the initial weight artifact was
`source_packages/yolo11n.pt` under the server's AIC2026 directory (personal absolute
path intentionally omitted). E002 must reuse that initial artifact, not E001 best.pt.
The initial artifact hash was subsequently confirmed by the user; see the completion record below.

Confirmed best-epoch results remain epoch=78, Precision=0.81605, Recall=0.55820,
mAP50=0.61119, mAP50-95=0.37346.

The historical args values below are configuration values, not proof that all were
explicit command-line overrides. In particular `optimizer=auto` remains unchanged:
the actual optimizer selected internally and effective learning rate are unavailable.
Do not replace auto with a guessed optimizer or claim lr0 is the effective auto LR.
Exact historical Python/PyTorch/Ultralytics/CUDA provenance and training commit remain
unavailable. A later server environment inspection must be labeled current confirmed
environment. Version differences limit strict cross-run equivalence despite matching
all recovered parameters. Unreported historical defaults are not inferred.

### E001 to E002 behavior controls

All following recovered args values are explicitly retained in the E002 config:

| Parameters | Values (same order) |
| --- | --- |
| epochs, batch, imgsz, workers, device | 100, 32, 640, 8, 0 |
| optimizer, seed, deterministic, pretrained, amp, cache | auto, 2026, true, true, true, false |
| patience, rect, cos_lr, close_mosaic, resume, freeze | 100, false, false, 10, false, null |
| fraction, multi_scale, dropout, val, plots | 1.0, false, 0.0, true, true |
| lr0, lrf, momentum, weight_decay | 0.01, 0.01, 0.937, 0.0005 |
| warmup_epochs, warmup_momentum, warmup_bias_lr, nbs | 3.0, 0.8, 0.1, 64 |
| hsv_h, hsv_s, hsv_v | 0.015, 0.7, 0.4 |
| degrees, translate, scale, shear, perspective | 0.0, 0.1, 0.5, 0.0, 0.0 |
| flipud, fliplr, bgr | 0.0, 0.5, 0.0 |
| mosaic, mixup, cutmix, copy_paste, copy_paste_mode | 1.0, 0.0, 0.0, 0.0, flip |

`save=true` follows the existing E001 experiment log. Output name changes to E002;
`exist_ok=false` avoids deliberately reusing an existing run. The only intended
major experimental variable is RGB -> IR raw3. Keep the recovered HSV augmentation
parameters even for IR; do not add CLAHE, extra normalization or modality-specific
augmentation. Fixed labels and split stay identical. No Fusion or prelim_test tuning.

### Historical E002 preparation plan (retained; completion recorded below)

Before training, confirm the local initial weights exist and match the E001 source
artifact SHA-256; do not rely on automatic weight downloading. Prepare on the server
from the exact approved checkout so generated YAML uses server-local paths:

```bash
python scripts/data/prepare_ir_yolo.py --representation raw3 --link-mode copy
```

Check the manifest, 1600/400 mappings and isolation before training. Existing output
requires deliberate inspection before using `--force`. No raw or label edits.

Small GPU smoke: reuse formal YAML with only smoke overrides `epochs=1`,
`fraction=0.04`, separate experiment ID/name `SMOKE_IR_001`. This selects approximately
64 of the 1600 train images (two batches at batch=32), preserving imgsz/workers/AMP.
Local installed Ultralytics source applies fraction only to train, so all 400 val
images remain; final validation may run again. This is not a 1600-image training
epoch and not a full validation subsample. `val=false` does not guarantee skipping
final validation. Confirm this behavior in the chosen server version before smoke.
The reduced run retains formal augmentation settings but cannot exercise the full
warmup/close-mosaic schedule; auto optimizer may differ for a short run. No smoke
metrics are used for model selection. If the selected images yield no usable targets,
inspect and revise the smoke plan before proceeding.

Generate the derived smoke config without changing the formal config:

```bash
python -c "from pathlib import Path; import yaml; c=yaml.safe_load(Path('configs/experiments/E002_IR_YOLO11N_CLEAN.yaml').read_text(encoding='utf-8')); c.update(experiment_id='SMOKE_IR_001',name='SMOKE_IR_001',epochs=1,fraction=0.04); p=Path('runs/SMOKE_IR_001_config.yaml'); assert not p.exists(), 'smoke config already exists'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')"
python scripts/train/train_e002.py --config runs/SMOKE_IR_001_config.yaml
```

Smoke acceptance: correct IR/label loading, finite loss, forward/backward completed,
CUDA and AMP checked in logs, no OOM at batch=32, record peak VRAM, checkpoint/log
outputs present. Smoke weights must never initialize the full run.

```bash
python scripts/train/train_e002.py --config configs/experiments/E002_IR_YOLO11N_CLEAN.yaml
```

Gate: review -> successful fetch and remote-state check -> authorized formal commit
-> bundle that commit -> server checkout exact SHA and clean-state check -> inspect
current environment and initial weight identity -> generate copy staging -> GPU smoke
-> separately authorized full training. If code/config changes after smoke, commit and
sync a new SHA before training. No server-first training or retrospective commit.

Local preparation checks (2026-09-12, existing aic2026 Conda environment): C1 plus
configuration tests 67 passed / 1 skipped; full repository suite 91 passed / 1 skipped.
The skip is the Windows real-symlink permission case. Configuration integration uses
a mocked model: no downloads, GPU smoke or training. Python 3.8 syntax check for the
new test passed; this is not an actual server Python 3.8.10 runtime validation.
`git diff --check` passed. Initial weights are not present locally; server artifact
verification, server compatibility checks and successful fetch remain pre-run gates.


### Final freeze review additions

The user supplied the additional historical args below; all are now explicit:

| Parameters | Values (same order) |
| --- | --- |
| box, cls, dfl | 7.5, 0.5, 1.5 |
| single_cls, profile, compile | false, false, false |
| split, conf, iou, max_det | val, null, 0.7, 300 |
| half, dnn, augment, agnostic_nms, classes | false, false, false, false, null |

The existing entry passes these fields unchanged to Ultralytics. Some fields apply
only in relevant backend/mode contexts: preserve them without claiming each changes
training. `conf=null` retains validation's default threshold behavior, not a submission
threshold; `max_det=300` is the historical validation setting, not the submission cap.
`augment=false` controls inference augmentation, not the training mosaic/HSV pipeline.
Historical `auto_augment=randaugment` and `erasing=0.4` belong to classification
augmentation in the inspected local Ultralytics source, not the detection pipeline;
they are intentionally omitted. No export-only settings were copied.

E002 initial weight SHA256 == E001 initial artifact SHA256 is mandatory before any
server run. The new `scripts/train/train_e002.py` is only a preflight adapter: it
requires the fixed nonempty local artifact and resume=false/pretrained=true, then
calls the unchanged `train_rgb.main()`. It neither downloads nor implements training.
Hash equality requires comparison with the original server artifact; file presence
alone is not a provenance check. Use this adapter for both smoke and full E002.
The E001 entry and C1 remain unchanged. Initial weights are not committed.


### E002 completion record — user-confirmed 2026-09-12

These values were supplied by the user from the completed run; this local task did
not rerun training, re-validation or submission. Historical metrics and later
re-validation remain separate. Runtime, per-class AP and exact runtime environment
are unavailable in this record.

| Metric source | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Historical training results.csv | 88 | 0.63125 | 0.32160 | 0.36203 | 0.20181 |
| best.pt re-validation | — | 0.6334556548796438 | 0.32113402692552656 | 0.3612698128051061 | 0.20172330761354743 |

| Artifact | SHA256 |
| --- | --- |
| Initial yolo11n.pt (same E001/E002 artifact) | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| best.pt | `f330a07afa0dd985a7ab9d240593028769ed82268ad3c117a9e309adf0618881` |
| last.pt | `d1d2f31df1d706a407f1f9c86e06c8b6a4655427ccad35787755512bd5dc6c34` |

## 6. Submission S002

| Field | Value |
| --- | --- |
| Internal name | `E002_IR_YOLO11N_CLEAN_PRELIM_001` |
| Source experiment | `E002_IR_YOLO11N_CLEAN` |
| Leaderboard score / rank | 20.7870 / 373 |
| Submission time | 2026-09-12 11:12:21 |
| Submission ZIP SHA256 | `6a0da1a7e5cd2e6dc466170dc5091af73481d2e30fdab25d176c17df5f319c86` |

Leaderboard is recorded for submission provenance only, not parameter selection.

## 7. E003_DEPTH_YOLO11N_CLEAN — frozen configuration, training pending

| Field | Value |
| --- | --- |
| Date / branch | 2026-09-12 / `feature/phase2-experiments` |
| Preparation HEAD | `305ff0602951fe07b1246b067b3542ca4ad833bc` |
| Main base | `5e6dfa3d140c45f81667225635838524fe78b81f` |
| Training code commit | The commit introducing this E003 config and adapter; resolve its full SHA from Git and verify server HEAD before execution. No training has run. |
| Config / run name | `configs/experiments/E003_DEPTH_YOLO11N_CLEAN.yaml` / `E003_DEPTH_YOLO11N_CLEAN` |
| Model | Same initial YOLO11n artifact at `weights/yolo11n.pt`; SHA256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| Staging directory | `data/processed/depth_trainable/inverse` |
| Dataset / manifest | `data/processed/depth_trainable/inverse/data.yaml` / `data/processed/depth_trainable/inverse/manifest.json` |
| Labels / split | `data/processed/train/labels_clean`; existing `data/splits/train.txt` (1600) and `data/splits/val.txt` (400) |
| PNG policy | uint16 mm; valid=depth>0; clip valid depth to [300,19999]; reciprocal scaled with 1+floor(254*x+0.5); invalid=0, valid=[1,255]; repeat to uint8 x3 |
| JPG policy | Physical unit unknown; byte-preserving passthrough, never interpreted as mm |
| Staging isolation | Generated PNG; copied JPG and labels; no source modification |
| GPU / runtime / metrics / per-class AP / checkpoint | pending; no E003 training or GPU smoke performed |
| Conclusion | pending |

C4 validation evidence: `outputs/analysis/depth_c2_c4_validation.json` and `.log`,
produced by validation code `3a3662d896ed08d184cbcae3ea30b166662310f6`.
The recorded inverse manifest SHA256 is
`328f9cfc2d3a0861038d4c5af521a86700e2a538dc01f87283916568d3123e09`.
This preparation checkout has no generated inverse staging; its manifest is not
claimed to have been read locally. Paths and metadata are established by the C4
builder and reviewed evidence. Server staging and manifest verification remain a gate.
The evidence records 2000 decodable uint8 x3 outputs, fixed identities, no missing,
duplicate or overlap, and real train/val Ultralytics batches. It is CPU loader
validation in Python 3.10, not GPU training or exact historical E001 environment.

| Controlled factor | E001 CLEAN | E002 | E003 |
| --- | --- | --- | --- |
| Input (only main variable) | RGB | IR raw3 | PNG inverse + JPG passthrough |
| Initial model artifact | Same user-confirmed YOLO11n SHA256 above | Same | Same, enforced by wrapper |
| Split / labels / seed | 1600/400 / labels_clean / 2026 | Same | Same |
| epochs / batch / imgsz / workers / device | 100 / 32 / 640 / 8 / 0 | Same | Same |
| Optimizer / scheduler / loss / augmentation / validation args | Recovered args documented above | Explicit formal YAML | All E002 YAML values unchanged |
| Historical software provenance | Exact historical environment unavailable | Unavailable in this record | Record actual environment at server Gate |

The E003 and E002 YAML mappings differ only in experiment_id, name and data.
Keep optimizer=auto; do not infer its historical internal selection. The new thin
`train_e003.py` checks the fixed inverse YAML, local weight SHA256 and fresh pretrained
start, then delegates to unchanged `train_rgb.main()`. E002's frozen adapter remains
unchanged. Missing dataset, missing/wrong weights and alternate candidates fail before
model creation. No E001/E002 best.pt, downloads, Fusion or prelim_test tuning.

### E003 future server commands — not executed in this task

After local tests, review and commit + push: sync exactly that commit, verify server
HEAD and clean status, inspect current CUDA/software environment, and stage:

```bash
python scripts/data/prepare_depth_candidate_yolo.py --candidate inverse
```

Default copy isolation is retained. Inspect existing staging before any explicit
`--force`. Verify generated manifest identities and representation on the server.

Derive a separate smoke config at runtime; leave formal YAML unchanged:

```bash
python -c "from pathlib import Path; import yaml; c=yaml.safe_load(Path('configs/experiments/E003_DEPTH_YOLO11N_CLEAN.yaml').read_text(encoding='utf-8')); c.update(experiment_id='SMOKE_DEPTH_001',name='SMOKE_DEPTH_001',epochs=1,fraction=0.04); p=Path('runs/SMOKE_DEPTH_001_config.yaml'); assert not p.exists(), 'smoke config already exists'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(yaml.safe_dump(c,sort_keys=False),encoding='utf-8')"
python scripts/train/train_e003.py --config runs/SMOKE_DEPTH_001_config.yaml
```

Smoke uses about 64 train images, batch=32, imgsz=640, workers=8, AMP=true; fraction
reduces train only, with 400 val images retained. Confirm behavior in the server
Ultralytics version. Check CUDA/AMP, forward/backward, finite losses, no OOM, peak
VRAM and checkpoint/log output. Short-run auto optimizer/scheduling may differ;
smoke scores are not performance evidence. Never initialize full training from smoke.

Only after the exact-SHA server Gate and successful GPU smoke:

```bash
python scripts/train/train_e003.py --config configs/experiments/E003_DEPTH_YOLO11N_CLEAN.yaml
```

This task stops after the tested E003 training commit is pushed. Bundle, server
checkout/staging, GPU smoke, full training and test inference remain unexecuted.


### E003 local freeze validation — 2026-09-12

Configuration/preflight tests: 20 passed. Full repository tests: 200 passed,
1 skipped (Windows symlink permission case). Model construction/training is mocked
in configuration integration tests; no actual training or download occurred.
Python 3.8 AST syntax checks passed for the adapter, shared entry and configuration
tests; actual Python 3.8 server runtime remains a server Gate check.
`git diff --check` passed. Successful `git fetch origin` confirmed main remains
`5e6dfa3d140c45f81667225635838524fe78b81f`, already included in A's branch,
with no divergence from the published A branch before this commit.
Final review confirmed only the E003 config/adapter, config tests and experiment log
changed; E001/E002 training code, C2-C4, split, labels and raw remain unchanged.
No weights or generated staging are added to Git. The earlier E001 historical
record and B/C analysis artifacts are retained.
