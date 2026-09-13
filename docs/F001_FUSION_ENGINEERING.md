# F001 RGB + IR gated fusion: pre-commit engineering review

Experiment: `F001_RGB_IR_GATED_P45_YOLO11N`

Required development branch: `feature/multimodal-fusion`

Baseline main: `6fb1a3d50c216797d49a0a3a7bacf4c4c6ce4597`

This implementation is repository-local. It does not modify Ultralytics,
canonical data, E001/E002/E003 outputs, or historical experiment records.

## Review corrections from the first skeleton

1. The first implementation rejected every version except local 8.4.144. The
   supported set is now limited to the two source-audited releases: 8.3.253 and
   8.4.144. Any other version fails closed and must be audited.
2. Overriding private `_predict_once` was avoidable and its signature differs
   between those releases. F001 now overrides public `predict`; there are no
   private method overrides.
3. Random initialization was unsuitable for a formal experiment. F001 now
   authenticates and loads `weights/yolo11n.pt` with SHA-256
   `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
4. The former “same as RGB-only” statement was too broad. The precise fallback
   definition appears below.
5. Ultralytics 8.3.253 treats `multi_scale` as bool, while 8.4.144 accepts its
   newer numeric form. The common configuration is explicitly `false`.
6. Letterbox `ratio_pad` metadata differs between the versions. The paired
   wrapper normalizes it to `((gain_h, gain_w), (pad_w, pad_h))`.
7. The entry point now supports real training through explicit `--train`, while
   no mode/default remains a safe check-only operation.
8. The server smoke now constructs `FusionTrainer(overrides=...)` and calls the
   audited upstream `BaseTrainer._setup_train()` entry point. It no longer
   builds a manual loader, SGD optimizer, scaler, EMA, scheduler, or checkpoint.

## Architecture and exact graph mapping

The model subclasses `DetectionModel`, constructs the native 12-class YOLO11n
graph, and deep-copies native backbone modules 0 through 10 for IR. RGB and IR
encoders begin with the same canonical pretrained values but do not share tensor
storage. A `[B,6,H,W]` transport tensor is immediately split into RGB `0:3` and
IR `3:6` before either encoder convolution.

| Meaning | YOLO11n module | Stride | Consumer |
| --- | ---: | ---: | --- |
| RGB/IR P3 backbone output | 4 (`C3k2`) | 8 | neck concat module 15 |
| RGB/IR P4 backbone output | 6 (`C3k2`) | 16 | fusion4, then neck concat module 12 |
| RGB/IR P5 backbone output | 10 (`C2PSA`, after 9 `SPPF`) | 32 | fusion5, then neck root and concat module 21 |
| neck P3 output | 16 | 8 | Detect input 0 |
| neck P4 output | 19 | 16 | Detect input 1 |
| neck P5 output | 22 | 32 | Detect input 2 |
| detection head | 23 (`Detect`) | 8, 16, 32 | modules 16, 19, 22 |

Every construction asserts all 24 module types, relevant `from` edges, module
indices, and `[8,16,32]` Detect strides. Tests with a 160-square input assert
backbone and Detect feature maps are respectively 20, 10, and 5 pixels.

P3 means the backbone skip at module 4 is RGB-only. The native top-down neck's
stride-8 prediction can still receive information propagated down from fused
higher levels; it is not an IR-isolated detection branch.

Fusion is:

```text
F4 = R4 + sigmoid(gate4(concat(R4,I4))) * proj4(I4)
F5 = R5 + sigmoid(gate5(concat(R5,I5))) * proj5(I5)
```

Projection weight and bias are zero. Gate weight and bias are also zero, so the
initial gate is exactly 0.5, within `[0,1]`. Because projection output is exactly
zero, fused P4/P5 equal RGB P4/P5 exactly at initialization.

The complete initialized dual model equals a separately instantiated **12-class
RGB YOLO11n graph with the exact same RGB backbone/neck/head state**. It does not
equal the canonical checkpoint's complete prediction tensor: that checkpoint is
an 80-class model, while F001 is 12-class and part of its classification branch
has different shapes. Tests and reports use this narrower, correct definition.

## Canonical pretrained initialization

Checkpoint bytes are SHA-256 verified before `torch.load`. Loading is explicit
shape-match only and fails before mutation for an unapproved missing key, shape
mismatch, or unexpected source key. The only allowed mismatches are under the
YOLO classification branch `model.23.cv3.*`, caused by 80 to 12 classes. These
target tensors retain deterministic seeded initialization. Fusion is reset to
the zero-projection/0.5-gate state after loading.

Observed load report on both audited model graphs:

| Group | Loaded keys | Total | Key ratio | Parameter ratio |
| --- | ---: | ---: | ---: | ---: |
| RGB encoder modules 0–10 | 240 | 240 | 100% | 100% |
| IR encoder modules 0–10 | 240 | 240 | 100% | 100% |
| neck + head modules 11–23 | 208 | 259 | 80.31% | 96.16% |

Overall report: 688 loaded target keys, 8 missing fusion keys, 0 unexpected
source keys, and 51 listed classification shape mismatches. The CLI prints the
full loaded/missing/unexpected/mismatch key lists and all ratios. E001 `best.pt`
is neither accepted nor used.

`weights/` is ignored by Git, so every checkout must provision the canonical
file separately. Check-only, smoke, formal training and server smoke all verify
the required hash.

## Pairing and augmentation contract

Fixed train=1600, val=400, seed=2026 and canonical `labels_clean` are enforced.
The loader reuses normalized split SHA checks, case-insensitive unique-stem
indexing and the clean-label aggregate SHA. Missing, extra, duplicate,
case-ambiguous, overlapping or altered split/label identities fail immediately.
Dataset construction decodes every RGB/IR record, requires equal spatial shape,
and checks uint8 three-channel raw3 plus label fields.

E002 raw3 is a byte-preserving link/copy of canonical raw IR. F001 reads the same
canonical source directly without regenerating or modifying the E002 view.

RGB and IR are concatenated before geometry. One LetterBox and one RandomFlip per
direction transform all six channels and one shared `Instances` object. RGB HSV
is applied only to channels 0:3; IR never receives HSV. Each BGR triplet is
converted independently to RGB during formatting.

Enabled: letterbox, horizontal flip 0.5, vertical flip support (formal value 0),
and RGB HSV. Disabled and rejected: affine rotation/translation/scale/shear,
perspective, mosaic, mixup, cutmix, copy-paste, multi-scale, BGR randomization,
and custom/Albumentations transforms.

### E001 comparability

The repository's canonical E001 record gives epochs, image size, batch, workers,
optimizer, seed, pretrained/AMP/cache/plots/validation, but does **not** preserve
its individual augmentation values or its exact Ultralytics version. Therefore
an exact augmentation equality claim cannot be audited from repository evidence.
Relative to Ultralytics 8.3.253 defaults, F001 changes at least mosaic 1.0 to 0,
translate 0.1 to 0, and scale 0.5 to 0; HSV and horizontal flip retain the same
numeric defaults. F001-vs-E001 consequently has an augmentation confound.

A required follow-up control should use the F001 data split, initialization,
hyperparameters and augmentation contract with one RGB encoder and the native
RGB neck/head. It must retain batch/accumulation and all optimization settings.
This `RGB_CONTROL_F001_AUG` design separates fusion gain/loss from augmentation
policy gain/loss. It is designed here only; no control or F001 training was run.

## Ultralytics compatibility audit

8.3.253 was downloaded as an uninstalled wheel, inspected, and run from an
isolated `PYTHONPATH`. Its wheel SHA-256 during review was
`c2ad53388ce00e0587a21745a9c5ab1bdcb4169b74545488fab3fd1b25fe86f2`.
Local 8.4.144 remained installed. No site-packages source was edited.

| Actual dependency | Risk | Protection |
| --- | --- | --- |
| `DetectionTrainer`, `YOLODataset` | public/stable class | subclass/reuse only |
| `DetectionModel`, `DetectionValidator` | semi-internal | exact-version probe and smoke |
| trainer dataset/model/preprocess/validator methods | semi-internal | signature probe; minimal overrides |
| `BaseTrainer._setup_train`, optimizer/checkpoint methods | private/semi-internal | exact-version probe; server smoke call only, never overridden |
| `BaseModel.forward/loss/predict` | semi-internal | preserve dict loss dispatch; public `predict` override |
| `YOLODataset.collate_fn` | semi-internal | batch-contract tests on both versions |
| `Compose`, geometry/color/format transforms, `Instances` | semi-internal | synthetic geometry tests on both versions |
| graph `.model/.save`, module `.i/.f`, index/key layout | private/internal | complete graph assertions and strict load report |
| checkpoint `['model'].state_dict()` layout | private/internal | SHA plus key/shape accounting |

There are no private method overrides. The server compatibility smoke directly
calls `_setup_train()` so it can stop before the epoch loop. `BaseDataset`,
`build_yolo_dataset` and `v8_transforms` were source-reviewed but are not runtime
dependencies.

Confirmed 8.3.253 differences handled by F001: `BaseModel.predict` includes a
`visualize` parameter; `set_model_names_for_load` is absent; validator uses
`args.half`; loss items are a tensor; LetterBox has older ratio metadata; and
`multi_scale` is bool. The repository probe passed against isolated 8.3.253 code
under the local Python/torch stack. Exact Python 3.8.10 + torch 2.4.1/cu118
execution remains mandatory server smoke, rather than a claimed local result.

All added Python files parse with the Python 3.8 AST grammar. A test rejects
`match/case`, PEP 604 annotation unions, and built-in generic annotations such as
`list[str]`, `dict[...]`, `tuple[...]`, and `set[...]`.

## CLI and server contract

Safe local modes:

```text
python scripts/train/train_fusion.py --config configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml
python scripts/train/train_fusion.py --check-only
python scripts/train/train_fusion.py --smoke --imgsz 64
```

No mode is equivalent to check-only. Formal training requires explicit:

```text
python scripts/train/train_fusion.py --config configs/experiments/F001_RGB_IR_GATED_P45_YOLO11N.yaml --train --expected-sha EXACT_40_CHAR_SHA
```

It prints experiment ID; commit, branch and dirty state; Python, torch,
torchvision, Ultralytics, CUDA and GPU; model/data/augmentation configuration;
checkpoint path/hash; counts; seed, epochs, batch, image size, workers, optimizer,
and batch policy. Formal `--train` requires `--expected-sha`, exact HEAD equality,
a clean worktree, and detached HEAD before constructing a trainer. Check-only
and local CPU smoke continue to allow the dirty feature branch while printing
its actual branch, SHA, and dirty state.

At the end of formal Trainer setup, an upstream callback prints resolved runtime
provenance: requested and resolved optimizer, optimizer parameter groups,
physical train batch, resolved validation batch, `nbs`, accumulation, nominal
effective batch, calculated optimizer iterations, resolved AMP, EMA, scaler,
and scheduler. These are runtime values rather than copies of the YAML request.

Formal target is physical batch 32. Ultralytics `nbs=64` means its recorded
post-warmup accumulation is 2 and nominal effective batch is 64. This matches the
framework policy explicitly; it is not disabled. Feasibility must be tested in
order 1, 2, 8, 16, 32. If 32 is infeasible, do not silently train at a lower
batch: change batch/accumulation only as a reviewed and recorded experiment
variable.

After a formal commit, on the unchanged server environment:

```text
git checkout --detach EXACT_SHA
git status --short
python scripts/train/probe_fusion_compatibility.py --server
python scripts/train/train_fusion.py --check-only
python scripts/train/smoke_fusion_server.py --expected-sha EXACT_SHA --batch 1
python scripts/train/smoke_fusion_server.py --expected-sha EXACT_SHA --batch 2
python scripts/train/smoke_fusion_server.py --expected-sha EXACT_SHA --batch 8
python scripts/train/smoke_fusion_server.py --expected-sha EXACT_SHA --batch 16
python scripts/train/smoke_fusion_server.py --expected-sha EXACT_SHA --batch 32
```

The probe checks exact server versions, CUDA 11.8, signatures, six-channel
LetterBox, graph construction and forward without modifying data. Each batch
smoke requires a clean, detached exact SHA. It constructs the real
`FusionTrainer`, calls `_setup_train()`, and checks the resulting train loader,
formal 2x validation loader, validator, ModelEMA, AMP scaler, AdamW optimizer,
scheduler, and accumulation. It then performs two real
preprocess/loss/AMP-backward/`trainer.optimizer_step()` cycles from one real
train-loader iterator. Step 1 must materialize finite AdamW state. Step 2 checks
that the same state is already resident before its forward/backward, retains
valid state afterward, and advances EMA for a total delta of two. Only then does
the smoke perform one first-batch forward through the formal validation loader.

For target batch 32, PASS therefore requires two train batch 32 optimization
cycles, including the state-resident second backward, plus formal validation
batch 64 first-batch forward. A val64 OOM is
a failed batch-32 feasibility result; this smoke does not change validation
batch policy. Metric integration is checked separately with a `Subset` whose
dataset length is exactly one. `trainer.save_model()` writes a temporary
EMA-based `best.pt`, then upstream `strip_optimizer()` rewrites that temporary
file exactly as `final_eval()` does. Both `ultralytics.nn.tasks.load_checkpoint`
and a standalone `DetectionValidator(model=the_same_stripped_checkpoint_path)`
reload it, confirm the restored
type is `DualStreamModel`, and execute six-channel forward/metrics. This covers
the same checkpoint-path boundary used by `final_eval()` without starting an
epoch or iterating the 400-image validation split.

All project and validator outputs, settings, incidental AMP-check downloads, and
checkpoint files are redirected to temporary directories. Stop the batch
sequence at OOM and record the first failing batch. GPU/AMP/server execution has
not been performed locally.

## Review verification (2026-09-13)

- Local installed stack compatibility probe: PASS on Python 3.10.21,
  torch 2.14.0+cpu, torchvision 0.29.0+cpu, Ultralytics 8.4.144.
- Isolated old-source probe: PASS on Ultralytics 8.3.253 using the local
  Python/torch runtime. All 46 non-real-data fusion tests also passed with that
  isolated version. This validates API behavior but does not replace the exact
  server-stack probe.
- Current-version full regression excluding the separately privileged real-data
  case: 279 passed, 1 skipped, 1 deselected in 31.15 seconds.
- Real-data case: 1 passed in 192.21 seconds. It decoded all 2000 RGB/IR pairs and
  executed train-loader and val-loader CPU forwards. The sole warning was pytest
  being unable to write its shared cache under the privileged process.
- Combined current-version result: 280 passed, 1 pre-existing Windows symlink
  skip. The 47 fusion tests include Python 3.8 syntax checks, source-isolated
  8.3 compatibility, actual `BaseTrainer.setup_model`, strict canonical weight
  loading, augmentation alignment, native loss/validator, serialization and
  dirty-tree training refusal.
- Canonical pretrained CLI smoke: train=1600, val=400; input `(1,6,64,64)`;
  prediction `(1,16,84)`; PASS. No backward or training was run locally.

Third-round runtime-fidelity verification:

- Targeted runtime/control/checkpoint tests: 13 passed against source-isolated
  Ultralytics 8.4.144 and 13 passed against source-isolated 8.3.253.
- All fusion tests excluding the separately privileged canonical decode case:
  56 passed, 1 deselected on each audited Ultralytics source tree.
- Full repository `tests/` on the 8.4.144 source tree: 288 passed, 1 existing
  Windows symlink skip, 1 canonical-data case deselected. The canonical case was
  run again after the final memory-fidelity micro-fix and passed after decoding
  all 2000 pairs in 126.86 seconds.
- The runtime tests call actual upstream `_setup_train()`, resolve AdamW at 2500
  iterations, assert accumulation 2 and validation batch 64 for physical batch
  32, execute a second backward with resident AdamW state and EMA delta two, and
  exercise `save_model -> strip_optimizer -> load_checkpoint -> DetectionValidator(path)`.
- These third-round commands used the audited wheels without installing them.
  This shell exposed Python 3.14.2, torch 2.9.1+cpu, and torchvision
  0.24.1+cpu, so they are source/API regression evidence rather than an exact
  substitute for the declared local or CUDA server stacks.
- Exact Python 3.8.10, torch 2.4.1+cu118, CUDA 11.8, RTX 3090 execution remains
  the required post-commit server smoke. No server or CUDA PASS is claimed.
