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
