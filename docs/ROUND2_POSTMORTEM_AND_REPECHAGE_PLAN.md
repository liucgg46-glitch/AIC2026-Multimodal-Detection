# Round 2 postmortem and repechage priorities (2026-09-15)

## Measured result

All Round 2 runs used commit `f48a6e44658baf9da45b8dee35ca6ed90d8cac8a`, the 1600/400
split and `labels_clean`.

| run | history peak mAP50-95 | best.pt reval | interpretation |
| --- | ---: | ---: | --- |
| S960 control | 0.45578 @ epoch 162 | about 0.4550 | keeping mosaic on did not improve the old S960 result |
| M960 | 0.47923 @ epoch 100 | 0.479013 | best validated RGB backbone so far |
| S960-P2 | 0.36258 @ epoch 169 | 0.361514 | invalid P2-vs-S comparison due to discarded initialization |

M960 improves best.pt reval over the prior S960 `0.45883` by about 0.0202 mAP. Its validation
box/DFL losses rise after the peak while train losses continue to fall; early stopping at 160 is
consistent with overfitting. The control weakens the earlier hypothesis that closing mosaic was the
main source of score loss. Capacity helped; training longer and changing only mosaic did not.

The locally synchronized M960 `best.pt` has SHA-256
`097852eb7357dc8b7db72fef82e0485e5542a67b051977451edd4305be70ac85`. Verify that the
server checkpoint used for any leaderboard submission has this same hash.

## P2 root cause and correction

The original P2 entry point called `initialize_p2_model(model.model, checkpoint)` immediately after
`YOLO(custom_yaml)`. In Ultralytics 8.3.253, `YOLO.train()` constructs a new trainer model with
`weights=self.model if self.ckpt else None`. A YAML-created YOLO instance has no checkpoint, so its
first model and the 421 copied tensors were discarded. The trained P2 was therefore initialized
from scratch. This explains the epoch-1 P2 box/cls/DFL train losses `3.660/5.633/4.019` and
`mAP50-95=0.00784`; the S960 control started at `1.522/2.276/1.209` and `0.22286`.

The repaired entry point applies the frozen S960 checkpoint inside the custom trainer's `get_model`
method, after Ultralytics constructs the actual training model. A local probe on that exact path
reports `421/593` tensors and `0.965216` parameter coverage, with strides `[4, 8, 16, 32]`. The
new P2 path and incompatible classification towers still start from random weights. Consequently,
the old P2 score says nothing reliable about whether the P2 architecture can beat S960. The retry
configuration is `RGB_R3_S960_P2_FIXED_INIT.yaml`, but it is lower priority than repechage fusion.

## Official repechage requirements

The [2026-09-14 repechage notice](https://www.aicomp.cn/notice/notice-1/5278.html) says that teams
with an initial-stage result document submitted by **2026-09-20 20:00** and a displayed leaderboard
score enter the repechage. The repechage dataset will be downloadable in the competition account.
Teams may submit **two** result files per day. The highest repechage leaderboard score is used, and
the score-2 award baseline is **40**. Awards are based on leaderboard rank; submissions also need
reproducible code, model and environment materials by the specified deadline.

The [score-2 rule](https://www.aicomp.cn/tracks/tracks-1/3700.html) asks for a three-modality RGB,
IR and Depth detector. It prohibits using the repechage test set for training, adding external
training data, and simply averaging/voting across differently trained models. Public COCO/ImageNet
pretrained weights are allowed. Repechage deliverables include runnable preprocessing, training and
inference code; a Markdown README with weight download, setup and commands; requirements; and the
leaderboard result file. The half-final additionally needs a PDF technical report.

The notice's schedule and exact upload naming are in two PDF attachments. Their web retrieval timed
out during this audit, so exact score-2 dates and attachment-specific naming must be verified from
the official account/download before preparing the final upload. No date is inferred from other
tracks.

## Next experiment order

1. Protect entry: make sure the current S960 leaderboard result `47.484` and its initial-stage
   result document are both present before September 20. M960 should be submitted only after its
   test predictions pass file/stem/box validation; a val improvement is not a guaranteed test gain.
2. Once the repechage data opens, submit a frozen M960 RGB prediction as a distribution check and
   fallback. Keep the result file and full 40-character model/code provenance.
3. Build a single trainable RGB+IR+Depth model from M960, using synchronized geometry, an IR
   valid-region/border mask, a depth-validity mask and quality-conditioned residual fusion at
   P3/P4/P5. Initialize the RGB path from the strongest verified M960 checkpoint. Use IR/Depth
   dropout and an exact RGB-only fallback to avoid F001's blind residual injection.
4. Train only on the official 2000 labeled pairs, using the fixed local 1600/400 split for model
   selection. Keep the repechage test set prediction-only. Compare val macro mAP50-95, per-class
   AP, modality ablations and aligned/unreliable subsets before spending limited daily submits.
5. Package the winning **single model** with deterministic preprocessing, one-command inference,
   model hash, versioned dependencies and a small offline reproduction test. The leaderboard result
   and submitted weights must come from the same frozen experiment.

P2 retry may be scheduled after the first quality-aware fusion result if tiny-target AP remains a
clear bottleneck. Do not spend another 200-epoch server run on P2 before checking the trainer-side
initialization report and the repechage schedule.
