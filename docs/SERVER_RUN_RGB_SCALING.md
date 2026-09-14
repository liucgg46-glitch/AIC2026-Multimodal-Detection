# RGB scaling 服务器执行单

目标：先测训练配方，再只改变模型容量，最后只改变输入分辨率。

## 实验矩阵

| 顺序 | 配置 | 模型 | imgsz | batch | 目的 |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `AUDIT_N640.yaml` | YOLO11n | 640 | 32 | 新配方相对 E001 的收益 |
| 2 | `AUDIT_S640.yaml` | YOLO11s | 640 | 32 | 只测模型容量 |
| 3 | `AUDIT_S960.yaml` | YOLO11s | 960 | 16 | 在有效 batch=64 下测小目标尺度 |

三个实验均使用固定 1600/400 split、labels_clean、AdamW、cosine LR、200 epoch、
patience=60 和相同增强。batch 变化由 `nbs=64` 的梯度累积保持名义有效 batch=64。

## 服务器准备

在正式 commit 已同步到服务器后：

```bash
cd /root/data1/AIC2026/AIC2026-Multimodal-Detection
source /root/data1/AIC2026/aic2026_env/bin/activate
git checkout --detach <40位实验SHA>
git rev-parse HEAD
git status --porcelain
python --version
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())"
```

`git status --porcelain` 必须为空。若服务器尚无 YOLO11s 公开预训练权重，先用可联网环境获取
官方 `yolo11s.pt`，放到仓库忽略目录 `weights/yolo11s.pt`，记录 SHA-256。正式训练与推理
不得依赖在线服务。

## 检查与 smoke

```bash
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_N640.yaml
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_S640.yaml
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_S960.yaml

python scripts/train/train_rgb.py --config configs/experiments/AUDIT_N640.yaml --smoke
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_S640.yaml --smoke
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_S960.yaml --smoke
```

默认命令只检查配置和权重，不训练。三个 smoke 均成功后再正式运行。若 S960 OOM，只调整
物理 batch 到 8，保留 `nbs=64`，并把变化保存为新的配置 commit；不要只在命令行临时覆盖。

## 正式训练

将 `<SHA>` 替换成同一个完整 40 位 detached commit：

```bash
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_N640.yaml --train --expected-sha <SHA>
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_S640.yaml --train --expected-sha <SHA>
python scripts/train/train_rgb.py --config configs/experiments/AUDIT_S960.yaml --train --expected-sha <SHA>
```

先完成 N640。只有它正常收敛才继续 S640；只有 S640 优于 N640 才继续 S960，避免浪费算力。

## 回收结果

```bash
python scripts/analysis/summarize_rgb_scaling.py \
  runs/AUDIT_N640 runs/AUDIT_S640 runs/AUDIT_S960 \
  --output outputs/analysis/rgb_scaling_summary.json
```

同步以下文件回本地：

- 三个 run 的 `results.csv`、`args.yaml`、`weights/best.pt`、`results.png`；
- `outputs/analysis/rgb_scaling_summary.json`；
- 三次完整终端输出或重定向日志；
- `git rev-parse HEAD`、`python --version`、PyTorch/Ultralytics/CUDA/GPU 信息；
- `weights/yolo11s.pt` 的 SHA-256。

若按阶段停止，只向汇总脚本传入已经完成的 run。
