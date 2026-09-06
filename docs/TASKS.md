# Development Tasks

## Phase 0 - Project Setup

- [x] 建立项目目录
- [x] 建立 Git 仓库
- [x] 建立项目规范文档
- [ ] 建立 Python 环境

## Phase 1 - Dataset Inspection

- [ ] check_dataset.py
- [ ] visualize_multimodal.py
- [ ] analyze_labels.py
- [ ] 确认标签格式
- [ ] 确认三模态尺寸
- [ ] 确认 Depth dtype/range
- [ ] 确认 Infrared 通道
- [ ] 确认模态文件对应关系

## Phase 2 - Dataset Split

- [ ] make_split.py
- [ ] seed=2026
- [ ] train/val 固定划分

## Phase 3 - RGB Baseline

- [ ] Ultralytics YOLO 环境
- [ ] RGB dataset
- [ ] RGB smoke test
- [ ] RGB full training
- [ ] validation
- [ ] inference
- [ ] competition submission

## Phase 4 - Single Modality Ablation

- [ ] IR-only
- [ ] Depth-only

## Phase 5 - Multimodal Baseline

- [ ] RGB+IR
- [ ] RGB+Depth
- [ ] RGB+IR+Depth