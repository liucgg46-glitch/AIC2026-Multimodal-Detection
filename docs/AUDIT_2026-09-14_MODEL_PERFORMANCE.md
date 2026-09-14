# AIC2026 模型性能审查（2026-09-14）

## 结论

当前低分主要来自模型容量、输入尺度、模态表示和融合训练策略，不是提交格式错误。
现有 F001 接线和 checkpoint 序列化没有发现足以解释全部掉分的致命错误，但实验设计
不能证明融合有效：它与 RGB 基线使用了不同增强策略，并且双编码器在 1600 张训练图上
明显过拟合。

本报告只把同步到 `artifacts/formal_artifacts_E002_E003_F001` 的正式产物作为
E002/E003/F001 事实来源。三项实验均没有独立 console log，不推断缺失日志内容。

## 官方约束

- 主指标是 12 类、IoU 0.50 到 0.95（步长 0.05）的 mAP，单阈值 AP 使用 101 点精度包络算术平均。
- 每张测试图必须有同名 TXT，空预测也要有空 TXT；每图最多 100 框。
- 只能使用官方训练数据，可用 ImageNet/COCO/Objects365 等公开预训练权重。
- 禁止把不同结构或训练阶段模型做简单投票、平均集成。

## 正式结果复核

| 实验 | 本地最佳 mAP50-95 | 最佳 epoch | 最后一轮 | 已知榜单 | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| E001 RGB YOLO11n | 0.37346 | 78 | 未同步 | 约 0.40 | 当前主基线容量和尺度不足 |
| E002 IR YOLO11n raw3 | 0.20181 | 88 | 0.19893 | 约 0.20 | 本地与榜单一致，非提交掉分 |
| E003 Depth YOLO11n inverse | 0.18269 | 99 | 0.17293 | 约 0.19 | 到第 99 轮仍在改善，100 epoch 不充分 |
| F001 RGB+IR gated P4/P5 | 0.32530 | 86 | 0.32269 | 约 0.35 | 低于 RGB，本地与榜单方向一致 |

E002/E003 的 `best.pt` 均确认是 12 类 YOLO11n、Ultralytics 8.3.253，并从本地
`weights/yolo11n.pt` 启动。F001 checkpoint 可完整加载为 `DualStreamModel`；P4/P5
projection 和 gate 参数均已离开零初始化，所以 IR 分支并非完全没有训练。

## 已确认的问题

### P0：先建立更强 RGB 基线

训练集只有 2000 组，E001 使用最小的 YOLO11n 和 640 输入。固定验证集 3041 个目标中，
tiny 757 个、small 1066 个；tiny 在 conf=0.25、IoU=0.50 下召回只有 0.3448。
1920x1080 图像 letterbox 到 640 后有效内容高度约 360 像素，小目标进一步缩小。

在没有证明 YOLO11s/YOLO11m 和 960/1280 输入的 RGB 上限前，围绕 YOLO11n-640 优化融合
不是冲击 0.66 的合理主线。已准备三个受控配置：`AUDIT_N640`、`AUDIT_S640`、
`AUDIT_S960`。先比较模型容量，再比较输入尺度；需在服务器 smoke 后冻结实际 batch。

### P0：F001 实验混入增强变量并明显过拟合

E001/E002/E003 使用 mosaic=1.0、translate=0.1、scale=0.5、close_mosaic=10；F001 将这些
全部关闭。F001 因而同时改变了模型和增强，不能将 0.37346 到 0.32530 的差异解释为
“IR 融合降低了性能”。仓库中的 `CTRL001_RGB_F001_AUG` 是必要对照，但它只用于归因，
不是下一条高分主线。

F001 训练 box loss 从 1.6874 降到 0.3279，验证 box loss 从 1.5846 后长期停在约 1.52；
第 40 轮 mAP 已为 0.32396，第 86 轮最高仅 0.32530。这是容量增加、增强不足和小数据共同
造成的明显泛化间隙。

### P0：F001 没有在高分辨率 P3 直接融合 IR

F001 的 P3 backbone skip 是 RGB-only，IR 仅直接进入 P4/P5。数据的主要困难恰好是 tiny/
small 目标，IR 对夜间行人等小目标的细节只能从深层特征间接传播到 P3 检测头。下一版融合
应在 P3/P4/P5 都有受控的 IR 路径，并加入模态 dropout、有效区 mask、对齐鲁棒模块和
RGB teacher/初始化约束，避免辅助模态破坏 RGB 表示。

### P0：Depth 的 PNG 与 JPG 表示语义不一致

E003 实际使用 `depth_trainable/inverse`。PNG uint16 经过 inverse-depth 固定映射，而 JPG
uint8 三通道被原样复制；JPG 的物理映射未知。这两类样本在同一网络中代表不同的数值语义。
训练集中有 1478 张 PNG、122 张 JPG，这会让模型学习格式特征并削弱统一的距离解释。

Depth 重做时至少要显式输入 `valid mask`，并把 source-format token/mask 纳入模型，或按格式
使用两个适配器再映射到公共特征空间。不能再把兼容 YOLO 的三通道图片视为已经解决 Depth
表示问题。E003 在 epoch 99 才最佳，重做表示后也应训练更久并使用 patience。

### P1：训练入口存在容易静默从零训练的配置陷阱

通用 RGB 入口过去允许 `model: yolo11n.yaml` 与 `pretrained: true`。该组合本身不会自动
指定 COCO checkpoint，容易把从零训练误认为预训练。入口现已拒绝该组合，要求 `.pt` 模型
或显式预训练路径。正式 E002/E003 使用 `.pt`，F001 自己显式校验并加载 checkpoint，因此
这个缺陷不是三项已同步实验低分的原因。

### P1：Depth 读取受 Ultralytics 全局 OpenCV patch 影响

Ultralytics 8.3.253 会替换全局 `cv2.imread`，使单通道图返回 HxWx1。旧 Depth 代码严格要求
HxW，因此只要同一 Python 进程先导入 Ultralytics，就会错误拒绝合法 uint16 PNG。现已改为
从字节调用 `cv2.imdecode(IMREAD_UNCHANGED)`，隔离全局 patch。该问题解释流程兼容故障，
不解释已经成功完成的 E003 低分。

### P1：推理默认阈值不适合 AP 提交

`predict_rgb.py` 过去默认 conf=0.25，会永久删掉低置信候选。默认已改为 0.001。同步的 E002/
E003 包最低置信度约为 0.001，因此这不是现有榜单低分原因。

E002 包含 75,774 框，443/1000 张达到 100 框；E003 包含 90,719 框，263/1000 张达到
100 框。格式合法，反映模型有大量低置信候选。赛事也会按置信度截断到 100，不能仅通过
提高提交 conf 修复模型 AP；应在固定 val 用官方规则重建评测器扫阈值。

## 已实施的代码改动

- 新增 `scripts/analysis/evaluate_submission.py`，按赛题 PDF 的 101 点算术平均规则评估固定
  val TXT，并支持 conf sweep；它明确标注为官方脚本重建版，赛事未说明的 tie/match 细节
  采用稳定规则。
- `predict_rgb.py` 默认提交阈值改为 0.001。
- `train_rgb.py` 拒绝含糊的 YAML + pretrained=true，并校验本地离线权重。
- Depth 三条读取路径绕过 Ultralytics 对 `cv2.imread` 的全局修改。
- 新增相关回归测试及三个 RGB 受控实验配置。

在 Ultralytics 8.3.253 隔离源码下，完整测试（排除耗时的 2000 对真实数据用例）为
321 passed、1 skipped、1 deselected；本次相关与 fusion 定向测试为 173 passed、
1 deselected；`git diff --check` 通过。

## 下一轮实验顺序

1. 跑 `AUDIT_N640`，确认新训练入口和显式 AdamW/200 epoch 配置没有回退。
2. 跑 `AUDIT_S640`，只测容量收益；若明显优于 N，再跑 `AUDIT_S960` 测小目标尺度收益。
3. 对最佳 RGB checkpoint 导出固定 val 的 conf<=0.001 TXT，用赛事重建评测器扫阈值并核对
   Ultralytics mAP；之后才生成榜单提交。
4. 完成 F001 的 NORMAL/RESIDUAL_OFF/IR_ZERO/IR_SHUFFLED 诊断和 CTRL001 对照，判断 IR
   是否真的提供信息。没有诊断增益就停止维护 F001。
5. 以最佳 RGB 为初始化重做 P3/P4/P5 可靠性融合；同步几何增强，恢复合理 mosaic/scale/
   translate，并加入 IR 黑边 mask 与 modality dropout。
6. Depth 单独重做 format-aware adapter + valid mask；在 RGB+IR 明确增益后再加入三模态，
   避免同时调试两种辅助模态。

F001 的榜单提交文件不在本次 artifact 中，因此只能核验其训练/验证和 checkpoint，不能核验
约 0.35 那次提交的具体 TXT、阈值、框统计与 SHA。
