# TASKS

## Phase 1：已完成

第一阶段已完成并合入 `main`：

- [x] Private Git 仓库与项目规范建立
- [x] Official training data validation
- [x] 标签合法性与类别、bbox 分布检查
- [x] Fixed 1600/400 split，seed = 2026
- [x] Clean labels 可复现生成与复检
- [x] Multimodal inspection、Depth 专项分析与最终报告
- [x] RGB training pipeline
- [x] RGB inference pipeline
- [x] Submission-format checker
- [x] E001 RAW historical baseline
- [x] E001 CLEAN canonical baseline
- [x] Official preliminary-test inference
- [x] First submission `S001`
- [x] RGB inference stability hotfix

第一阶段正式结果与历史记录见 `docs/EXPERIMENT_LOG.md`。三份第一阶段任务书继续保留：

- `CODEX_A_RGB_BASELINE.md`
- `CODEX_B_DATA_ANALYSIS.md`
- `CODEX_C_MULTIMODAL_INSPECTION.md`

Leaderboard score：`PENDING`

---

## Current Stage: Phase 2 - Single-Modality Understanding

第二阶段包含三条并行工作线。

### 成员 A：E002 / E003 Formal Experiments

- [ ] 持续维护 `docs/EXPERIMENT_LOG.md`
- [ ] 等待 C 的 IR interface 合入后执行 E002 IR-only
- [ ] 等待 C 的 Depth interface 合入后执行 E003 Depth-only
- [ ] 使用固定 1600/400 split、`labels_clean` 和 seed = 2026
- [ ] 完成 RGB / IR / Depth 统一比较表
- [ ] 给出 Phase 2 单模态结论与 Fusion 优先顺序

详细边界见 `docs/CODEX_A_PHASE2_EXPERIMENTS.md`。

### 成员 B：E001 CLEAN Error Analysis

- [x] 整理固定 val 上的 per-class metrics
- [x] 完成 confusion、FP、FN 分析
- [x] 沿用既定 tiny/small/medium/large size bin 分析性能
- [x] 整理代表性 hard cases
- [x] 分析数据属性与性能的 correlation / hypothesis
- [x] 提炼 E002/E003 和后续 Fusion 的验证重点

详细边界见 `docs/CODEX_B_PHASE2_ERROR_ANALYSIS.md`。

### 成员 C：IR / Depth Trainable Preprocessing

- [ ] 提供 deterministic IR-only trainable interface
- [ ] 提供 format-aware Depth-only trainable interface
- [ ] 明确 PNG uint16 与 JPG uint8 的不同处理策略
- [ ] 将 Depth 预处理候选实现为独立可配置开关
- [ ] 完成接口 smoke tests
- [ ] 给出 E002/E003 默认候选建议

详细边界见 `docs/CODEX_C_PHASE2_MODALITY_PREPROCESSING.md`。

### 依赖关系

```text
C 的 IR interface
→ A 才能正式执行 E002

C 的 Depth interface
→ A 才能正式执行 E003

B 的 E001 CLEAN error analysis 可与 C 并行
```

### Phase 2 Gate

以下事项全部完成后才允许进入 Fusion：

1. [x] E001 CLEAN error analysis
2. [ ] IR preprocessing review
3. [ ] Depth preprocessing review
4. [ ] E002 IR-only
5. [ ] E003 Depth-only
6. [ ] RGB / IR / Depth unified comparison
7. [ ] Phase 2 conclusion

---

## Phase 3：Simple Multimodal Baselines

Phase 2 gate 通过后，按 Phase 2 结论决定优先顺序：

- [ ] E004 RGB+IR
- [ ] E005 RGB+Depth
- [ ] E006 RGB+IR+Depth Early Fusion
- [ ] 建立完整消融表

当前不得提前启动 E004/E005/E006。

---

## Phase 4：Feature Fusion

根据 E001～E006 结果选择候选方法：

- [ ] Add
- [ ] Concat + 1x1 Conv
- [ ] Gated Fusion
- [ ] 轻量 Attention

一次实验只验证一个主要改动。

---

## Phase 5：Adaptive / Reliability Fusion

- [ ] Modality Quality
- [ ] Adaptive Weight
- [ ] Depth Invalid-region Reliability
- [ ] Low-light IR Utilization

---

## Phase 6：专项优化

根据固定 val 的实验结果选择：

- [ ] P2 Head / 小目标
- [ ] 输入分辨率
- [ ] 类别不均衡
- [ ] Depth normalization
- [ ] 模态专用增强

---

## Phase 7：Leaderboard 与复赛准备

- [ ] 记录每次正式 submission
- [ ] 以固定 val 为主要模型选择依据
- [ ] 保留 submission 对应的 commit、配置和权重说明
- [ ] 整理训练代码、推理代码和环境说明
- [ ] 整理消融实验表、模型架构图、创新点和局限性
- [ ] 准备技术报告

测试集只允许用于 inference 和 submission，leaderboard 不得替代固定 val。
