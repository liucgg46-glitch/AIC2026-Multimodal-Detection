# CODEX_A_PHASE2_EXPERIMENTS

负责人：成员 A

未来分支：`feature/phase2-experiments`

## 1. 阶段定位

成员 A 负责第二阶段正式单模态实验主线。第一阶段已经完成 RGB baseline pipeline、E001 RAW、E001 CLEAN、inference、submission 和 submission-format validation。

第二阶段开始时，不直接实现复杂 Fusion。必须先完成 E002 IR-only、E003 Depth-only 和统一的单模态比较，再决定 E004/E005/E006 的优先顺序。

## 2. A1：维护 EXPERIMENT_LOG

每次正式实验开始前，在 `docs/EXPERIMENT_LOG.md` 中记录：

- experiment ID；
- branch；
- commit；
- config；
- data version。

训练完成后补充：

- metrics；
- checkpoint；
- runtime；
- conclusion。

未知字段必须使用 `unavailable` 或 `pending`，不得猜测。

## 3. A2：E002 IR-only

前置条件：成员 C 的 IR preprocessing/interface PR 已完成审查并合入 `main`。

E002 必须满足：

- 使用固定 split：train = 1600，val = 400；
- 使用 `labels_clean`；
- seed = 2026；
- 尽可能保持 E001 CLEAN 的训练参数；
- 主要变量仅为 RGB → IR 输入以及实验所需的单一 IR representation；
- 在实验日志中明确记录 IR representation 和预处理开关。

## 4. A3：E003 Depth-only

前置条件：成员 C 的 Depth preprocessing/interface PR 已完成审查并合入 `main`。

E003 必须满足：

- 使用固定 split；
- 使用 `labels_clean`；
- seed = 2026；
- 尽可能保持 E001 CLEAN 的训练参数；
- 明确记录 Depth representation、PNG/JPG policy 和无效值处理；
- 一次只验证一个主要 Depth representation。

## 5. A4：单模态正式比较

统一比较：

- E001 RGB；
- E002 IR；
- E003 Depth。

至少比较：

- Precision；
- Recall；
- mAP50；
- mAP50-95；
- per-class AP；
- best epoch；
- runtime。

比较必须基于相同固定 val，不能使用 prelim_test 或 leaderboard 替代。

## 6. A5：Inference 与 Submission

成员 A 负责正式 inference 和 submission。测试集只能用于 inference 和 submission，禁止用于：

- model selection；
- 根据 hidden-test 表现调 threshold；
- manual labeling；
- pseudo-label training。

## 7. A6：Phase 2 Conclusion

第二阶段结论必须回答：

1. RGB、IR、Depth 哪个单模态整体最好；
2. 哪些类别 IR 有潜在优势；
3. 哪些类别 Depth 有潜在优势；
4. 是否值得进入 Fusion；
5. E004、E005、E006 的优先顺序。

## 8. 禁止事项

- 第二阶段一开始就实现复杂 Fusion；
- 未完成 E002/E003 就启动 E004/E005/E006；
- 修改 fixed split；
- 使用 raw labels；
- 同时修改模型结构、输入模态、imgsz、augmentation 等多个主要变量。

## 9. 完成标准

- E002 完成；
- E003 完成；
- 单模态统一对比表完成；
- `EXPERIMENT_LOG` 完整；
- 得出 Phase 3 Fusion 决策。
