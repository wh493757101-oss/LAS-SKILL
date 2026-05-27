# 视频高光检测 — 评测方案行业调研

> 调研目标：对比行业主流评测方案与我们的评测方案，识别差距和可补充的维度。

---

## 一、行业评测方案全景

### 1.1 两大评测范式

视频高光检测/摘要领域存在两套并行的评测范式：

| 范式 | 核心指标 | 适用场景 | 代表基准 |
|------|---------|---------|---------|
| **片段重叠式** | Precision / Recall / F1（基于时间片段与 GT 的重叠） | 视频摘要（输出 keyshot 集合） | SumMe, TVSum |
| **排序相关式** | Kendall's τ / Spearman's ρ（预测分数排序与人工标注排序的相关性） | 高光检测（输出帧级 saliency score） | TVSum |
| **检测式** | mAP + HIT@1（将高光视为检测任务，IoU 匹配） | 时刻检索 + 高光联合检测 | QVHighlights |

### 1.2 各基准的具体评测方案

#### TVSum（2015，50 视频）

评测的是**帧级重要性分数的排序质量**，而非片段裁剪精度：

| 指标 | 计算方式 | 说明 |
|------|---------|------|
| **Kendall's τ** | 预测帧分数排序 vs 人工标注排序的秩相关性 | 主流指标，范围 [-1, 1] |
| **Spearman's ρ** | 同上，单调关系度量 | 比 τ 对异常值更敏感 |
| **Top-5 mAP** | 取 top 5 片段，计算与 GT 的 mAP | 少数工作使用 |
| **F1 (keyshot)** | 先 KTS 分割 → 0/1 knapsack 选段 → 与人工摘要比 F1 | 传统做法，但 Otani 2019 证明随机也能高分 |

**关键问题**：Otani et al. (2019) 发现 TVSum/SumMe 上随机摘要的 F1 与 SOTA 相当，甚至超过人类。此后 Kendall's τ / Spearman's ρ 成为更受认可的指标。

#### QVHighlights（2021，12,562 视频）

目前最完善的评测方案，**同时评测时刻检索 (MR) 和高光检测 (HD)**：

**时刻检索 (MR) 指标**：
| 指标 | 计算方式 | 阈值 |
|------|---------|------|
| **R@1 (Recall@1)** | Top-1 预测时刻与任一 GT 时刻 IoU ≥ θ 的比例 | θ=0.5, 0.7 |
| **mAP@θ** | 多 IoU 阈值下的平均精度 | θ=0.5, 0.75 |
| **Avg mAP** | 多阈值平均 | [0.5:0.05:0.95] |

**高光检测 (HD) 指标**：
| 指标 | 计算方式 | 说明 |
|------|---------|------|
| **HD mAP** | 每 2 秒 clip 的 saliency score 与 GT 的 mAP | GT 来自 3 个标注者的 Likert 5 级评分 |
| **HIT@1** | 最高分 clip 是否被标注为 "Very Good" | 对 3 个标注者分别计算取平均 |

**标注质量**：QVHighlights 的 3 标注者间 IoU > 0.9（90% 的 query），标注质量很高。

#### Mr. HiSum（2023，31,892 视频）

用 YouTube "Most Replayed" 数据作为大规模弱监督 GT：

| 指标 | 计算方式 |
|------|---------|
| **mAP@50** | top 50% shot 的 mAP |
| **mAP@15** | top 15% shot 的 mAP |
| **F1** | 基于 KTS 分割 + knapsack 选段的 F1 |

### 1.3 其他评测维度（学术论文中常见）

| 维度 | 指标 | 使用场景 |
|------|------|---------|
| **用户研究** | 人工打分（informativeness, enjoyability, 偏好选择） | 早期工作 + 重要论文的补充验证 |
| **Ablation Study** | 控制变量对比各模块贡献 | 几乎所有论文标配 |
| **跨数据集泛化** | 在 A 数据集训练，B 数据集测试（zero-shot / fine-tuned） | 验证泛化能力 |
| **效率指标** | 参数量、GFLOPs、推理时间 | 近年越来越受重视 |
| **多样性/冗余度** | 摘要片段之间的语义相似度 | 少数工作（如 VISIOCITY） |
| **连续性** | 摘要片段时间连续性评分 | 少数工作 |
| **CLIP-based 语义评分** | $F_{CLIP}$：用 CLIP 相似度替代精确帧匹配 | V2Xum-LLM (2024) |

---

## 二、我们的评测方案 vs 行业

### 2.1 我们的方案

| 维度 | 指标 | 对标 |
|------|------|------|
| **IoU 评测** | Precision, Recall, F1, tIoU | 片段重叠式 (SumMe/TVSum 的 F1 变体) |
| **Hit Rate** | Hit Rate @1, @3 | QVHighlights 的 HIT@1 |
| **MAE** | 预测与 GT 边界的时间偏差 | 我们独创，行业无直接对应 |
| **tIoU 分布** | excellent(≥0.8) / qualified(≥0.5) / unqualified(<0.5) | 我们独创的分层统计 |
| **异常率** | error case / total case | 我们独创，行业无此维度 |
| **Token 效率** | total/prompt/completion tokens, tokens/min | 行业近年关注，但无标准指标 |
| **LLM Judge** | 节奏感/完整性/精彩度/契合度 (1-5) | 类似用户研究，但用 LLM 替代人工 |

### 2.2 对比分析

| 维度 | 行业有 | 我们有 | 差距/建议 |
|------|--------|--------|-----------|
| **Precision/Recall/F1** | ✓ | ✓ | 一致 |
| **tIoU** | ✓ (mAP@多阈值) | ✓ (单一均值) | **可补充**：行业用多 IoU 阈值 mAP (0.5, 0.75, [0.5:0.95])，我们只算了一个均值 |
| **HIT@K** | ✓ (HIT@1) | ✓ (@1, @3) | 我们更细（多了 @3） |
| **MAE** | ✗ | ✓ | 我们的差异化指标 |
| **tIoU 分布** | ✗ | ✓ | 我们的差异化指标 |
| **异常率** | ✗ | ✓ | 我们的差异化指标 |
| **Token 效率** | 少数工作有 GFLOPs | ✓ | 我们的差异化指标 |
| **LLM Judge** | 少数工作有用户研究 | ✓ | 我们的差异化指标 |
| **Kendall's τ / Spearman's ρ** | ✓ (TVSum 标配) | ✗ | **缺失**：排序质量指标，适合评测 saliency score 的排序能力 |
| **跨类别/跨难度分组** | ✓ (per-category) | ✓ | 一致 |
| **多标注者一致性** | ✓ (QVHighlights 3 标注者取平均) | ✗ | 我们的 GT 是单人标注，缺少标注质量验证 |
| **Ablation Study** | ✓ (标配) | ✗ | **缺失**：无法量化各模块的贡献 |
| **跨数据集泛化** | ✓ | ✗ | 受限于自建数据集 |
| **效率指标** | GFLOPs, 推理时间 | Token 效率 | 可补充推理时间 |
| **CLIP 语义评分** | 新兴方向 | ✗ | 可选：用 CLIP 评估预测片段与 instruction 的语义匹配度 |

---

## 三、建议补充的评测维度

### 3.1 高优先级（直接对标行业标准）

#### (1) 多 IoU 阈值 mAP

当前我们只算了一个平均 tIoU。行业标准做法是报告多个阈值下的 mAP：

```
mAP@0.5  — 宽松匹配
mAP@0.75 — 严格匹配
Avg mAP  — [0.5:0.05:0.95] 平均
```

**改动**：在 `evaluator.py` 中增加多阈值 mAP 计算，`report.py` 中增加对应展示。

#### (2) Kendall's τ / Spearman's ρ（排序相关性）

这是 TVSum 评测的标配。衡量预测的 segment score 排序与 GT score 排序之间的相关性。

**改动**：在 `evaluator.py` 中增加 `compute_rank_correlation()` 方法，对 predicted segments 的 score 排序与 GT 的 score 排序计算 τ 和 ρ。

### 3.2 中优先级（增强评测深度）

#### (3) Ablation 对比框架

评测框架支持对比不同配置：
- Full pipeline vs 仅 FFmpeg 降级
- Full pipeline vs 纯规则引擎
- 有 ASR vs 无 ASR
- 有后处理 vs 无后处理

**改动**：在 `runner.py` 中增加 `ablation_configs` 参数，一次运行输出多组对比结果。

#### (4) 推理时间统计

补充端到端推理时间（视频预处理 / 模型推理 / 后处理 / 剪辑各阶段耗时）。

**改动**：在 `PipelineResult` 或 `CostStats` 中增加 `timing` 字段。

### 3.3 低优先级（锦上添花）

#### (5) CLIP 语义匹配度

用 CLIP 计算预测高光片段与用户 instruction 的语义相似度，作为对 LLM Judge 的补充。

#### (6) 多样性/冗余度

计算预测片段之间的语义相似度，避免输出重复内容。

---

## 四、总结

| 优先级 | 补充项 | 改动量 | 价值 |
|--------|--------|--------|------|
| **高** | 多 IoU 阈值 mAP | 小（evaluator + report） | 对标 QVHighlights 标准 |
| **高** | Kendall's τ / Spearman's ρ | 小（evaluator + report） | 对标 TVSum 标准 |
| **中** | Ablation 对比框架 | 中（runner + report） | 量化各模块贡献 |
| **中** | 推理时间统计 | 小（main + evaluator） | 实际部署参考 |
| **低** | CLIP 语义匹配度 | 中（新增依赖） | 补充 LLM Judge |
| **低** | 多样性/冗余度 | 小 | 摘要质量评估 |
