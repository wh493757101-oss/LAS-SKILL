# 视频高光剪辑 — 评测方案

## 一、评测目标

评估多模态大模型 + FFmpeg 视频高光剪辑 Pipeline 的剪辑质量：给定一段长视频 + 自然语言剪辑指令，多模态模型识别的高光片段是否准确、完整、精彩。

评测体系分三层：**tIoU 量化评测**（客观，有 GT 标注）→ **双 LLM Judge**（主观，片段级 + 集锦级）→ **加权融合**。

---

## 二、评测指标体系

### 2.1 量化评测（tIoU 时间轴匹配）

将 Pipeline 输出的片段与人工标注的 ground truth 做时间轴匹配。这是唯一能客观回答"检出率"的评测层。

| 指标 | 公式 | 判断标准 | 说明 |
|------|------|----------|------|
| **IoU** | 交集时长 / 并集时长 | ≥0.8 优秀 / ≥0.5 合格 / <0.5 不合格 | 单片段与 GT 的重叠度 |
| **Precision** | hit_count / len(predicted) | 越高越好 | 预测片段中命中 GT 的比例 |
| **Recall** | hit_count / len(ground_truth) | 越高越好 | GT 片段中被找到的比例 |
| **F1** | 2 × P × R / (P + R) | 越高越好，核心指标 | Precision 和 Recall 的调和均值 |
| **Hit Rate @1** | Top-1 是否命中任意 GT | 越高越好 | 最优片段是否命中 |
| **Hit Rate @3** | Top-3 命中率 | 越高越好 | 前三片段覆盖能力 |
| **MAE** | 命中片段起止时间平均偏差（秒） | 越小越好 | IoU 之外的精细时间偏差 |
| **mAP@0.5** | 单阈值 Average Precision | 越高越好 | 对标 QVHighlights，IoU≥0.5 匹配 |
| **mAP@0.75** | 严格阈值 Average Precision | 越高越好 | 对标 QVHighlights，IoU≥0.75 匹配 |
| **Avg mAP** | [0.5:0.05:0.95] 10 个阈值均值 | 越高越好 | 对标 QVHighlights 多阈值标准 |
| **Kendall's τ** | 预测 score 排序 vs GT score 排序的秩相关性 | [-1, 1]，越高越好 | 对标 TVSum 标准，衡量排序质量 |
| **Spearman's ρ** | 同上，单调关系度量 | [-1, 1]，越高越好 | 对标 TVSum 标准，对异常值更敏感 |

**匹配规则**：贪心匹配，每个预测片段找 IoU 最大的未使用 GT，IoU ≥ 0.5 算命中。

### 2.2 主观评测（双 LLM Judge）

LLM Judge 拆分为两个独立 Judge，**任务边界清晰，各司其职**：

#### 2.2.1 Segment Judge（片段质量评测）— 权重 25%

逐个观看多模态模型识别的每个高光片段视频，判断片段本身的质量。**不评测"检出率"**（看不到原始视频，无法判断遗漏了什么）。

| 维度 | 评估内容 | 评分范围 |
|------|----------|----------|
| **内容完整性** | 每个片段是否完整保留了关键动作/事件，有无截断 | 1-10 分 |
| **片段质量** | 画面质量、内容精彩程度、是否混入空镜头/静止画面/重复内容 | 1-10 分 |
| **指令契合度** | 每个片段是否符合剪辑目标和风格要求 | 1-10 分 |

#### 2.2.2 Video Judge（集锦质量评测）— 权重 25%

观看拼接后的完整集锦视频（含画面和音频），判断整体观感质量。**不评测"检出率"和"冗余控制"**（看不到原始视频，无法判断遗漏/多余）。

| 维度 | 评估内容 | 评分范围 |
|------|----------|----------|
| **节奏感** | 整体剪辑节奏是否流畅、符合风格要求 | 1-10 分 |
| **转场质量** | 片段间过渡是否自然，有无黑屏/卡顿/跳帧 | 1-10 分 |
| **音画同步** | 音频与画面是否同步，BGM 是否匹配 | 1-10 分 |
| **内容完整性** | 集锦中各片段是否有截断，关键内容是否完整 | 1-10 分 |
| **指令契合度** | 集锦整体是否符合剪辑目标和风格要求 | 1-10 分 |

#### 2.2.3 设计原则

- **信息充分性**：每个 Judge 只评测自己能看到的、信息充分的东西
- **互补不重叠**：Segment Judge 管"每个片段好不好"，Video Judge 管"拼在一起好不好"
- **检出率归量化**："核心高光检出率"和"遗漏了什么"由 tIoU 量化评测覆盖（有 GT 标注，客观准确）

### 2.3 加权总分

```
加权总分 = tIoU F1 × 0.5 + Segment Judge 归一化分 × 0.25 + Video Judge 归一化分 × 0.25
```

其中 Segment Judge 归一化分 = segment_average / 10.0，Video Judge 归一化分 = video_average / 10.0。

**降级策略**：
- 两个 Judge 都不可用 → 总分仅基于 F1（纯量化）
- 仅一个 Judge 可用 → 该 Judge 占满 50% 权重
- 向后兼容：旧版 JudgeScore 路径仍可用

### 2.4 微平均指标

为解决宏平均（每个 case 等权）受极端值影响的问题，同时计算微平均：

| 指标 | 公式 | 说明 |
|------|------|------|
| **微平均 Precision** | sum(hit_count) / sum(len(predicted)) | 全局命中率，片段多的 case 权重更大 |
| **微平均 Recall** | sum(hit_count) / sum(len(GT)) | 全局召回率 |
| **微平均 F1** | 2 × mP × mR / (mP + mR) | 微平均的调和均值 |

### 2.5 片段质量指标

| 指标 | 公式 | 判断标准 |
|------|------|----------|
| **片段数偏差率** | \|len(pred) - len(GT)\| / len(GT) | 越接近 0 越好 |
| **集锦时长占比** | sum(pred 时长) / 视频总时长 | 合理范围 5%-30% |
| **指令时长契合度** | 1.0 - \|实际时长 - 目标时长\| / 目标时长 | 1.0 完全契合，clamp 到 [0,1] |

### 2.6 辅助指标

| 指标 | 说明 |
|------|------|
| **异常率** | 执行失败 case 占比 |
| **tIoU 分布** | 优秀(≥0.8)/合格(≥0.5)/不合格(<0.5) 三档分布 |
| **F1 按类别** | 按视频类型分组（sports/news/vlog/entertainment/education/outdoor/gaming） |
| **F1 按难度** | 按 easy/medium/hard 分组 |
| **F1 按来源** | 按 local/remote 分组 |
| **Token 消耗** | 总 Token / Prompt Token / Completion Token / 每分钟视频 Token |
| **API 调用统计** | API 调用次数 / 重试次数 |
| **阶段耗时** | 视频获取 / 高光检测 / FFmpeg 拼接 各阶段平均耗时 |
| **处理倍速** | 处理耗时 / 视频时长 |
| **内存峰值** | 单 case 最大/平均内存占用 |
| **预估费用** | 总费用 / 平均每 case 费用（元） |
| **并发吞吐量** | 并发压测模式下的 case/s |

---

## 三、评测流程

```
1. TestCaseLoader 加载用例（cases.yaml + instruction.json + ground_truth.json）
2. EvalRunner 遍历用例，调用 Pipeline.run(source, description, skip_edit=False)
3. 从 PipelineResult.edit.segments 提取 predicted 片段（含 clip_url）
4. 并行执行：
   a. HighlightEvaluator 做 tIoU 匹配，计算所有量化指标
   b. LLMJudge 并行运行 Segment Judge + Video Judge
5. compute_weighted_score() 计算三层加权总分
6. ReportGenerator 生成文本报告 + JSON + 图表，可选上传 TOS
```

---

## 四、评测用例集

### 4.1 来源

| 数据集 | 数量 | 类型 | 说明 |
|--------|------|------|------|
| open_data（SumMe） | 35 组 | local | 通用精彩片段标注，涵盖运动/新闻/生活等 |
| self-built_data | 10 组 | remote | 自建 URL 用例，针对特定剪辑场景 |

### 4.2 用例结构

```
case_XXX/
├── video.mp4              # 原始视频
├── instruction.json       # 剪辑指令 {"prompt": "帮我把精彩片段剪成60秒集锦", "style": "快节奏", "core_highlight_definition": "进球、助攻、精彩扑救"}
└── ground_truth.json      # 人工标注 {"highlights": [{"start_time": 10.0, "end_time": 25.0, "label": "精彩动作", "score": 0.8}]}
```

在 `cases.yaml` 中注册：

```yaml
cases:
  - id: case_001
    category: sports
    difficulty: medium
    video_file: video.mp4
```

### 4.3 用例维度

- **视频类型**：sports / news / vlog / entertainment / education / outdoor / gaming
- **难度**：easy（单场景少切换）/ medium（多场景中等变化）/ hard（快速切换复杂场景）
- **来源**：local（本地文件）/ remote（URL 下载）

---

## 五、评测配置

```python
from evaluation.runner import EvalRunner, EvalRunConfig

config = EvalRunConfig(
    test_cases_root="evaluation/test_cases",
    output_dir="reports",
    iou_threshold=0.5,       # IoU 命中阈值
    skip_llm_judge=False,    # 是否跳过 LLM Judge（跳过则总分仅基于 F1）
    skip_edit=False,         # 是否跳过 FFmpeg 剪辑（必须 False）
    judge_weight=0.5,        # LLM Judge 在总分中的权重
    judge_max_retries=3,     # LLM Judge 失败重试次数
    concurrency=1,           # 并发数（>1 开启压测模式）
)
runner = EvalRunner(config)
eval_report, judge_report, report_text = runner.run()
print(report_text)
```

---

## 六、输出产物

| 产物 | 格式 | 说明 |
|------|------|------|
| 文本报告 | `reports/report.txt` | 完整评测结果，含量化指标 + 双 Judge 评分 + 加权总分 + 分组统计 |
| JSON 报告 | `reports/report.json` | 结构化数据，含 `iou_eval` / `segment_judge` / `video_judge` / `llm_judge` / `weighted_score` |
| 可视化图表 | `reports/charts.png` | 4 张子图：F1 按类别/难度/来源 + Video Judge 维度柱状图 |

---

## 七、典型分析模板（待实现）

评测完成后，针对以下维度做 case-level 分析：

1. **Top-3 最佳案例**：分析为什么多模态模型能精准命中，什么类型的视频/指令效果最好
2. **Top-3 最差案例**：分析失败原因——是 GT 标注偏差、多模态模型理解错误、还是视频本身不适合
3. **指令敏感性**：同一视频不同指令（"精彩集锦" vs "进球片段"）的结果差异
4. **视频类型对比**：运动/新闻/Vlog 等不同类别的 F1 差异及原因
5. **双 Judge 分歧分析**：Segment Judge 高分但 Video Judge 低分的 case（片段好但拼得差），反之亦然
