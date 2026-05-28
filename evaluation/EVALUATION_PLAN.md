# 视频高光剪辑 — 评测方案

## 一、评测目标

评估 LAS 端到端视频高光剪辑 Pipeline 的剪辑质量：给定一段长视频 + 自然语言剪辑指令，LAS 返回的高光片段是否准确、完整、精彩。

## 二、评测指标体系

### 2.1 量化评测（tIoU 时间轴匹配）

将 Pipeline 输出的片段与人工标注的 ground truth 做时间轴匹配。

| 指标 | 公式 | 判断标准 | 说明 |
|------|------|----------|------|
| **IoU** | 交集时长 / 并集时长 | ≥0.8 优秀 / ≥0.5 合格 / <0.5 不合格 | 单片段与 GT 的重叠度 |
| **Precision** | hit_count / len(predicted) | 越高越好 | 预测片段中命中 GT 的比例 |
| **Recall** | hit_count / len(ground_truth) | 越高越好 | GT 片段中被找到的比例 |
| **F1** | 2 × P × R / (P + R) | 越高越好，核心指标 | Precision 和 Recall 的调和均值 |
| **Hit Rate @1** | Top-1 是否命中任意 GT | 越高越好 | 最优片段是否命中 |
| **Hit Rate @3** | Top-3 命中率 | 越高越好 | 前三片段覆盖能力 |
| **MAE** | 命中片段起止时间平均偏差（秒） | 越小越好 | IoU 之外的精细时间偏差 |

**匹配规则**：贪心匹配，每个预测片段找 IoU 最大的未使用 GT，IoU ≥ 0.5 算命中。

### 2.2 主观评测（LLM Judge）

LLM 对剪辑结果进行四维打分（1-10 分），支持纯文本模式和视频观看模式。

| 维度 | 评估内容 | 判断标准 |
|------|----------|----------|
| **节奏感** | 片段衔接是否流畅，节奏是否符合风格要求 | 1-10 分 |
| **内容完整性** | 每个高光片段是否完整表达关键内容，有无截断感 | 1-10 分 |
| **精彩程度** | 选中的片段是否真正精彩 | 1-10 分 |
| **指令契合度** | 剪辑结果是否符合用户的剪辑目标和要求 | 1-10 分 |

### 2.3 加权总分

```
加权总分 = F1 × 0.5 + (LLM Judge 均分 / 10.0) × 0.5
```

若 LLM Judge 不可用，总分仅基于 F1。

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
| **tIoU 分布** | 优秀/合格/不合格三档分布 |
| **F1 按类别** | 按视频类型分组（sports/news/vlog...） |
| **F1 按难度** | 按 easy/medium/hard 分组 |
| **F1 按来源** | 按 local/remote 分组 |
| **Token 消耗** | 总 Token / 每分钟视频 Token |
| **处理倍速** | 处理耗时 / 视频时长 |
| **内存峰值** | 单 case 最大内存占用 |

## 三、评测流程

```
1. TestCaseLoader 加载用例（cases.yaml + instruction.json + ground_truth.json）
2. EvalRunner 遍历用例，调用 Pipeline.run(source, description, skip_edit=False)
3. 从 PipelineResult.edit.segments 提取 predicted 片段
4. HighlightEvaluator 做 tIoU 匹配，计算各指标
5. LLMJudge 并行做主观打分
6. compute_weighted_score() 计算加权总分
7. ReportGenerator 生成文本报告 + JSON + 图表
```

## 四、评测用例集

### 4.1 来源

| 数据集 | 数量 | 类型 | 说明 |
|--------|------|------|------|
| open_data（SumMe） | 35 组 | local | 通用精彩片段标注，涵盖运动/新闻/生活等 |
| self-built_data | 10 组 | remote | 自建 URL 用例，针对特定剪辑场景 |

### 4.2 用例结构

每个用例包含：

```
case_XXX/
├── video.mp4              # 原始视频
├── instruction.json       # 剪辑指令 {"prompt": "帮我把精彩片段剪成60秒集锦"}
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

- **视频类型**：sports / news / vlog / entertainment / education
- **难度**：easy（单场景少切换）/ medium（多场景中等变化）/ hard（快速切换复杂场景）
- **来源**：local（本地文件）/ remote（URL 下载）

## 五、评测配置

```python
from evaluation.runner import EvalRunner, EvalRunConfig

config = EvalRunConfig(
    test_cases_root="evaluation/test_cases",
    output_dir="reports",
    iou_threshold=0.5,       # IoU 命中阈值
    skip_llm_judge=False,    # 是否跳过 LLM Judge
    skip_edit=False,         # 是否跳过 LAS 剪辑（必须 False）
    judge_weight=0.5,        # LLM Judge 在总分中的权重
    judge_max_retries=3,     # LLM Judge 失败重试次数
    concurrency=1,           # 并发数（>1 开启压测模式）
)
runner = EvalRunner(config)
eval_report, judge_report, report_text = runner.run()
print(report_text)
```

## 六、输出产物

| 产物 | 格式 | 说明 |
|------|------|------|
| 文本报告 | `reports/report.txt` | 完整评测结果，含所有指标和分组统计 |
| JSON 报告 | `reports/report.json` | 结构化数据，含每个 case 的详细分数 |
| 可视化图表 | `reports/charts.png` | F1 按类别/难度/来源的柱状图 + LLM Judge 评分图 |

## 七、典型分析模板（待实现）

评测完成后，针对以下维度做 case-level 分析：

1. **Top-3 最佳案例**：分析为什么 LAS 能精准命中，什么类型的视频/指令效果最好
2. **Top-3 最差案例**：分析失败原因——是 GT 标注偏差、LAS 理解错误、还是视频本身不适合
3. **指令敏感性**：同一视频不同指令（"精彩集锦" vs "进球片段"）的结果差异
4. **视频类型对比**：运动/新闻/Vlog 等不同类别的 F1 差异及原因
