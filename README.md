# 视频高光剪辑 Skill

基于多模态大模型识别高光 + FFmpeg 流拷贝拼接，输入原始长视频 + 剪辑指令，自动识别高光片段并输出集锦视频。

## 核心能力

1. **多模态高光识别** — 多模态大模型根据剪辑指令识别高光片段
2. **FFmpeg 无损拼接** — 流拷贝模式拼接，零成本、秒级完成
3. **结构化输出** — 输出集锦视频 + 时间戳 + 置信度评分 + 标签
4. **多源输入** — 支持本地文件、URL、TOS 路径

## 输入输出

| 输入 | 说明 | 必需 |
|------|------|------|
| 原始视频 | 本地文件路径 / URL / TOS 路径 | 是 |
| 剪辑描述 | 如"精彩集锦""进球片段""高能时刻" | 否 |

| 输出 | 说明 |
|------|------|
| 高光视频 | 剪辑后的集锦视频（本地 mp4 文件） |
| 时间戳说明 | 每个片段的起止时间、置信度评分、标签 |
| JSON 导出 | 结构化片段数据，含时间戳和评分 |

## 架构

```
用户输入（视频 + 描述）
  → VideoFetcher（下载/校验/转码）
    → HighlightDetector（多模态大模型识别高光）
      → VideoEditor.edit_with_ffmpeg（FFmpeg 流拷贝拼接）
        → 输出（集锦视频 + 片段列表 + JSON）
```

多模态识别失败直接报错，不降级。

## 安装

```bash
git clone https://github.com/wh493757101-oss/Video-Highlight-Skill.git
cd Video-Highlight-Skill
pip install -e ".[dev]"
```

外部工具：
- **FFmpeg** — 用于视频转码和流拷贝拼接剪辑
- **yt-dlp** — 用于 URL 视频下载（`pip install yt-dlp`）

## 环境变量

```bash
# ========== 多模态高光识别（必需）==========
export ARK_HIGHLIGHT_API_KEY="your-ark-key"    # 火山引擎 Ark API Key
export ARK_HIGHLIGHT_MODEL="your-model"        # 多模态模型（如 doubao-seed-2-0-pro）

# ========== TOS 对象存储（TOS 路径输入时必需）==========
export TOS_ACCESS_KEY="your-access-key"        # TOS Access Key
export TOS_SECRET_KEY="your-secret-key"        # TOS Secret Key
export TOS_ENDPOINT="tos-cn-guangzhou.volces.com"  # TOS Endpoint

# ========== LLM Judge 评测（可选）==========
export ARK_JUDGE_API_KEY="your-judge-key"      # LLM Judge API Key
export ARK_JUDGE_MODEL="your-model"            # LLM Judge 模型（如 qwen3.5-omni-plus）
export ARK_JUDGE_BASE_URL="your-base-url"      # LLM Judge Base URL
```

## 使用示例

```python
from src.main import VideoHighlightPipeline

pipeline = VideoHighlightPipeline()

# 本地视频
result = pipeline.run_from_path(
    video_path="/path/to/video.mp4",
    description="剪辑最精彩的 60 秒",
)

# URL 视频
result = pipeline.run_from_url(
    url="https://example.com/video.mp4",
    description="进球集锦",
)

# TOS 视频
from src.video_fetcher import TosSource
result = pipeline.run(
    TosSource("tos://my-bucket/path/to/video.mp4"),
    description="精彩集锦",
)

# 查看结果
print(pipeline.format_result(result))
print(pipeline.export_json(result))
```

## 项目结构

```
video-highlight-skill/
├── SKILL.md                    # Skill 定义
├── README.md
├── pyproject.toml
├── src/
│   ├── main.py                 # Pipeline 主入口
│   ├── video_fetcher.py        # 视频获取与预处理（校验/转码）
│   ├── video_editor.py         # FFmpeg 智能拼接（流拷贝/重编码自适应 + 音频归一化 + 转场）
│   ├── highlight_detector.py   # 多模态大模型高光识别
│   ├── ark_client.py           # Ark API 封装（多模态识别/文件上传/LLM Judge）
│   ├── cost_estimator.py       # Ark token 费用估算
│   └── rule_engine.py          # 规则引擎（保留，未使用）
├── evaluation/
│   ├── evaluator.py            # tIoU 自动评测
│   ├── llm_judge.py            # LLM-as-Judge 多维度打分
│   ├── report.py               # 可视化报告生成
│   ├── runner.py               # 评测流程编排
│   └── test_cases/
│       ├── open_data/          # 35 组本地视频用例（SumMe 数据集）
│       └── self-built_data/    # 10 组远程视频用例
├── scripts/
│   ├── verify_e2e.py           # 端到端验证脚本
│   ├── verify_e2e_strict.py    # 严格端到端验证
│   ├── verify_connectivity.py   # API 连通性验证
└── tests/
    └── test_*.py
```

## 运行测试

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov=evaluation --cov-report=term-missing
```

## 运行评测

```python
from evaluation.runner import EvalRunner, EvalRunConfig

config = EvalRunConfig(
    test_cases_root="evaluation/test_cases",
    output_dir="reports",
    skip_edit=True,
)
runner = EvalRunner(config)
eval_report, judge_report, report_text = runner.run()
print(report_text)
```

### 评测体系

三层评测架构：**tIoU 量化评测（50%）→ 双 LLM Judge（50%）→ 加权融合**

#### 量化评测（tIoU 时间轴匹配）

| 指标 | 说明 | 对标 |
|------|------|------|
| Precision / Recall / F1 | 宏平均 + 微平均，贪心 IoU 匹配 | 核心指标 |
| Hit Rate @1 / @3 | Top-K 片段命中率 | — |
| MAE | 命中片段的时间偏差（秒） | — |
| mAP@0.5 / mAP@0.75 / Avg mAP | 10 个 IoU 阈值 [0.5:0.05:0.95] | QVHighlights |
| Kendall's τ / Spearman's ρ | 预测排序 vs GT 排序相关性 | TVSum |

#### 双 LLM Judge（主观评测）

| Judge | 输入 | 评测维度 | 权重 |
|-------|------|----------|------|
| Segment Judge | 逐个高光片段视频 | 内容完整性、片段质量、指令契合度 | 25% |
| Video Judge | 拼接后的集锦视频 | 节奏感、转场质量、音画同步、内容完整性、指令契合度 | 25% |

#### 加权总分

```
总分 = tIoU F1 × 0.5 + Segment Judge × 0.25 + Video Judge × 0.25
```

### 添加测试用例

1. 在 `cases.yaml` 中注册用例（id / category / difficulty / instruction）
2. 创建 `case_XXX/` 目录，放入视频文件
3. 编写 `instruction.json`（含 `prompt`、`style`、`core_highlight_definition` 字段）
4. 编写 `ground_truth.json`（含 `highlights` 数组）

```json
// instruction.json
{
  "prompt": "帮我把精彩片段剪成60秒集锦",
  "style": "快节奏",
  "core_highlight_definition": "进球、助攻、精彩扑救"
}

// ground_truth.json
{
  "highlights": [
    {"start_time": 10.0, "end_time": 25.0, "label": "精彩动作", "score": 0.8}
  ]
}
```

## 错误处理

Pipeline 采用"快速失败"策略：多模态识别不可用时直接返回错误，不降级。

```python
result = pipeline.run_from_path("/path/to/video.mp4", description="精彩集锦")
if result.error:
    print(f"处理失败: {result.error}")
else:
    print(pipeline.format_result(result))
```

## 故障排查

| 症状 | 可能原因 | 解决方案 |
|------|----------|----------|
| `ARK_HIGHLIGHT_API_KEY 未设置` | 环境变量缺失 | `export ARK_HIGHLIGHT_API_KEY=your-key` |
| `TOS 配置不完整` | TOS 环境变量缺失 | 设置 `TOS_ACCESS_KEY`/`TOS_SECRET_KEY` |
| `视频下载失败` | URL 不可访问或 yt-dlp 版本过旧 | 检查 URL，升级 yt-dlp |
| `无法打开视频` | 文件损坏或格式不支持 | 检查文件完整性 |
| `视频转码失败` | FFmpeg 不可用或磁盘空间不足 | 安装 FFmpeg，清理磁盘 |
| `多模态识别失败` | API Key 无效或模型不可用 | 检查 ARK_HIGHLIGHT_API_KEY 和模型配置 |
| `FFmpeg 拼接失败` | FFmpeg 不可用或磁盘空间不足 | 安装 FFmpeg，清理磁盘 |
