# 视频高光剪辑 Skill

基于火山引擎 LAS 端到端算子，输入原始长视频 + 剪辑指令，自动识别高光片段并输出集锦视频。

## 核心能力

1. **端到端高光识别+剪辑** — LAS 算子同时完成视频理解和高光剪辑，一步到位
2. **结构化输出** — 输出集锦视频 + 时间戳 + 置信度评分 + 标签
3. **多源输入** — 支持本地文件、URL、TOS 路径

## 输入输出

| 输入 | 说明 | 必需 |
|------|------|------|
| 原始视频 | 本地文件路径 / URL / TOS 路径 | 是 |
| 剪辑描述 | 如"精彩集锦""进球片段""高能时刻" | 否 |

| 输出 | 说明 |
|------|------|
| 高光视频 | 剪辑后的集锦视频（LAS 云端输出 URL） |
| 时间戳说明 | 每个片段的起止时间、置信度评分、标签 |
| JSON 导出 | 结构化片段数据，含时间戳和评分 |

## 架构

```
用户输入（视频 + 描述）
  → VideoFetcher（下载/校验/转码）
    → VideoEditor.edit_e2e（LAS 端到端识别+剪辑）
      → 输出（集锦视频 + 片段列表 + JSON）
```

LAS 失败直接报错，不降级。

## 安装

```bash
git clone https://github.com/wh493757101-oss/LAS-SKILL.git
cd LAS-SKILL
pip install -e ".[dev]"
```

外部工具：
- **FFmpeg** — 仅用于视频转码（非 mp4 格式时），非剪辑用途
- **yt-dlp** — 用于 URL 视频下载（`pip install yt-dlp`）

## 环境变量

```bash
# ========== LAS 云端剪辑（必需）==========
export LAS_API_KEY="your-las-api-key"          # LAS 算子 API Key
export LAS_OPERATOR_ID="las_video_edit"        # LAS 算子 ID
export TOS_OUTPUT_PATH="tos://bucket/output/"  # LAS 剪辑输出路径（tos:// 目录）

# ========== TOS 对象存储（本地文件上传时必需）==========
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
│   ├── video_editor.py         # LAS 端到端剪辑
│   ├── ark_client.py           # Ark API 封装（文件上传/LLM Judge）
│   ├── las_client.py           # LAS 算子 API 封装
│   ├── highlight_detector.py   # 旧版高光检测（保留，未使用）
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
│   └── verify_dual_path.py     # API 连通性验证
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

### 评测指标

| 指标 | 说明 |
|------|------|
| IoU / F1 | 片段时间戳匹配精度 |
| Hit Rate @1/@3 | Top-K 片段命中率 |
| MAE | 平均时间偏差（秒） |
| LLM Judge | 节奏感/完整性/精彩度/指令契合度（1-5 分） |

### 添加测试用例

1. 在 `cases.yaml` 中注册用例（id / category / difficulty / instruction）
2. 创建 `case_XXX/` 目录，放入视频文件
3. 编写 `instruction.json`（含 `prompt` 字段）
4. 编写 `ground_truth.json`（含 `highlights` 数组）

```json
// instruction.json
{"prompt": "帮我把精彩片段剪成60秒集锦，节奏要快"}

// ground_truth.json
{
  "highlights": [
    {"start_time": 10.0, "end_time": 25.0, "label": "精彩动作", "score": 0.8}
  ]
}
```

## 错误处理

Pipeline 采用"快速失败"策略：LAS 不可用时直接返回错误，不降级到本地剪辑。

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
| `LAS_API_KEY 未设置` | 环境变量缺失 | `export LAS_API_KEY=your-key` |
| `TOS 配置不完整` | TOS 环境变量缺失 | 设置 `TOS_ACCESS_KEY`/`TOS_SECRET_KEY` |
| `视频下载失败` | URL 不可访问或 yt-dlp 版本过旧 | 检查 URL，升级 yt-dlp |
| `无法打开视频` | 文件损坏或格式不支持 | 检查文件完整性 |
| `视频转码失败` | FFmpeg 不可用或磁盘空间不足 | 安装 FFmpeg，清理磁盘 |
| `LAS 任务超时` | 视频过大或 LAS 服务繁忙 | 重试或缩短视频 |
| `LAS 未返回 task_id` | API Key 无效或算子不存在 | 检查 LAS_API_KEY 和算子配置 |
