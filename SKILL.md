---
name: video-highlight
description: >
  视频高光剪辑 — 输入长视频+剪辑指令，LAS 端到端自动识别高光片段并输出集锦视频。
  触发词: 视频高光、高光剪辑、精彩集锦、视频剪辑、剪辑视频、highlight、highlight reel、video highlight。
  使用场景: 用户需要从长视频中提取精彩片段、制作集锦、高光时刻剪辑。
version: 1.0.0
metadata:
  openclaw:
    requires:
      env:
        - LAS_API_KEY
      bins:
        - ffmpeg
    primaryEnv: LAS_API_KEY
    envVars:
      - name: LAS_API_KEY
        description: "LAS 算子 API Key，用于端到端视频识别+剪辑"
        required: true
      - name: TOS_ACCESS_KEY
        description: "TOS Access Key（本地文件上传时必需）"
        required: false
      - name: TOS_SECRET_KEY
        description: "TOS Secret Key（本地文件上传时必需）"
        required: false
---

# 视频高光剪辑 Skill

基于火山引擎 LAS 端到端算子，自动识别视频中的高光片段并生成集锦视频。

## 输入输出

### 输入

| 字段 | 说明 | 必需 |
|------|------|------|
| 原始视频 | 长视频文件（本地路径 / URL / TOS 路径） | 是 |
| 剪辑描述 | 如"精彩集锦""进球片段""高能时刻""快节奏" | 否 |

### 输出

| 字段 | 说明 |
|------|------|
| 高光视频 | LAS 云端输出的集锦视频 URL |
| 片段列表 | 每个片段的起止时间、置信度评分、标签 |
| JSON 导出 | 结构化片段数据 |

## 核心能力

1. **端到端高光识别+剪辑** — LAS 算子同时完成视频理解和高光剪辑，一步到位
2. **结构化输出** — 输出集锦视频 + 时间戳 + 置信度评分 + 标签
3. **多源输入** — 支持本地文件、URL、TOS 路径

## 触发条件

当用户消息包含以下关键词时触发：
- 视频高光、高光剪辑、精彩集锦、视频剪辑、剪辑视频
- highlight、highlight reel、video highlight

## 工作流（严格按步骤执行）

复制此清单并跟踪进度：

```
执行进度：
- [ ] Step 0: 前置检查
- [ ] Step 1: 视频获取与预处理
- [ ] Step 2: LAS 端到端识别+剪辑
- [ ] Step 3: 结果呈现
```

### Step 0: 前置检查

**环境变量检查：**
- 确认 `LAS_API_KEY` 已配置
- 如输入为本地文件或 TOS 路径，确认 `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` 已配置
- 缺失时必须向用户索要

**输入来源检查：**
- 本地文件：确认文件存在，格式为常见视频格式（mp4/mov/avi 等）
- URL 链接：确认链接可访问，支持 yt-dlp 兼容平台
- TOS 路径：确认凭证有效，路径格式为 `tos://bucket/key`

**输出路径检查：**
- LAS 剪辑时 `output_tos_path` 必须为 `tos://` 前缀的目录（不能以文件名结尾）
- LAS Region 必须与 TOS Bucket 区域一致

### Step 1: 视频获取与预处理

- **本地文件**: 直接加载，非 mp4 或无法解码时通过 FFmpeg 转码
- **URL**: 使用 yt-dlp 下载（默认 600 秒超时）
- **TOS 路径**: 使用 tos SDK 下载

预处理只做：格式校验（空文件/超大文件/时长超限）、元数据提取（fps/duration/分辨率）、按需转码。不提取音频、不采样关键帧。

### Step 2: LAS 端到端识别+剪辑

- 本地文件上传到 TOS 获取 `tos://` URL（远程 URL 直接透传）
- 提交 LAS 异步任务（`las_video_edit` 算子，参数：video_url + task_description + mode）
- 轮询等待完成（默认 600 秒超时）
- 解析返回的 clips（clip_url + 时间戳 + confidence + description）
- LAS 失败直接报错，不降级

### Step 3: 结果呈现

1. **视频信息**: 原视频时长、分辨率、帧率
2. **片段列表**: 每个片段的起止时间、置信度评分、标签
3. **输出视频**: 集锦视频 URL
4. **JSON 导出**: 结构化片段数据

## 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `LAS_API_KEY` | LAS 算子 API Key | 是 |
| `TOS_ACCESS_KEY` | TOS Access Key | 否（本地文件上传时必需） |
| `TOS_SECRET_KEY` | TOS Secret Key | 否（本地文件上传时必需） |

## 错误处理

Pipeline 采用"快速失败"策略：LAS 不可用时直接返回错误（`PipelineResult.error`），不降级到本地剪辑。

## 评测体系

三层架构：**tIoU 量化评测（50%）→ 双 LLM Judge（50%）→ 加权融合**

- **量化评测** (`evaluator.py`) — tIoU 片段匹配，Precision/Recall/F1/Hit Rate/MAE/mAP/Kendall's τ
- **Segment Judge** (`llm_judge.py`) — 逐片段评测：内容完整性、片段质量、指令契合度（权重 25%）
- **Video Judge** (`llm_judge.py`) — 集锦整体评测：节奏感、转场质量、音画同步、内容完整性、指令契合度（权重 25%）
- **评测编排** (`runner.py`) — 自动加载用例 → 运行 Pipeline → 并行评测
- **报告生成** (`report.py`) — 文本报告 + JSON 导出 + 可视化图表

测试用例基于 SumMe 数据集（35 组本地）和自建 URL 用例（10 组远程）。

## 审查标准

**运维层面：**
- [ ] 环境变量是否正确配置（LAS_API_KEY 已设置）
- [ ] 输入文件是否成功加载（非空、可解码）
- [ ] 输出结果是否正确呈现（视频路径 + 片段列表 + JSON）

**业务层面：**
- [ ] 高光片段时长合理（建议 2-10 秒/段）
- [ ] 集锦总时长不超过原视频的 30%
- [ ] 输出视频格式为 mp4（H.264 + AAC）

## Gotchas

- **URL 下载**: 使用 yt-dlp，支持主流平台，默认 600 秒超时
- **TOS 上传**: 本地文件需要先上传到 TOS 才能被 LAS 处理，需要 `TOS_ACCESS_KEY`/`TOS_SECRET_KEY`
- **LAS 轮询超时**: 默认 600 秒，大文件可能需要更长时间
- **LAS Region**: 必须与 TOS Bucket 区域一致，否则权限异常
- **空文件/无效视频**: 0 字节或非视频文件会抛出明确错误
- **API Key 安全**: 使用环境变量或 `.env` 文件，不要硬编码
- **output_tos_path**: LAS 剪辑时必须是 `tos://` 前缀的目录路径
