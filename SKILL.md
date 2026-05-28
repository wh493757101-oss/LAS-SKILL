---
name: video-highlight
description: >
  视频高光剪辑 — 输入长视频+剪辑指令，多模态大模型识别高光 + FFmpeg 无损拼接输出集锦视频。
  触发词: 视频高光、高光剪辑、精彩集锦、视频剪辑、剪辑视频、highlight、highlight reel、video highlight。
  使用场景: 用户需要从长视频中提取精彩片段、制作集锦、高光时刻剪辑。
version: 2.0.0
metadata:
  openclaw:
    requires:
      env:
        - ARK_HIGHLIGHT_API_KEY
      bins:
        - ffmpeg
    primaryEnv: ARK_HIGHLIGHT_API_KEY
    envVars:
      - name: ARK_HIGHLIGHT_API_KEY
        description: "火山引擎 Ark API Key，用于多模态高光识别"
        required: true
      - name: TOS_ACCESS_KEY
        description: "TOS Access Key（TOS 路径输入时必需）"
        required: false
      - name: TOS_SECRET_KEY
        description: "TOS Secret Key（TOS 路径输入时必需）"
        required: false
---

# 视频高光剪辑 Skill

基于多模态大模型识别高光 + FFmpeg 流拷贝拼接，自动识别视频中的高光片段并生成集锦视频。

## 输入输出

### 输入

| 字段 | 说明 | 必需 |
|------|------|------|
| 原始视频 | 长视频文件（本地路径 / URL / TOS 路径） | 是 |
| 剪辑描述 | 如"精彩集锦""进球片段""高能时刻""快节奏" | 否 |

### 输出

| 字段 | 说明 |
|------|------|
| 高光视频 | 本地集锦视频文件路径 |
| 片段列表 | 每个片段的起止时间、置信度评分、标签 |
| JSON 导出 | 结构化片段数据 |

## 核心能力

1. **多模态高光识别** — 多模态大模型根据剪辑指令识别高光片段
2. **FFmpeg 无损拼接** — 流拷贝模式拼接，零成本、秒级完成
3. **结构化输出** — 输出集锦视频 + 时间戳 + 置信度评分 + 标签
4. **多源输入** — 支持本地文件、URL、TOS 路径

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
- [ ] Step 2: 多模态高光识别 + FFmpeg 拼接
- [ ] Step 3: 结果呈现
```

### Step 0: 前置检查

**环境变量检查：**
- 确认 `ARK_HIGHLIGHT_API_KEY` 已配置
- 如输入为本地文件或 TOS 路径，确认 `TOS_ACCESS_KEY` / `TOS_SECRET_KEY` 已配置
- 缺失时必须向用户索要

**输入来源检查：**
- 本地文件：确认文件存在，格式为常见视频格式（mp4/mov/avi 等）
- URL 链接：确认链接可访问，支持 yt-dlp 兼容平台
- TOS 路径：确认凭证有效，路径格式为 `tos://bucket/key`

**输出路径检查：**
- 输出视频为本地 mp4 文件，无需 TOS 上传

### Step 1: 视频获取与预处理

- **本地文件**: 直接加载，非 mp4 或无法解码时通过 FFmpeg 转码
- **URL**: 使用 yt-dlp 下载（默认 600 秒超时）
- **TOS 路径**: 使用 tos SDK 下载

预处理只做：格式校验（空文件/超大文件/时长超限）、元数据提取（fps/duration/分辨率）、按需转码。不提取音频、不采样关键帧。

### Step 2: 多模态高光识别 + FFmpeg 拼接

- 多模态大模型（Doubao）观看完整视频，根据剪辑指令识别高光片段
- 返回片段列表（start_time + end_time + score + label）
- FFmpeg 智能拼接：自动探测源编码，流拷贝（`-c copy`）与重编码自适应切换
  - 默认流拷贝模式（零质量损失、秒级完成），自动检测编码兼容性
  - 编码不兼容时自动切换重编码（CRF 23 + medium preset）
  - 可选：关键帧对齐切割、EBU R128 音频归一化、xfade 片段转场
- 输出本地 mp4 集锦视频
- 识别失败直接报错，不降级

### Step 3: 结果呈现

1. **视频信息**: 原视频时长、分辨率、帧率
2. **片段列表**: 每个片段的起止时间、置信度评分、标签
3. **输出视频**: 集锦视频 URL
4. **JSON 导出**: 结构化片段数据

## 环境变量

| 变量 | 说明 | 必需 |
|------|------|------|
| `ARK_HIGHLIGHT_API_KEY` | 火山引擎 Ark API Key | 是 |
| `TOS_ACCESS_KEY` | TOS Access Key | 否（TOS 路径输入时必需） |
| `TOS_SECRET_KEY` | TOS Secret Key | 否（TOS 路径输入时必需） |

## 错误处理

Pipeline 采用"快速失败"策略：多模态识别不可用时直接返回错误（`PipelineResult.error`），不降级。

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
- [ ] 环境变量是否正确配置（ARK_HIGHLIGHT_API_KEY 已设置）
- [ ] 输入文件是否成功加载（非空、可解码）
- [ ] 输出结果是否正确呈现（视频路径 + 片段列表 + JSON）

**业务层面：**
- [ ] 高光片段时长合理（建议 2-10 秒/段）
- [ ] 集锦总时长不超过原视频的 30%
- [ ] 输出视频格式为 mp4（H.264 + AAC）

## Gotchas

- **URL 下载**: 使用 yt-dlp，支持主流平台，默认 600 秒超时
- **URL 下载**: 使用 yt-dlp，支持主流平台，默认 600 秒超时
- **多模态识别**: 使用火山引擎 Ark API（Doubao 模型），需要 `ARK_HIGHLIGHT_API_KEY`
- **FFmpeg 拼接**: 使用流拷贝模式（`-c copy`），不重新编码，秒级完成
- **空文件/无效视频**: 0 字节或非视频文件会抛出明确错误
- **API Key 安全**: 使用环境变量或 `.env` 文件，不要硬编码
