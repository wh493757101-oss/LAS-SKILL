import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ============================================================
# Prompt 模板
# ============================================================

JUDGE_PROMPT_SEGMENT = """你是一个严格、客观、标准统一的视频片段质量评审专家。
你的任务是逐个观看以下高光片段视频，判断每个片段本身的质量。

⚠️ 重要规则：
1. 你只评测这些片段视频本身的质量，不需要判断"遗漏了什么高光"（你看不到原始视频）
2. 严格按照下面的评分标准打分，禁止主观臆断
3. 所有评分必须是 1-10 之间的整数
4. 只输出合法的 JSON 格式，不要任何额外文字、解释、markdown 或代码块
5. 如果信息不足无法评测，输出 {{"error": "信息不足无法评测"}}

## 上下文信息
- 视频类型: {category}
- 剪辑目标: {target}
- 风格要求: {style}

## 待评测的高光片段列表
{segments}

## 评分维度及量化标准（每项 1-10 分）

1. **内容完整性**（10 分）—— 每个片段是否完整保留了关键动作/事件
   - 10 分：所有片段都完整，无任何截断，开头结尾恰到好处
   - 8 分：1 个片段有轻微截断（如开头少了 0.5 秒），不影响理解
   - 6 分：2 个片段有截断，或 1 个片段关键内容被截断（如进球动作只播了一半）
   - 4 分：3 个及以上片段有截断，或多个片段关键内容缺失
   - 2 分：大部分片段都不完整，严重影响观看
   - 1 分：所有片段都被严重截断，无法正常观看

2. **片段质量**（10 分）—— 每个片段本身是否值得入选（画面质量、内容精彩程度、无冗余）
   - 10 分：所有片段都是高质量内容，画面清晰，内容精彩，无任何空镜头/静止画面/重复
   - 8 分：1 个片段质量一般（如画面略有抖动），或混入 1 个 <3 秒的无关内容
   - 6 分：2 个片段质量一般，或混入 1 个 ≥3 秒的无关/重复片段
   - 4 分：3 个及以上片段质量差，或混入多个无关片段
   - 2 分：超过 30% 的内容是低质量或无关片段
   - 1 分：大部分片段都不值得入选

3. **指令契合度**（10 分）—— 每个片段是否符合剪辑目标和风格要求
   - 10 分：所有片段完全符合剪辑目标和风格要求，无任何偏离
   - 8 分：1 个片段与目标略有偏差，不影响整体效果
   - 6 分：2 个片段有轻微偏差，或 1 个片段明显不符合要求
   - 4 分：多个片段明显偏离目标或风格
   - 2 分：大部分片段不符合指令要求
   - 1 分：所有片段都与指令要求无关

## 输出要求
必须严格按照以下 JSON 格式输出，不要任何额外内容：
{{
  "内容完整性": <整数>,
  "片段质量": <整数>,
  "指令契合度": <整数>,
  "总体得分": <保留 1 位小数的平均值>,
  "主要优点": "<1-2 个具体优点，无则写'无'>",
  "主要问题": "<至少 1 个具体问题，必须指出哪个片段有什么问题，例：'片段#1 00:01:23-00:01:45 进球画面被截断，片段#3 00:05:00-00:05:30 为无关空镜头'>"
}}"""

JUDGE_PROMPT_VIDEO = """你是一个严格、客观、标准统一的视频剪辑质量评审专家。
你的任务是完整观看以下集锦视频（包含画面和音频），对多模态大模型识别生成的高光集锦的整体质量进行量化评分。

⚠️ 重要规则：
1. 你只评测集锦视频本身的观感质量（节奏、转场、音画、完整性），不需要判断"遗漏了什么高光"（你看不到原始视频）
2. 严格按照下面的评分标准打分，禁止主观臆断
3. 所有评分必须是 1-10 之间的整数
4. 只输出合法的 JSON 格式，不要任何额外文字、解释、markdown 或代码块
5. 如果无法观看视频或信息不足，输出 {{"error": "无法观看视频或信息不足"}}

## 上下文信息
- 视频类型: {category}
- 剪辑目标: {target}
- 风格要求: {style}

## 多模态大模型选出的高光片段列表（参考，帮助你了解每个片段的时间位置）
{segments}

## 评分维度及量化标准（每项 1-10 分）

1. **节奏感**（10 分）—— 整体剪辑节奏是否流畅、符合风格要求
   - 10 分：画面衔接流畅，转场自然，节奏张弛有度，完全符合风格要求（如快节奏体育集锦应有紧凑的剪辑节奏）
   - 8 分：1 处转场略显突兀或节奏略有波动，整体观感良好
   - 6 分：2-3 处转场突兀，或节奏有明显问题（如该快的地方慢了）
   - 4 分：多处转场生硬，节奏混乱，影响观看体验
   - 2 分：转场非常突兀，完全没有节奏感
   - 1 分：画面跳切严重，无法正常观看

2. **转场质量**（10 分）—— 片段之间的过渡是否自然、有无黑屏/卡顿
   - 10 分：所有转场都自然流畅，无黑屏、无卡顿、无跳帧
   - 8 分：1 处转场有轻微瑕疵（如 0.5 秒内的黑场），不影响观感
   - 6 分：2-3 处转场有瑕疵，或 1 处明显黑屏/卡顿
   - 4 分：多处转场有明显问题（黑屏 >1 秒、跳帧、重复帧）
   - 2 分：转场质量很差，严重影响观看
   - 1 分：几乎每个转场都有问题

3. **音画同步**（10 分）—— 音频与画面是否同步，BGM 是否匹配
   - 10 分：音频与画面完全同步，BGM/音效与画面内容完美匹配
   - 8 分：1 处轻微不同步（<0.3 秒偏差），BGM 整体匹配
   - 6 分：2-3 处不同步，或 BGM 与画面配合一般
   - 4 分：多处明显不同步，或 BGM 风格严重不匹配
   - 2 分：音画严重不同步，几乎无法观看
   - 1 分：完全没有音频或音频完全混乱

4. **内容完整性**（10 分）—— 集锦中各片段是否有截断、关键内容是否完整
   - 10 分：所有片段都完整，关键动作/事件有头有尾，音频无断裂
   - 8 分：1 个片段有轻微截断，不影响理解
   - 6 分：2 个片段有截断，或 1 个片段关键内容被截断
   - 4 分：3 个及以上片段有截断，或音频多处断裂
   - 2 分：大部分片段都不完整
   - 1 分：所有片段都被严重截断

5. **指令契合度**（10 分）—— 集锦整体是否符合剪辑目标和风格要求
   - 10 分：完全符合剪辑目标和风格要求，无任何偏离
   - 8 分：有 1 处轻微偏离，不影响整体效果
   - 6 分：有 2 处轻微偏离，或 1 处明显偏离
   - 4 分：有多处明显偏离，或 1 处严重偏离
   - 2 分：大部分内容不符合指令要求
   - 1 分：完全不符合指令要求

## 输出要求
必须严格按照以下 JSON 格式输出，不要任何额外内容：
{{
  "节奏感": <整数>,
  "转场质量": <整数>,
  "音画同步": <整数>,
  "内容完整性": <整数>,
  "指令契合度": <整数>,
  "总体得分": <保留 1 位小数的平均值>,
  "主要优点": "<1-2 个具体优点，无则写'无'>",
  "主要问题": "<至少 1 个具体问题，必须指出时间戳和具体问题，例：'00:01:23 转场出现 1 秒黑屏，00:03:00-00:03:15 片段被截断'>"
}}"""

# 保留旧 prompt 用于向后兼容的纯文本降级路径
JUDGE_PROMPT_TEXT = JUDGE_PROMPT_SEGMENT


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SegmentJudgeScore:
    """片段评测分数 — 逐个观看高光片段视频后的评分。

    评测的是每个片段本身的质量，不涉及"遗漏了哪些高光"（看不到原视频）。
    """
    content_completeness: float = 0.0
    segment_quality: float = 0.0
    instruction_fit: float = 0.0
    issues: list[dict[str, Any]] = field(default_factory=list)
    overall_comment: str = ""
    error: str | None = None

    @property
    def average(self) -> float:
        scores = [self.content_completeness, self.segment_quality, self.instruction_fit]
        return sum(scores) / len(scores)


@dataclass
class VideoJudgeScore:
    """集锦评测分数 — 观看拼接后集锦视频的评分。

    评测的是集锦视频的整体观感质量，不涉及"遗漏了哪些高光"（看不到原视频）。
    """
    rhythm: float = 0.0
    transition_quality: float = 0.0
    audiovisual_sync: float = 0.0
    content_completeness: float = 0.0
    instruction_fit: float = 0.0
    issues: list[dict[str, Any]] = field(default_factory=list)
    overall_comment: str = ""
    error: str | None = None

    @property
    def average(self) -> float:
        scores = [self.rhythm, self.transition_quality, self.audiovisual_sync,
                  self.content_completeness, self.instruction_fit]
        return sum(scores) / len(scores)


@dataclass
class JudgeScore:
    """向后兼容：旧版评测分数，从 VideoJudgeScore 映射而来。"""
    rhythm: float = 0.0
    transition_quality: float = 0.0
    audiovisual_sync: float = 0.0
    completeness: float = 0.0
    instruction_fit: float = 0.0
    overall_comment: str = ""
    error: str | None = None

    @property
    def average(self) -> float:
        scores = [self.rhythm, self.transition_quality, self.audiovisual_sync,
                  self.completeness, self.instruction_fit]
        return sum(scores) / len(scores)


@dataclass
class JudgeReport:
    # Segment judge 聚合（3 维度）
    segment_scores: list[SegmentJudgeScore] = field(default_factory=list)
    segment_content_completeness: float = 0.0
    segment_quality: float = 0.0
    segment_instruction_fit: float = 0.0
    segment_average: float = 0.0

    # Video judge 聚合（5 维度）
    video_scores: list[VideoJudgeScore] = field(default_factory=list)
    video_rhythm: float = 0.0
    video_transition_quality: float = 0.0
    video_audiovisual_sync: float = 0.0
    video_content_completeness: float = 0.0
    video_instruction_fit: float = 0.0
    video_average: float = 0.0

    # 降级标志
    degraded: bool = False
    segment_degraded: bool = False
    video_degraded: bool = False

    # 向后兼容聚合字段
    scores: list[JudgeScore] = field(default_factory=list)
    overall_rhythm: float = 0.0
    overall_transition_quality: float = 0.0
    overall_audiovisual_sync: float = 0.0
    overall_completeness: float = 0.0
    overall_instruction_fit: float = 0.0
    overall_average: float = 0.0


@dataclass
class JudgeConfig:
    api_key: str = field(default_factory=lambda: os.environ.get("ARK_JUDGE_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.environ.get("ARK_JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    model: str = field(default_factory=lambda: os.environ.get("ARK_JUDGE_MODEL", ""))
    max_retries: int = 3


class LLMJudge:
    def __init__(self, ark_client=None, config: JudgeConfig | None = None):
        self._ark_client = ark_client
        self.config = config or JudgeConfig()

    @property
    def ark_client(self):
        if self._ark_client is None:
            from src.ark_client import ArkClient, ArkConfig
            self._ark_client = ArkClient(ArkConfig(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model,
            ))
        return self._ark_client

    # ============================================================
    # 向后兼容方法（保留给 verify_e2e.py 等旧调用方）
    # ============================================================

    def build_prompt(
        self,
        category: str,
        target: str,
        style: str,
        segments: list[dict[str, Any]],
    ) -> str:
        segment_lines = self._format_segment_lines(segments)
        return JUDGE_PROMPT_TEXT.format(
            category=category,
            target=target or "精彩集锦",
            style=style or "无特定要求",
            core_highlight_definition="视频中最重要的高光时刻和关键场景",
            segments="\n".join(segment_lines) if segment_lines else "无",
        )

    def judge(
        self,
        category: str,
        target: str,
        style: str,
        segments: list[dict[str, Any]],
        video_path: str = "",
        max_retries: int = 3,
    ) -> JudgeScore:
        """向后兼容：委托给 judge_video，映射到旧 JudgeScore 格式。"""
        core_def = "视频中最重要的高光时刻和关键场景"
        vid_score = self.judge_video(category, target, style, core_def, segments, video_path, max_retries)
        return JudgeScore(
            rhythm=vid_score.rhythm,
            transition_quality=vid_score.transition_quality,
            audiovisual_sync=vid_score.audiovisual_sync,
            completeness=vid_score.content_completeness,
            instruction_fit=vid_score.instruction_fit,
            overall_comment=vid_score.overall_comment,
            error=vid_score.error,
        )

    # ============================================================
    # 核心方法
    # ============================================================

    def _format_segment_lines(self, segments: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for i, seg in enumerate(segments):
            label = seg.get("label", "")
            label_str = f" ({label})" if label else ""
            clip_url = seg.get("clip_url", "")
            url_str = f" [视频: {clip_url}]" if clip_url else ""
            lines.append(
                f"  #{i + 1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s"
                f" (置信度: {seg.get('score', 0):.2f}){label_str}{url_str}"
            )
        return lines

    def _resolve_video_url(self, video_path: str) -> str:
        if video_path.startswith(("http://", "https://", "tos://", "data:")):
            return video_path

        path = Path(video_path)
        if "dashscope" in self.config.base_url:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"data:video/mp4;base64,{data}"
        else:
            result = self.ark_client.upload_file(str(path))
            download_url = result.get("download_url", "")
            if not download_url:
                raise RuntimeError(
                    "Files API 未返回 download_url: "
                    f"{json.dumps(result, ensure_ascii=False)[:200]}"
                )
            return download_url

    def _call_judge_api(
        self,
        prompt: str,
        video_url: str | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """调用 LLM API 并返回解析后的 JSON 字典。失败抛出异常。"""
        last_error: str = ""
        for attempt in range(max_retries):
            try:
                if video_url and "dashscope" in self.config.base_url:
                    response = self.ark_client.chat_with_video_omni(
                        text=prompt,
                        video_url=video_url,
                        model=self.config.model,
                        temperature=0.3,
                        max_tokens=1024,
                    )
                elif video_url:
                    response = self.ark_client.chat(
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "video_url", "video_url": {"url": video_url}},
                                {"type": "text", "text": prompt},
                            ],
                        }],
                        model=self.config.model,
                        temperature=0.3,
                        max_tokens=1024,
                    )
                else:
                    response = self.ark_client.chat(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.config.model,
                        temperature=0.3,
                        max_tokens=1024,
                    )
                return self.ark_client.extract_json(response)
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    logger.warning(
                        "LLM Judge API 调用失败（第 %d/%d 次）: %s，重试中...",
                        attempt + 1, max_retries, e,
                    )
                    time.sleep(1)
                    continue
                raise RuntimeError(last_error) from e

    def judge_segment(
        self,
        category: str,
        target: str,
        style: str,
        core_highlight_definition: str,
        segments: list[dict[str, Any]],
        max_retries: int = 3,
    ) -> SegmentJudgeScore:
        """片段评测：逐个观看高光片段视频，判断每个片段本身的质量。

        注意：不评测"核心高光检出率"——看不到原始视频，无法判断遗漏。
        """
        segment_lines = self._format_segment_lines(segments)
        prompt = JUDGE_PROMPT_SEGMENT.format(
            category=category,
            target=target or "精彩集锦",
            style=style or "无特定要求",
            segments="\n".join(segment_lines) if segment_lines else "无",
        )

        # 收集有效的 clip_url
        clip_urls = [s.get("clip_url", "") for s in segments if s.get("clip_url")]
        video_url = clip_urls[0] if clip_urls else None

        try:
            parsed = self._call_judge_api(prompt, video_url=video_url, max_retries=max_retries)
            if "error" in parsed:
                return SegmentJudgeScore(error=str(parsed["error"]))
            return SegmentJudgeScore(
                content_completeness=float(parsed.get("内容完整性", 0)),
                segment_quality=float(parsed.get("片段质量", 0)),
                instruction_fit=float(parsed.get("指令契合度", 0)),
                overall_comment=str(parsed.get("主要优点", parsed.get("总体评价", ""))),
            )
        except Exception as e:
            logger.warning("Segment Judge 失败: %s", e)
            return SegmentJudgeScore(error=str(e))

    def judge_video(
        self,
        category: str,
        target: str,
        style: str,
        core_highlight_definition: str,
        segments: list[dict[str, Any]],
        video_path: str = "",
        max_retries: int = 3,
    ) -> VideoJudgeScore:
        """集锦评测：观看拼接后的集锦视频，对整体观感质量评分。

        注意：不评测"核心高光检出率"和"冗余控制"——看不到原始视频，无法判断遗漏/多余。
        """
        segment_lines = self._format_segment_lines(segments)
        prompt = JUDGE_PROMPT_VIDEO.format(
            category=category,
            target=target or "精彩集锦",
            style=style or "无特定要求",
            segments="\n".join(segment_lines) if segment_lines else "无",
        )

        is_video = video_path and (
            video_path.startswith(("http://", "https://", "tos://"))
            or Path(video_path).exists()
        )

        if is_video:
            video_url = self._resolve_video_url(video_path)
            logger.info("LLM Judge 视频评分: %s", video_path)
        else:
            video_url = None
            logger.info("LLM Judge 纯文本评分（无视频）")

        try:
            parsed = self._call_judge_api(prompt, video_url=video_url, max_retries=max_retries)
            if "error" in parsed:
                return VideoJudgeScore(error=str(parsed["error"]))
            return VideoJudgeScore(
                rhythm=float(parsed.get("节奏感", 0)),
                transition_quality=float(parsed.get("转场质量", 0)),
                audiovisual_sync=float(parsed.get("音画同步", 0)),
                content_completeness=float(parsed.get("内容完整性", 0)),
                instruction_fit=float(parsed.get("指令契合度", 0)),
                overall_comment=str(parsed.get("主要优点", parsed.get("总体评价", ""))),
            )
        except Exception as e:
            logger.warning("Video Judge 失败: %s", e)
            return VideoJudgeScore(error=str(e))

    def judge_all(self, cases: list[dict[str, Any]], max_retries: int = 3) -> JudgeReport:
        report = JudgeReport()
        if not cases:
            return report

        # Segment judge 聚合（3 维度）
        seg_total_cc = 0.0
        seg_total_sq = 0.0
        seg_total_if = 0.0
        seg_valid = 0

        # Video judge 聚合（5 维度）
        vid_total_rhythm = 0.0
        vid_total_tq = 0.0
        vid_total_avs = 0.0
        vid_total_cc = 0.0
        vid_total_if = 0.0
        vid_valid = 0

        # 向后兼容聚合（5 维度）
        compat_total_rhythm = 0.0
        compat_total_tq = 0.0
        compat_total_avs = 0.0
        compat_total_cc = 0.0
        compat_total_if = 0.0
        compat_valid = 0

        for case in cases:
            core_def = case.get("core_highlight_definition", "视频中最重要的高光时刻和关键场景")
            category = case.get("category", "")
            target = case.get("target", "")
            style = case.get("style", "")
            segments = case.get("segments", [])
            video_path = case.get("video_path", "")

            # Segment judge
            seg_score = self.judge_segment(
                category=category, target=target, style=style,
                core_highlight_definition=core_def,
                segments=segments, max_retries=max_retries,
            )
            report.segment_scores.append(seg_score)
            if not seg_score.error:
                seg_total_cc += seg_score.content_completeness
                seg_total_sq += seg_score.segment_quality
                seg_total_if += seg_score.instruction_fit
                seg_valid += 1

            # Video judge
            vid_score = self.judge_video(
                category=category, target=target, style=style,
                core_highlight_definition=core_def,
                segments=segments, video_path=video_path, max_retries=max_retries,
            )
            report.video_scores.append(vid_score)
            if not vid_score.error:
                vid_total_rhythm += vid_score.rhythm
                vid_total_tq += vid_score.transition_quality
                vid_total_avs += vid_score.audiovisual_sync
                vid_total_cc += vid_score.content_completeness
                vid_total_if += vid_score.instruction_fit
                vid_valid += 1

            # 向后兼容 JudgeScore
            compat = JudgeScore(
                rhythm=vid_score.rhythm,
                transition_quality=vid_score.transition_quality,
                audiovisual_sync=vid_score.audiovisual_sync,
                completeness=vid_score.content_completeness,
                instruction_fit=vid_score.instruction_fit,
                overall_comment=vid_score.overall_comment,
                error=vid_score.error,
            )
            report.scores.append(compat)
            if not compat.error:
                compat_total_rhythm += compat.rhythm
                compat_total_tq += compat.transition_quality
                compat_total_avs += compat.audiovisual_sync
                compat_total_cc += compat.completeness
                compat_total_if += compat.instruction_fit
                compat_valid += 1

        # 聚合 segment judge（3 维度）
        if seg_valid > 0:
            report.segment_content_completeness = seg_total_cc / seg_valid
            report.segment_quality = seg_total_sq / seg_valid
            report.segment_instruction_fit = seg_total_if / seg_valid
            report.segment_average = (
                report.segment_content_completeness + report.segment_quality
                + report.segment_instruction_fit
            ) / 3
        else:
            report.segment_degraded = True

        # 聚合 video judge（5 维度）
        if vid_valid > 0:
            report.video_rhythm = vid_total_rhythm / vid_valid
            report.video_transition_quality = vid_total_tq / vid_valid
            report.video_audiovisual_sync = vid_total_avs / vid_valid
            report.video_content_completeness = vid_total_cc / vid_valid
            report.video_instruction_fit = vid_total_if / vid_valid
            report.video_average = (
                report.video_rhythm + report.video_transition_quality
                + report.video_audiovisual_sync + report.video_content_completeness
                + report.video_instruction_fit
            ) / 5
        else:
            report.video_degraded = True

        # 向后兼容聚合（5 维度）
        if compat_valid > 0:
            report.overall_rhythm = compat_total_rhythm / compat_valid
            report.overall_transition_quality = compat_total_tq / compat_valid
            report.overall_audiovisual_sync = compat_total_avs / compat_valid
            report.overall_completeness = compat_total_cc / compat_valid
            report.overall_instruction_fit = compat_total_if / compat_valid
            report.overall_average = (
                report.overall_rhythm + report.overall_transition_quality
                + report.overall_audiovisual_sync + report.overall_completeness
                + report.overall_instruction_fit
            ) / 5
        else:
            report.degraded = True

        return report


def format_judge_report(report: JudgeReport) -> str:
    lines = [
        "=" * 60,
        "LLM Judge 评分报告",
        "=" * 60,
        "",
        "## 一、Segment Judge（片段质量评测 — 逐个观看片段视频）",
        "",
    ]

    if report.segment_degraded:
        lines.append("  [降级] Segment Judge 不可用")
    else:
        lines.append(f"  内容完整性: {report.segment_content_completeness:.2f} / 10.0")
        lines.append(f"  片段质量:   {report.segment_quality:.2f} / 10.0")
        lines.append(f"  指令契合度: {report.segment_instruction_fit:.2f} / 10.0")
        lines.append(f"  综合均分:   {report.segment_average:.2f} / 10.0")

    lines.append("")
    lines.append("  各用例 Segment 评价:")
    for i, score in enumerate(report.segment_scores):
        if score.error:
            lines.append(f"    #{i + 1}: [ERROR] {score.error}")
        else:
            lines.append(
                f"    #{i + 1}: {score.average:.1f}/10.0 — {score.overall_comment}"
            )

    lines.append("")
    lines.append("## 二、Video Judge（集锦质量评测 — 观看拼接后集锦视频）")
    lines.append("")

    if report.video_degraded:
        lines.append("  [降级] Video Judge 不可用")
    else:
        lines.append(f"  节奏感:     {report.video_rhythm:.2f} / 10.0")
        lines.append(f"  转场质量:   {report.video_transition_quality:.2f} / 10.0")
        lines.append(f"  音画同步:   {report.video_audiovisual_sync:.2f} / 10.0")
        lines.append(f"  内容完整性: {report.video_content_completeness:.2f} / 10.0")
        lines.append(f"  指令契合度: {report.video_instruction_fit:.2f} / 10.0")
        lines.append(f"  综合均分:   {report.video_average:.2f} / 10.0")

    lines.append("")
    lines.append("  各用例 Video 评价:")
    for i, score in enumerate(report.video_scores):
        if score.error:
            lines.append(f"    #{i + 1}: [ERROR] {score.error}")
        else:
            lines.append(
                f"    #{i + 1}: {score.average:.1f}/10.0 — {score.overall_comment}"
            )

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
