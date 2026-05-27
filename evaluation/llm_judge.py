import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEXT = """你是一个专业的视频剪辑质量评审员。请根据以下信息对高光集锦视频的剪辑质量进行多维度评分。

## 原始视频信息
- 视频类型: {category}
- 剪辑目标: {target}
- 风格要求: {style}

## 高光片段列表
{segments}

## 评分维度（每项 1-10 分）

1. **节奏感** — 片段衔接是否流畅，节奏是否符合风格要求
2. **内容完整性** — 每个高光片段是否完整表达了关键内容，是否有截断感
3. **精彩程度** — 选中的片段是否是真正的高光时刻
4. **指令契合度** — 剪辑结果是否符合用户的剪辑目标和要求

请以 JSON 格式返回评分结果：
{{
  "节奏感": <1-10>,
  "内容完整性": <1-10>,
  "精彩程度": <1-10>,
  "指令契合度": <1-10>,
  "总体评价": "<一句话评价>"
}}

只返回 JSON，不要包含其他文字。"""

JUDGE_PROMPT_VIDEO = """你是一个专业的视频剪辑质量评审员。请完整观看以下集锦视频（包含画面和音频），结合提供的原始视频信息和片段列表，对剪辑质量进行多维度评分。

## 原始视频信息
- 视频类型: {category}
- 剪辑目标: {target}
- 风格要求: {style}

## 高光片段列表（从原视频中选出的片段）
{segments}

## 评分维度（每项 1-10 分）

1. **节奏感** — 画面衔接是否流畅，转场是否自然，BGM 与画面的配合度
2. **内容完整性** — 每个高光片段是否完整表达了关键内容，音频是否自然过渡
3. **精彩程度** — 选中的画面是否真正精彩，音频高潮是否与画面同步
4. **指令契合度** — 剪辑结果是否符合用户的剪辑目标和要求

请以 JSON 格式返回评分结果：
{{
  "节奏感": <1-10>,
  "内容完整性": <1-10>,
  "精彩程度": <1-10>,
  "指令契合度": <1-10>,
  "总体评价": "<一句话评价>"
}}

只返回 JSON，不要包含其他文字。"""


@dataclass
class JudgeScore:
    rhythm: float = 0.0
    completeness: float = 0.0
    excitement: float = 0.0
    instruction_fit: float = 0.0
    overall_comment: str = ""
    error: str | None = None

    @property
    def average(self) -> float:
        scores = [self.rhythm, self.completeness, self.excitement, self.instruction_fit]
        return sum(scores) / len(scores)


@dataclass
class JudgeReport:
    scores: list[JudgeScore] = field(default_factory=list)
    overall_rhythm: float = 0.0
    overall_completeness: float = 0.0
    overall_excitement: float = 0.0
    overall_instruction_fit: float = 0.0
    overall_average: float = 0.0
    degraded: bool = False


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

    def build_prompt(
        self,
        category: str,
        target: str,
        style: str,
        segments: list[dict[str, Any]],
    ) -> str:
        segment_lines: list[str] = []
        for i, seg in enumerate(segments):
            label = seg.get("label", "")
            label_str = f" ({label})" if label else ""
            segment_lines.append(
                f"  #{i + 1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s"
                f" (精彩度: {seg.get('score', 0):.2f}){label_str}"
            )

        return JUDGE_PROMPT_TEXT.format(
            category=category,
            target=target or "精彩集锦",
            style=style or "无特定要求",
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
        if video_path and Path(video_path).exists():
            return self._judge_video(category, target, style, segments, video_path, max_retries)
        return self._judge_text_only(category, target, style, segments, max_retries)

    def _resolve_video_url(self, video_path: str) -> str:
        """将本地视频路径转为模型可访问的 URL。

        - Ark 后端: 上传到 Files API，返回 download_url
        - DashScope 后端: 转 base64 data URI
        - 已经是 URL: 直接返回
        """
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

    def _judge_video(
        self,
        category: str,
        target: str,
        style: str,
        segments: list[dict[str, Any]],
        video_path: str,
        max_retries: int,
    ) -> JudgeScore:
        segment_lines: list[str] = []
        for i, seg in enumerate(segments):
            label = seg.get("label", "")
            label_str = f" ({label})" if label else ""
            segment_lines.append(
                f"  #{i + 1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s"
                f" (精彩度: {seg.get('score', 0):.2f}){label_str}"
            )

        prompt = JUDGE_PROMPT_VIDEO.format(
            category=category,
            target=target or "精彩集锦",
            style=style or "无特定要求",
            segments="\n".join(segment_lines) if segment_lines else "无",
        )

        video_url = self._resolve_video_url(video_path)
        logger.info("LLM Judge 视频评分: %s", video_path)

        last_error: str = ""
        for attempt in range(max_retries):
            try:
                if "dashscope" in self.config.base_url:
                    response = self.ark_client.chat_with_video_omni(
                        text=prompt,
                        video_url=video_url,
                        model=self.config.model,
                        temperature=0.3,
                        max_tokens=1024,
                    )
                else:
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
                parsed = self.ark_client.extract_json(response)
                return JudgeScore(
                    rhythm=float(parsed.get("节奏感", 0)),
                    completeness=float(parsed.get("内容完整性", 0)),
                    excitement=float(parsed.get("精彩程度", 0)),
                    instruction_fit=float(parsed.get("指令契合度", 0)),
                    overall_comment=str(parsed.get("总体评价", "")),
                )
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    logger.warning("LLM Judge 视频评分失败（第 %d/%d 次）: %s，重试中...", attempt + 1, max_retries, e)
                    time.sleep(1)
                    continue
                logger.warning("LLM Judge 视频评分失败（已重试 %d 次）: %s", max_retries, last_error)

        logger.warning("视频 Judge 全部失败，降级到纯文本 Judge")
        return self._judge_text_only(category, target, style, segments, max_retries)

    def _judge_text_only(
        self,
        category: str,
        target: str,
        style: str,
        segments: list[dict[str, Any]],
        max_retries: int,
    ) -> JudgeScore:
        if max_retries < 1:
            raise ValueError(f"max_retries 必须 >= 1，当前值: {max_retries}")

        prompt = self.build_prompt(category, target, style, segments)

        last_error: str = ""
        for attempt in range(max_retries):
            try:
                response = self.ark_client.chat(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.config.model,
                    temperature=0.3,
                    max_tokens=1024,
                )
                parsed = self.ark_client.extract_json(response)

                return JudgeScore(
                    rhythm=float(parsed.get("节奏感", 0)),
                    completeness=float(parsed.get("内容完整性", 0)),
                    excitement=float(parsed.get("精彩程度", 0)),
                    instruction_fit=float(parsed.get("指令契合度", 0)),
                    overall_comment=str(parsed.get("总体评价", "")),
                )
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    logger.warning("LLM Judge 评分失败（第 %d/%d 次）: %s，重试中...", attempt + 1, max_retries, e)
                    time.sleep(1)
                    continue
                logger.warning("LLM Judge 评分失败（已重试 %d 次）: %s", max_retries, last_error)
                return JudgeScore(error=last_error)

    def judge_all(self, cases: list[dict[str, Any]], max_retries: int = 3) -> JudgeReport:
        report = JudgeReport()
        if not cases:
            return report

        total_rhythm = 0.0
        total_completeness = 0.0
        total_excitement = 0.0
        total_fit = 0.0
        valid_count = 0

        for case in cases:
            score = self.judge(
                category=case.get("category", ""),
                target=case.get("target", ""),
                style=case.get("style", ""),
                segments=case.get("segments", []),
                video_path=case.get("video_path", ""),
                max_retries=max_retries,
            )
            report.scores.append(score)

            if score.error:
                continue

            total_rhythm += score.rhythm
            total_completeness += score.completeness
            total_excitement += score.excitement
            total_fit += score.instruction_fit
            valid_count += 1

        if valid_count > 0:
            report.overall_rhythm = total_rhythm / valid_count
            report.overall_completeness = total_completeness / valid_count
            report.overall_excitement = total_excitement / valid_count
            report.overall_instruction_fit = total_fit / valid_count
            report.overall_average = (
                report.overall_rhythm
                + report.overall_completeness
                + report.overall_excitement
                + report.overall_instruction_fit
            ) / 4
        else:
            report.degraded = True

        return report


def format_judge_report(report: JudgeReport) -> str:
    lines = [
        "=" * 50,
        "LLM Judge 评分报告",
        "=" * 50,
        "",
        f"节奏感:       {report.overall_rhythm:.2f} / 10.0",
        f"内容完整性:   {report.overall_completeness:.2f} / 10.0",
        f"精彩程度:     {report.overall_excitement:.2f} / 10.0",
        f"指令契合度:   {report.overall_instruction_fit:.2f} / 10.0",
        f"综合均分:     {report.overall_average:.2f} / 10.0",
        "",
        "各用例评价:",
    ]

    for i, score in enumerate(report.scores):
        if score.error:
            lines.append(f"  #{i + 1}: [ERROR] {score.error}")
        else:
            lines.append(
                f"  #{i + 1}: 均分 {score.average:.1f} — {score.overall_comment}"
            )

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)
