import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ark_client import ArkClient, ArkConfig
from .video_fetcher import VideoMetadata


@dataclass
class HighlightSegment:
    start_time: float
    end_time: float
    audio_score: float = 0.0
    visual_score: float = 0.0
    combined_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

logger = logging.getLogger(__name__)


@dataclass
class DetectorConfig:
    ark_model: str = field(default_factory=lambda: os.environ.get("ARK_HIGHLIGHT_MODEL", ""))
    ark_temperature: float = 0.3
    ark_max_tokens: int = 4096


HIGHLIGHT_PROMPT = """你是一个专业的视频高光检测分析器。请完整观看以下视频（包含画面和音频），根据用户的剪辑需求识别出最精彩的高光片段。

视频总时长: {duration:.1f} 秒。

## 用户剪辑需求
{user_instruction}

请根据以下维度综合判断高光片段：
1. 画面内容：场景切换、运动强度、视觉冲击力、关键动作和表情
2. 音频特征：音量变化、语速节奏、情绪爆发点、背景音乐高潮
3. 内容精彩度：事件重要性、情绪张力、叙事节奏
4. 音画配合：画面与音频的协调性和同步冲击力
5. 用户需求匹配度：片段是否符合用户的剪辑目标和风格要求

请以 JSON 格式返回结果，包含一个 segments 数组，每个元素包含：
- start_time: 起始时间（秒）
- end_time: 结束时间（秒）
- label: 高光类型标签（如 "精彩动作", "关键场景", "情绪爆发", "转场亮点", "音频高潮"）
- score: 精彩度评分（0.0-1.0）
- reason: 简短理由（一句话）

只返回 JSON，不要包含其他文字。"""


@dataclass
class DetectionResult:
    segments: list[HighlightSegment] = field(default_factory=list)
    source: str = "multimodal"
    raw_response: dict[str, Any] | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class HighlightDetector:
    def __init__(
        self,
        config: DetectorConfig | None = None,
        ark_client: ArkClient | None = None,
    ):
        self.config = config or DetectorConfig()
        self._ark_client = ark_client

    @property
    def ark_client(self) -> ArkClient:
        if self._ark_client is None:
            api_key = os.environ.get("ARK_HIGHLIGHT_API_KEY", "")
            self._ark_client = ArkClient(ArkConfig(api_key=api_key, model=self.config.ark_model))
        return self._ark_client

    def detect(
        self,
        metadata: VideoMetadata,
        description: str = "",
    ) -> DetectionResult:
        """多模态高光检测。失败直接抛异常，不降级。"""
        return self._detect_multimodal(metadata, description)

    @property
    def call_count(self) -> int:
        return self.ark_client.call_count

    @property
    def retry_count(self) -> int:
        return self.ark_client.retry_count

    def _detect_multimodal(
        self, metadata: VideoMetadata, description: str = ""
    ) -> DetectionResult:
        if not metadata.path:
            raise ValueError("视频路径为空，无法进行多模态检测")
        video_path = Path(metadata.path)
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {metadata.path}，请检查文件路径后重试")

        prompt = HIGHLIGHT_PROMPT.format(
            duration=metadata.duration,
            user_instruction=description or "识别视频中最精彩的高光片段",
        )

        response = self.ark_client.chat_with_video(
            text=prompt,
            video_path=str(video_path),
            model=self.config.ark_model,
            temperature=self.config.ark_temperature,
            max_tokens=self.config.ark_max_tokens,
        )

        parsed = self.ark_client.extract_json(response)
        segments = self._parse_segments(parsed, metadata.duration)

        usage = parsed.get("_usage", {})
        input_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))

        return DetectionResult(
            segments=segments,
            source="multimodal",
            raw_response=parsed,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def _parse_segments(
        self, parsed: dict[str, Any], duration: float
    ) -> list[HighlightSegment]:
        raw_segments = parsed.get("segments", [])
        if not raw_segments:
            return []

        result: list[HighlightSegment] = []
        for item in raw_segments:
            try:
                seg = HighlightSegment(
                    start_time=float(item.get("start_time", 0)),
                    end_time=float(item.get("end_time", 0)),
                    combined_score=float(item.get("score", 0.5)),
                )
                seg.start_time = max(0.0, min(seg.start_time, duration))
                seg.end_time = max(seg.start_time + 0.5, min(seg.end_time, duration))
                result.append(seg)
            except (ValueError, TypeError):
                continue

        result.sort(key=lambda s: s.combined_score, reverse=True)
        return result
