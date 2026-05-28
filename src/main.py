import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .highlight_detector import DetectorConfig, HighlightDetector
from .video_editor import EditResult, EditorConfig, VideoEditor
from .video_fetcher import (
    LocalFileSource,
    TosSource,
    UrlSource,
    VideoFetcher,
    VideoMetadata,
    VideoSource,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    editor: EditorConfig = field(default_factory=EditorConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    output_dir: str = ""


@dataclass
class PipelineTiming:
    """各阶段耗时（秒），-1 表示未执行该阶段。"""
    fetch: float = 0.0
    detection: float = 0.0
    clip_concat: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "fetch": round(self.fetch, 2),
            "detection": round(self.detection, 2),
            "clip_concat": round(self.clip_concat, 2),
        }


@dataclass
class TokenUsage:
    """单次 API 调用的 token 消耗。"""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total,
        }


@dataclass
class PipelineResult:
    metadata: VideoMetadata
    edit: EditResult | None = None
    error: str | None = None
    session_dir: str = ""
    elapsed_time: float = 0.0
    timing: PipelineTiming = field(default_factory=PipelineTiming)
    detection_usage: TokenUsage = field(default_factory=TokenUsage)
    judge_usage: TokenUsage = field(default_factory=TokenUsage)


class VideoHighlightPipeline:
    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._fetcher: VideoFetcher | None = None
        self._editor: VideoEditor | None = None
        self._detector: HighlightDetector | None = None

    @property
    def fetcher(self) -> VideoFetcher:
        if self._fetcher is None:
            self._fetcher = VideoFetcher(output_dir=self.config.output_dir or None)
        return self._fetcher

    @property
    def editor(self) -> VideoEditor:
        if self._editor is None:
            self._editor = VideoEditor(self.config.editor)
        return self._editor

    @property
    def detector(self) -> HighlightDetector:
        if self._detector is None:
            self._detector = HighlightDetector(self.config.detector)
        return self._detector

    def run(
        self,
        source: VideoSource,
        description: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        try:
            return self._run_impl(source, description, skip_edit)
        except Exception as e:
            logger.error("Pipeline 执行失败: %s", e, exc_info=True)
            return PipelineResult(
                metadata=VideoMetadata(path="", duration=0, fps=0, width=0, height=0),
                error=f"处理失败，请稍后重试: {e}",
                session_dir="",
            )

    def _make_session_dir(self, video_path: str) -> str:
        base = Path(self.config.output_dir) if self.config.output_dir else Path.cwd() / "output"
        video_name = Path(video_path).stem
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_dir = base / f"{video_name}_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir)

    def _run_impl(
        self,
        source: VideoSource,
        description: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        t_start = time.time()

        t0 = time.time()
        metadata = self.fetcher.fetch(source)
        t_fetch = time.time() - t0
        logger.info("视频预处理完成: duration=%.1fs, fps=%.1f", metadata.duration, metadata.fps)

        session_dir = self._make_session_dir(metadata.path)

        if skip_edit:
            return PipelineResult(
                metadata=metadata, session_dir=session_dir,
                elapsed_time=time.time() - t_start,
                timing=PipelineTiming(fetch=t_fetch),
            )

        self.config.editor.output_dir = session_dir

        t1 = time.time()
        detection_result = self.detector.detect(metadata, description)
        t_detect = time.time() - t1
        logger.info("多模态高光检测完成: source=%s, segments=%d",
                    detection_result.source, len(detection_result.segments))

        detection_usage = TokenUsage(
            input_tokens=detection_result.input_tokens,
            output_tokens=detection_result.output_tokens,
        )

        segments = [
            {
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "score": seg.combined_score,
                "label": getattr(seg, "label", ""),
            }
            for seg in detection_result.segments
        ]
        edit = self.editor.edit_with_ffmpeg(metadata.path, segments)
        logger.info("FFmpeg 拼接完成: output=%s, segments=%d",
                    edit.output_path, len(edit.segments))

        timing = PipelineTiming(
            fetch=t_fetch,
            detection=t_detect,
            clip_concat=edit.timing.ffmpeg_concat if edit.timing else -1,
        )

        return PipelineResult(
            metadata=metadata, edit=edit, session_dir=session_dir,
            elapsed_time=time.time() - t_start,
            timing=timing,
            detection_usage=detection_usage,
        )

    def run_from_path(
        self,
        video_path: str,
        description: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        return self.run(LocalFileSource(video_path), description, skip_edit)

    def run_from_url(
        self,
        url: str,
        description: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        return self.run(UrlSource(url), description, skip_edit)

    def run_from_tos(
        self,
        tos_path: str,
        description: str = "",
        skip_edit: bool = False,
    ) -> PipelineResult:
        return self.run(TosSource(tos_path), description, skip_edit)

    def format_result(self, result: PipelineResult) -> str:
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("视频高光剪辑 — 处理结果")
        lines.append("=" * 60)

        if result.session_dir:
            lines.append(f"\n[输出目录] {result.session_dir}")

        lines.append("\n[视频信息]")
        lines.append(f"  文件: {result.metadata.path}")
        lines.append(f"  时长: {result.metadata.duration:.1f}s")
        lines.append(f"  分辨率: {result.metadata.width}x{result.metadata.height}")
        lines.append(f"  帧率: {result.metadata.fps:.1f} fps")

        if result.edit:
            lines.append("\n[高光片段]")
            lines.append(f"  识别方式: {result.edit.source}")
            lines.append(f"  片段数: {len(result.edit.segments)}")
            for i, seg in enumerate(result.edit.segments):
                lines.append(
                    f"  #{i + 1}: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s"
                    f" (置信度: {seg.get('score', 0):.2f})"
                )
            lines.append("\n[剪辑输出]")
            lines.append(f"  输出路径: {result.edit.output_path}")

        if result.detection_usage.total > 0:
            lines.append(f"\n[Token 消耗 - 多模态识别]")
            lines.append(f"  输入: {result.detection_usage.input_tokens} tokens")
            lines.append(f"  输出: {result.detection_usage.output_tokens} tokens")
            lines.append(f"  合计: {result.detection_usage.total} tokens")

        if result.error:
            lines.append(f"\n[警告] {result.error}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    def export_json(self, result: PipelineResult) -> str:
        output: dict[str, Any] = {
            "video": {
                "path": result.metadata.path,
                "duration": result.metadata.duration,
                "fps": result.metadata.fps,
                "width": result.metadata.width,
                "height": result.metadata.height,
            },
        }

        if result.edit:
            output["edit"] = {
                "source": result.edit.source,
                "output_path": result.edit.output_path,
                "segments": result.edit.segments,
            }

        if result.detection_usage.total > 0:
            output["token_usage"] = {
                "detection": result.detection_usage.to_dict(),
                "judge": result.judge_usage.to_dict(),
            }

        if result.error:
            output["error"] = result.error

        if result.session_dir:
            output["session_dir"] = result.session_dir

        return json.dumps(output, ensure_ascii=False, indent=2)
