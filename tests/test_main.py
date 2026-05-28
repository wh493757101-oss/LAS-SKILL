import json

import pytest

from src.main import PipelineConfig, PipelineResult, VideoHighlightPipeline
from src.video_editor import EditResult
from src.video_fetcher import LocalFileSource, UrlSource, VideoMetadata


class TestPipelineConfig:
    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.output_dir == ""

    def test_custom(self):
        cfg = PipelineConfig(output_dir="/tmp/pipeline")
        assert cfg.output_dir == "/tmp/pipeline"


class TestPipelineResult:
    def test_with_edit(self):
        metadata = VideoMetadata(
            path="/tmp/v.mp4", duration=10.0, fps=30.0, width=1920, height=1080
        )
        edit = EditResult(output_path="/tmp/out.mp4", source="multimodal",
                          segments=[{"start_time": 1.0, "end_time": 4.0, "score": 0.9}])
        result = PipelineResult(metadata=metadata, edit=edit)
        assert result.edit is not None
        assert result.edit.output_path == "/tmp/out.mp4"
        assert len(result.edit.segments) == 1
        assert result.error is None

    def test_with_error(self):
        metadata = VideoMetadata(
            path="/tmp/v.mp4", duration=10.0, fps=30.0, width=1920, height=1080
        )
        result = PipelineResult(
            metadata=metadata, error="处理失败"
        )
        assert result.error == "处理失败"
        assert result.edit is None


class TestVideoHighlightPipeline:
    def _make_metadata(self, tmp_path, duration=10.0, fps=30.0):
        return VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=duration,
            fps=fps,
            width=1920,
            height=1080,
        )

    def test_run_full_pipeline(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "editor",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata

        from src.rule_engine import HighlightSegment
        pipeline.detector.detect.return_value.segments = [
            HighlightSegment(start_time=1.0, end_time=4.0, combined_score=0.9),
            HighlightSegment(start_time=6.0, end_time=9.0, combined_score=0.7),
        ]
        pipeline.detector.detect.return_value.source = "multimodal"

        pipeline.editor.edit_with_ffmpeg.return_value = EditResult(
            output_path="/tmp/output/highlight_reel.mp4",
            source="multimodal",
            segments=[
                {"start_time": 1.0, "end_time": 4.0, "score": 0.9},
                {"start_time": 6.0, "end_time": 9.0, "score": 0.7},
            ],
        )

        result = pipeline.run(
            LocalFileSource(str(tmp_path / "video.mp4")),
            description="剪辑精彩片段",
        )

        assert result.edit is not None
        assert result.edit.source == "multimodal"
        assert len(result.edit.segments) == 2
        assert result.error is None

    def test_run_skip_edit(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata

        result = pipeline.run(
            LocalFileSource(str(tmp_path / "video.mp4")),
            skip_edit=True,
        )

        assert result.edit is None
        assert result.error is None

    def test_run_from_path(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "editor",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata

        from src.rule_engine import HighlightSegment
        pipeline.detector.detect.return_value.segments = [
            HighlightSegment(start_time=1.0, end_time=4.0, combined_score=0.9),
        ]
        pipeline.detector.detect.return_value.source = "multimodal"

        pipeline.editor.edit_with_ffmpeg.return_value = EditResult(
            output_path="highlight_reel.mp4",
            source="multimodal",
            segments=[{"start_time": 1.0, "end_time": 4.0, "score": 0.9}],
        )

        result = pipeline.run_from_path("/tmp/video.mp4", description="剪辑")
        assert result.edit.source == "multimodal"

    def test_run_from_url(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "editor",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )

        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata

        pipeline.detector.detect.return_value.segments = []
        pipeline.detector.detect.return_value.source = "multimodal"

        pipeline.editor.edit_with_ffmpeg.return_value = EditResult(
            output_path="highlight_reel.mp4",
            source="multimodal",
            segments=[],
        )

        result = pipeline.run_from_url("https://example.com/video.mp4")
        assert result.edit is not None

    def test_format_result_basic(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=15.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        edit = EditResult(
            output_path="/tmp/output/highlight_reel.mp4",
            source="multimodal",
            segments=[
                {"start_time": 2.0, "end_time": 5.0, "score": 0.9},
                {"start_time": 8.0, "end_time": 12.0, "score": 0.7},
            ],
        )
        result = PipelineResult(metadata=metadata, edit=edit)

        pipeline = VideoHighlightPipeline()
        formatted = pipeline.format_result(result)

        assert "视频高光剪辑" in formatted
        assert "15.0s" in formatted
        assert "1920x1080" in formatted
        assert "#1: 2.0s - 5.0s" in formatted
        assert "#2: 8.0s - 12.0s" in formatted
        assert "multimodal" in formatted

    def test_format_result_with_error(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=10.0,
            fps=24.0,
            width=1280,
            height=720,
        )
        result = PipelineResult(
            metadata=metadata, error="处理失败"
        )

        pipeline = VideoHighlightPipeline()
        formatted = pipeline.format_result(result)

        assert "[警告] 处理失败" in formatted

    def test_export_json_basic(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=10.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        edit = EditResult(
            output_path="/tmp/output/highlight_reel.mp4",
            source="multimodal",
            segments=[
                {"start_time": 1.0, "end_time": 3.0, "score": 0.9},
            ],
        )
        result = PipelineResult(metadata=metadata, edit=edit)

        pipeline = VideoHighlightPipeline()
        exported = pipeline.export_json(result)
        data = json.loads(exported)

        assert data["video"]["duration"] == 10.0
        assert data["video"]["width"] == 1920
        assert data["edit"]["source"] == "multimodal"
        assert len(data["edit"]["segments"]) == 1
        assert data["edit"]["segments"][0]["score"] == 0.9

    def test_export_json_with_error(self, tmp_path):
        metadata = VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=5.0,
            fps=25.0,
            width=640,
            height=480,
        )
        result = PipelineResult(
            metadata=metadata, error="处理失败"
        )

        pipeline = VideoHighlightPipeline()
        exported = pipeline.export_json(result)
        data = json.loads(exported)

        assert data["error"] == "处理失败"
        assert "edit" not in data

    def test_lazy_init_properties(self):
        pipeline = VideoHighlightPipeline()
        assert pipeline._fetcher is None
        assert pipeline._editor is None
        assert pipeline._detector is None

    def test_run_fetch_exception_returns_error(self, mocker):
        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.side_effect = FileNotFoundError("视频文件不存在")

        result = pipeline.run(LocalFileSource("/nonexistent/video.mp4"))
        assert result.error is not None
        assert "视频文件不存在" in result.error

    def test_run_editor_exception_returns_error(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mocker.patch.object(
            VideoHighlightPipeline, "fetcher",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "detector",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        mocker.patch.object(
            VideoHighlightPipeline, "editor",
            new_callable=mocker.PropertyMock,
            return_value=mocker.MagicMock(),
        )
        pipeline = VideoHighlightPipeline()
        pipeline.fetcher.fetch.return_value = metadata

        from src.rule_engine import HighlightSegment
        pipeline.detector.detect.return_value.segments = [
            HighlightSegment(start_time=1.0, end_time=4.0, combined_score=0.9),
        ]
        pipeline.detector.detect.return_value.source = "multimodal"

        pipeline.editor.edit_with_ffmpeg.side_effect = RuntimeError("剪辑失败")

        result = pipeline.run(LocalFileSource(str(tmp_path / "video.mp4")), description="剪辑")
        assert result.error is not None
        assert "剪辑失败" in result.error
