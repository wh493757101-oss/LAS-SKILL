import os
from pathlib import Path

import numpy as np
import pytest

from src.highlight_detector import (
    DetectorConfig,
    DetectionResult,
    HighlightDetector,
)
from src.rule_engine import HighlightSegment
from src.video_fetcher import VideoMetadata


class TestDetectorConfig:
    def test_defaults(self):
        cfg = DetectorConfig()
        assert cfg.frame_interval == 2.0
        assert cfg.max_frames_per_batch == 16
        assert cfg.ark_model == os.environ.get("ARK_HIGHLIGHT_MODEL", "")
        assert cfg.fallback_enabled is True

    def test_custom(self):
        cfg = DetectorConfig(frame_interval=5.0, max_frames_per_batch=8, fallback_enabled=False)
        assert cfg.frame_interval == 5.0
        assert cfg.fallback_enabled is False


class TestDetectionResult:
    def test_defaults(self):
        result = DetectionResult()
        assert result.segments == []
        assert result.source == "rule"
        assert result.raw_response is None

    def test_multimodal_result(self):
        segments = [HighlightSegment(start_time=1.0, end_time=3.0, combined_score=0.9)]
        result = DetectionResult(
            segments=segments,
            source="multimodal",
            raw_response={"segments": [{"start_time": 1.0, "end_time": 3.0, "score": 0.9}]},
        )
        assert result.source == "multimodal"
        assert len(result.segments) == 1
        assert result.raw_response is not None


class TestHighlightDetector:
    def _make_metadata(self, tmp_path, duration=10.0, fps=30.0):
        return VideoMetadata(
            path=str(tmp_path / "video.mp4"),
            duration=duration,
            fps=fps,
            width=1920,
            height=1080,
        )

    def test_detect_multimodal_success(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setenv("ARK_HIGHLIGHT_API_KEY", "test-key")
        monkeypatch.setenv("ARK_HIGHLIGHT_MODEL", "test-model")
        metadata = self._make_metadata(tmp_path)
        video_file = Path(metadata.path)
        video_file.write_bytes(b"fake mp4 content")

        mock_chat_response = {
            "choices": [{"message": {"content": '{"segments": [{"start_time": 2.0, "end_time": 5.0, "label": "精彩动作", "score": 0.9, "reason": "画面变化剧烈"}]}'}}],
        }

        detector = HighlightDetector()
        mocker.patch.object(detector.ark_client, "chat", return_value=mock_chat_response)

        result = detector.detect(metadata, asr_text="测试语音文本")

        assert result.source == "multimodal"
        assert len(result.segments) == 1
        assert result.segments[0].start_time == 2.0
        assert result.segments[0].end_time == 5.0
        assert result.segments[0].combined_score == 0.9

    def test_detect_multimodal_no_video(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)
        metadata.path = str(tmp_path / "nonexistent.mp4")

        mock_rule = mocker.patch.object(
            HighlightDetector, "_detect_rule_based",
            return_value=DetectionResult(source="rule"),
        )

        detector = HighlightDetector()
        result = detector.detect(metadata)

        assert result.source == "rule"
        mock_rule.assert_called_once()

    def test_detect_fallback_on_ark_error(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        mock_rule = mocker.patch.object(
            HighlightDetector, "_detect_rule_based",
            return_value=DetectionResult(
                segments=[HighlightSegment(start_time=1.0, end_time=3.0, combined_score=0.7)],
                source="rule",
            ),
        )

        detector = HighlightDetector()
        detector._detect_multimodal = mocker.MagicMock(side_effect=RuntimeError("Ark API 不可用"))

        result = detector.detect(metadata)

        assert result.source == "rule"
        assert len(result.segments) == 1
        mock_rule.assert_called_once()

    def test_detect_fallback_disabled(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        detector = HighlightDetector(DetectorConfig(fallback_enabled=False))
        detector._detect_multimodal = mocker.MagicMock(side_effect=RuntimeError("Ark API 不可用"))

        with pytest.raises(RuntimeError, match="Ark API 不可用"):
            detector.detect(metadata)

    def test_detect_rule_based(self, mocker, tmp_path):
        metadata = self._make_metadata(tmp_path)

        fake_audio = np.zeros(16000, dtype=np.float32)
        mocker.patch("librosa.load", return_value=(fake_audio, 16000))

        mock_engine = mocker.MagicMock()
        mock_engine.detect.return_value = [
            HighlightSegment(start_time=0.0, end_time=3.0, combined_score=0.8),
            HighlightSegment(start_time=5.0, end_time=8.0, combined_score=0.6),
        ]

        detector = HighlightDetector(rule_engine=mock_engine)
        result = detector._detect_rule_based(metadata)

        assert result.source == "rule"
        assert len(result.segments) == 2
        mock_engine.detect.assert_called_once()

    def test_parse_segments_valid(self):
        detector = HighlightDetector()
        parsed = {
            "segments": [
                {"start_time": 1.0, "end_time": 4.0, "score": 0.9},
                {"start_time": 6.0, "end_time": 9.0, "score": 0.7},
            ]
        }
        segments = detector._parse_segments(parsed, 10.0)

        assert len(segments) == 2
        assert segments[0].combined_score == 0.9
        assert segments[1].combined_score == 0.7

    def test_parse_segments_empty(self):
        detector = HighlightDetector()
        segments = detector._parse_segments({}, 10.0)
        assert segments == []

    def test_parse_segments_clamp_to_duration(self):
        detector = HighlightDetector()
        parsed = {
            "segments": [
                {"start_time": -5.0, "end_time": 15.0, "score": 0.8},
            ]
        }
        segments = detector._parse_segments(parsed, 10.0)

        assert len(segments) == 1
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 10.0

    def test_parse_segments_invalid_entries_skipped(self):
        detector = HighlightDetector()
        parsed = {
            "segments": [
                {"start_time": "bad", "end_time": "data", "score": "nope"},
                {"start_time": 2.0, "end_time": 5.0, "score": 0.8},
            ]
        }
        segments = detector._parse_segments(parsed, 10.0)

        assert len(segments) == 1
        assert segments[0].combined_score == 0.8

    def test_ark_client_lazy_init(self, mocker, monkeypatch):
        monkeypatch.setenv("ARK_HIGHLIGHT_API_KEY", "test-key")
        monkeypatch.setenv("ARK_HIGHLIGHT_MODEL", "test-model")
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"segments": []}'}}],
        }
        mock_resp.raise_for_status = mocker.MagicMock()
        mocker.patch("httpx.post", return_value=mock_resp)

        detector = HighlightDetector()
        assert detector._ark_client is None
        client = detector.ark_client
        assert client is not None
        assert detector._ark_client is not None

    def test_rule_engine_lazy_init(self):
        detector = HighlightDetector()
        assert detector._rule_engine is None
        engine = detector.rule_engine
        assert engine is not None
        assert detector._rule_engine is not None
