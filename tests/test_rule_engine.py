import numpy as np
import pytest

from src.rule_engine import (
    AudioAnalyzer,
    HighlightSegment,
    RuleEngine,
    RuleEngineConfig,
    VisualAnalyzer,
)


class TestHighlightSegment:
    def test_duration(self):
        seg = HighlightSegment(start_time=1.0, end_time=5.0)
        assert seg.duration == 4.0

    def test_default_scores(self):
        seg = HighlightSegment(start_time=0.0, end_time=3.0)
        assert seg.audio_score == 0.0
        assert seg.visual_score == 0.0
        assert seg.combined_score == 0.0


class TestRuleEngineConfig:
    def test_defaults(self):
        cfg = RuleEngineConfig()
        assert cfg.top_k == 5
        assert cfg.segment_window == 3.0
        assert cfg.audio_weight == 0.5
        assert cfg.visual_weight == 0.5

    def test_custom(self):
        cfg = RuleEngineConfig(top_k=10, audio_weight=0.7, visual_weight=0.3)
        assert cfg.top_k == 10
        assert cfg.audio_weight == 0.7


class TestAudioAnalyzer:
    def test_analyze_empty(self):
        analyzer = AudioAnalyzer()
        result = analyzer.analyze(np.array([]))
        assert len(result) == 0

    def test_analyze_sine_wave(self):
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        analyzer = AudioAnalyzer(sr=sr)
        result = analyzer.analyze(y)

        assert len(result) > 0
        assert result.max() <= 1.0
        assert result.min() >= 0.0

    def test_analyze_with_loud_spike(self):
        sr = 22050
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        y[int(sr * 1.0) : int(sr * 1.3)] *= 5.0

        analyzer = AudioAnalyzer(sr=sr)
        result = analyzer.analyze(y)

        spike_region = result[len(result) // 3 : 2 * len(result) // 3]
        quiet_region = np.concatenate([result[: len(result) // 6], result[-len(result) // 6 :]])
        assert spike_region.max() > quiet_region.max()

    def test_detect_silence(self):
        sr = 22050
        y = np.zeros(int(sr * 2.0), dtype=np.float32)
        y[int(sr * 0.5) : int(sr * 1.5)] = 0.5 * np.sin(
            2 * np.pi * 440 * np.linspace(0, 1.0, int(sr * 1.0), endpoint=False)
        ).astype(np.float32)

        analyzer = AudioAnalyzer(sr=sr)
        intervals = analyzer.detect_silence(y, top_db=20.0)

        assert len(intervals) >= 1


class TestVisualAnalyzer:
    def test_analyze_empty(self):
        analyzer = VisualAnalyzer()
        result = analyzer.analyze([], 30.0)
        assert len(result) == 0

    def test_analyze_single_frame(self):
        analyzer = VisualAnalyzer()
        result = analyzer.analyze(["/fake/frame.jpg"], 30.0)
        assert len(result) == 0

    def test_analyze_static_frames(self, mocker):
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        mock_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128

        mocker.patch("cv2.imread", return_value=mock_frame)
        mocker.patch("cv2.cvtColor", return_value=gray)

        analyzer = VisualAnalyzer()
        result = analyzer.analyze(["f1.jpg", "f2.jpg", "f3.jpg"], 30.0)

        assert len(result) == 3
        assert all(0.0 <= s <= 1.0 for s in result)

    def test_analyze_motion_frames(self, mocker):
        dark = np.ones((100, 100), dtype=np.uint8) * 50
        bright = np.ones((100, 100), dtype=np.uint8) * 200

        mock_reads = [
            np.ones((100, 100, 3), dtype=np.uint8) * 50,
            np.ones((100, 100, 3), dtype=np.uint8) * 200,
        ]
        mock_grays = [dark, bright]

        mocker.patch("cv2.imread", side_effect=mock_reads)
        mocker.patch("cv2.cvtColor", side_effect=mock_grays)

        analyzer = VisualAnalyzer()
        result = analyzer.analyze(["f1.jpg", "f2.jpg"], 30.0)

        assert len(result) == 2
        assert result[1] > result[0]

    def test_analyze_unreadable_frame(self, mocker):
        mocker.patch("cv2.imread", side_effect=[None, np.ones((100, 100, 3), dtype=np.uint8) * 128])
        mocker.patch("cv2.cvtColor", return_value=np.ones((100, 100), dtype=np.uint8) * 128)

        analyzer = VisualAnalyzer()
        result = analyzer.analyze(["bad.jpg", "good.jpg"], 30.0)

        assert len(result) == 2
        assert result[0] == 0.0


class TestRuleEngine:
    def test_detect_basic(self, mocker, tmp_path):
        sr = 22050
        duration = 10.0
        fps = 30.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        audio[int(sr * 3.0) : int(sr * 4.0)] *= 10.0

        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        frame_paths = []
        for i in range(20):
            fp = frames_dir / f"frame_{i:06d}.jpg"
            fp.touch()
            frame_paths.append(str(fp))

        mock_frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        mock_gray = np.ones((100, 100), dtype=np.uint8) * 128
        mocker.patch("cv2.imread", return_value=mock_frame)
        mocker.patch("cv2.cvtColor", return_value=mock_gray)

        engine = RuleEngine(RuleEngineConfig(top_k=3, segment_window=3.0))
        segments = engine.detect(audio, sr, frame_paths, fps, duration)

        assert len(segments) >= 1
        assert len(segments) <= 3
        for seg in segments:
            assert 0.0 <= seg.start_time <= duration
            assert seg.start_time < seg.end_time <= duration
            assert seg.combined_score >= 0.0

    def test_detect_empty_inputs(self):
        engine = RuleEngine()
        segments = engine.detect(
            np.array([], dtype=np.float32), 22050, [], 30.0, 10.0
        )
        assert segments == []

    def test_align_and_combine(self):
        engine = RuleEngine()
        audio_scores = np.array([0.1, 0.5, 0.9, 0.3, 0.1])
        visual_scores = np.array([0.2, 0.4, 0.8, 0.6, 0.2])
        combined = engine._align_and_combine(audio_scores, visual_scores, 10.0)

        assert len(combined) > 0
        peak_idx = np.argmax(combined)
        assert 0.3 < peak_idx / len(combined) < 0.7

    def test_align_and_combine_empty(self):
        engine = RuleEngine()
        combined = engine._align_and_combine(
            np.array([]), np.array([]), 10.0
        )
        assert len(combined) > 0
        assert combined.sum() == 0.0

    def test_extract_segments(self):
        engine = RuleEngine()
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.1, 0.1])
        segments = engine._extract_segments(scores, 10.0)

        assert len(segments) >= 1
        for seg in segments:
            assert seg.start_time < seg.end_time

    def test_extract_segments_empty(self):
        engine = RuleEngine()
        segments = engine._extract_segments(np.array([]), 10.0)
        assert segments == []

    def test_merge_overlapping(self):
        engine = RuleEngine(RuleEngineConfig(min_segment_gap=1.0))
        segments = [
            HighlightSegment(start_time=0.0, end_time=3.0, combined_score=0.8),
            HighlightSegment(start_time=3.5, end_time=6.0, combined_score=0.9),
            HighlightSegment(start_time=7.5, end_time=10.0, combined_score=0.7),
        ]
        merged = engine._merge_overlapping(segments)
        assert len(merged) == 2

    def test_merge_overlapping_empty(self):
        engine = RuleEngine()
        assert engine._merge_overlapping([]) == []

    def test_filter_by_duration(self):
        engine = RuleEngine(RuleEngineConfig(min_duration=2.0, max_duration=10.0))
        segments = [
            HighlightSegment(start_time=0.0, end_time=1.0),
            HighlightSegment(start_time=1.0, end_time=5.0),
            HighlightSegment(start_time=5.0, end_time=18.0),
        ]
        filtered = engine._filter_by_duration(segments)
        assert len(filtered) == 1
        assert filtered[0].duration == 4.0

    def test_top_k_limit(self, mocker):
        sr = 22050
        duration = 30.0
        fps = 30.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = np.random.randn(len(t)).astype(np.float32) * 0.1

        frame_paths = [f"/fake/f{i:06d}.jpg" for i in range(60)]
        mock_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        mocker.patch("cv2.imread", return_value=mock_frame)
        mocker.patch("cv2.cvtColor", return_value=mock_gray)

        engine = RuleEngine(RuleEngineConfig(top_k=3))
        segments = engine.detect(audio, sr, frame_paths, fps, duration)

        assert len(segments) <= 3
