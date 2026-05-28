import json
import subprocess
from pathlib import Path

import pytest

from src.video_editor import (
    EditResult,
    EditorConfig,
    EncodingPreset,
    StreamInfo,
    VideoEditor,
    VideoProbe,
    _check_codec_consistency,
)


# ============================================================
# Data classes
# ============================================================

class TestStreamInfo:
    def test_fps_from_fraction(self):
        s = StreamInfo(r_frame_rate="30000/1001")
        assert pytest.approx(s.fps, 0.01) == 29.97

    def test_fps_from_integer_string(self):
        s = StreamInfo(r_frame_rate="30")
        assert s.fps == 30.0

    def test_fps_zero_division(self):
        s = StreamInfo(r_frame_rate="30000/0")
        assert s.fps == 0.0

    def test_fps_empty(self):
        s = StreamInfo(r_frame_rate="")
        assert s.fps == 0.0

    def test_fps_invalid(self):
        s = StreamInfo(r_frame_rate="not_a_number")
        assert s.fps == 0.0


class TestVideoProbe:
    def test_video_stream(self):
        vs = StreamInfo(codec_type="video", codec_name="h264", index=0)
        probe = VideoProbe(streams=[vs])
        assert probe.video_stream is not None
        assert probe.video_stream.codec_name == "h264"

    def test_video_stream_none(self):
        probe = VideoProbe(streams=[])
        assert probe.video_stream is None

    def test_audio_stream(self):
        audio = StreamInfo(codec_type="audio", codec_name="aac", index=1)
        probe = VideoProbe(streams=[StreamInfo(codec_type="video", codec_name="h264", index=0), audio])
        assert probe.audio_stream is not None
        assert probe.audio_stream.codec_name == "aac"

    def test_audio_stream_none(self):
        probe = VideoProbe(streams=[StreamInfo(codec_type="video", codec_name="h264", index=0)])
        assert probe.audio_stream is None


class TestEncodingPreset:
    def test_high_quality(self):
        assert EncodingPreset.HIGH_QUALITY.crf == 18
        assert EncodingPreset.HIGH_QUALITY.preset == "slow"

    def test_balanced(self):
        assert EncodingPreset.BALANCED.crf == 23
        assert EncodingPreset.BALANCED.preset == "medium"

    def test_compressed(self):
        assert EncodingPreset.COMPRESSED.crf == 28
        assert EncodingPreset.COMPRESSED.preset == "fast"

    def test_immutable(self):
        with pytest.raises(Exception):
            EncodingPreset.BALANCED.crf = 99  # type: ignore[misc]


# ============================================================
# EditorConfig
# ============================================================

class TestEditorConfig:
    def test_defaults(self):
        cfg = EditorConfig()
        assert cfg.output_dir == ""
        assert cfg.concat_list_filename == "concat_list.txt"

    def test_custom(self):
        cfg = EditorConfig(output_dir="/tmp/out", concat_list_filename="my_list.txt")
        assert cfg.output_dir == "/tmp/out"
        assert cfg.concat_list_filename == "my_list.txt"

    def test_new_fields_defaults(self):
        cfg = EditorConfig()
        assert cfg.video_codec == "libx264"
        assert cfg.audio_codec == "aac"
        assert cfg.crf == 23
        assert cfg.preset == "medium"
        assert cfg.audio_bitrate == "128k"
        assert cfg.pixel_format == "yuv420p"
        assert cfg.force_reencode is False
        assert cfg.auto_reencode is True
        assert cfg.normalize_audio is False
        assert cfg.add_transitions is False
        assert cfg.transition_duration == 0.5
        assert cfg.keyframe_accurate is True
        assert cfg.output_format == "mp4"
        assert cfg.log_ffmpeg_stderr is True
        assert cfg.ffmpeg_timeout == 300

    def test_encoding_custom(self):
        cfg = EditorConfig(crf=18, preset="slow", force_reencode=True)
        assert cfg.crf == 18
        assert cfg.preset == "slow"
        assert cfg.force_reencode is True


# ============================================================
# EditResult
# ============================================================

class TestEditResult:
    def test_defaults(self):
        result = EditResult(output_path="/tmp/out.mp4")
        assert result.output_path == "/tmp/out.mp4"
        assert result.segments == []
        assert result.source == "multimodal"

    def test_with_segments(self):
        seg_info = [{"start_time": 1.0, "end_time": 3.0, "score": 0.9}]
        result = EditResult(output_path="/tmp/out.mp4", segments=seg_info, source="multimodal")
        assert result.source == "multimodal"
        assert len(result.segments) == 1


# ============================================================
# Codec consistency
# ============================================================

class TestCodecConsistency:
    def test_h264_aac_mp4(self):
        probe = VideoProbe(streams=[
            StreamInfo(codec_type="video", codec_name="h264", index=0),
            StreamInfo(codec_type="audio", codec_name="aac", index=1),
        ])
        assert _check_codec_consistency(probe, "mp4") is True

    def test_vp9_opus_to_mp4(self):
        probe = VideoProbe(streams=[
            StreamInfo(codec_type="video", codec_name="vp9", index=0),
            StreamInfo(codec_type="audio", codec_name="opus", index=1),
        ])
        assert _check_codec_consistency(probe, "mp4") is False

    def test_h264_to_mkv(self):
        probe = VideoProbe(streams=[
            StreamInfo(codec_type="video", codec_name="h264", index=0),
        ])
        assert _check_codec_consistency(probe, "mkv") is True

    def test_vp9_opus_to_webm(self):
        probe = VideoProbe(streams=[
            StreamInfo(codec_type="video", codec_name="vp9", index=0),
            StreamInfo(codec_type="audio", codec_name="opus", index=1),
        ])
        assert _check_codec_consistency(probe, "webm") is True

    def test_h264_to_webm(self):
        probe = VideoProbe(streams=[
            StreamInfo(codec_type="video", codec_name="h264", index=0),
        ])
        assert _check_codec_consistency(probe, "webm") is False

    def test_no_video_stream(self):
        probe = VideoProbe(streams=[])
        assert _check_codec_consistency(probe, "mp4") is True


# ============================================================
# Keyframe alignment
# ============================================================

class TestKeyframeAlignment:
    def test_start_direction(self):
        # GOP=30 frames, fps=30 → gop_duration=1.0s
        result = VideoEditor._align_cut_to_keyframe(2.3, 30, 30.0, "start")
        assert result == 2.0

    def test_end_direction(self):
        result = VideoEditor._align_cut_to_keyframe(2.3, 30, 30.0, "end")
        assert result == 3.0

    def test_gop_zero_noop(self):
        result = VideoEditor._align_cut_to_keyframe(2.3, 0, 30.0, "start")
        assert result == 2.3

    def test_fps_zero_noop(self):
        result = VideoEditor._align_cut_to_keyframe(2.3, 30, 0.0, "start")
        assert result == 2.3


# ============================================================
# Build encode args
# ============================================================

class TestBuildEncodeArgs:
    def test_defaults(self):
        editor = VideoEditor()
        args = editor._build_encode_args()
        assert "-c:v" in args
        assert "libx264" in args
        assert "-crf" in args
        assert "23" in args
        assert "-preset" in args
        assert "medium" in args
        assert "-c:a" in args
        assert "aac" in args
        assert "-b:a" in args
        assert "128k" in args
        assert "-pix_fmt" in args
        assert "yuv420p" in args
        assert "-movflags" in args
        assert "+faststart" in args

    def test_custom(self):
        cfg = EditorConfig(video_codec="libx265", crf=18, preset="slow", audio_bitrate="192k")
        editor = VideoEditor(cfg)
        args = editor._build_encode_args()
        assert "libx265" in args
        assert "18" in args
        assert "slow" in args
        assert "192k" in args


# ============================================================
# Loudnorm JSON parsing
# ============================================================

class TestParseLoudnormJSON:
    def test_valid_output(self):
        stderr = '\n{"input_i" : "-23.05", "input_tp" : "-3.62", "input_lra" : "5.10", "input_thresh" : "-33.77", "target_offset" : "0.05"}\n'
        result = VideoEditor._parse_loudnorm_json(stderr)
        assert result is not None
        assert result["input_i"] == -23.05
        assert result["input_tp"] == -3.62
        assert result["input_lra"] == 5.10
        assert result["input_thresh"] == -33.77

    def test_no_json(self):
        result = VideoEditor._parse_loudnorm_json("just some random text")
        assert result is None

    def test_incomplete_json(self):
        result = VideoEditor._parse_loudnorm_json('{"input_i": "-23.0"}')
        assert result is None


# ============================================================
# _run_ffmpeg logging
# ============================================================

class TestRunFFmpegLogging:
    def test_stderr_logged_on_success(self, mocker, caplog):
        import logging
        caplog.set_level(logging.DEBUG)
        mock_run = mocker.patch("subprocess.run")
        mock_run.return_value = mocker.MagicMock(
            returncode=0, stderr="info output", stdout="",
            spec=subprocess.CompletedProcess,
        )
        editor = VideoEditor()
        editor._run_ffmpeg(["ffmpeg", "-i", "test.mp4"])
        assert "info output" in caplog.text

    def test_stderr_in_exception(self, mocker):
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["ffmpeg"], stderr="Something went wrong",
        )
        editor = VideoEditor()
        with pytest.raises(RuntimeError, match="Something went wrong"):
            editor._run_ffmpeg(["ffmpeg", "-i", "test.mp4"])

    def test_timeout_converted_to_runtime_error(self, mocker):
        mock_run = mocker.patch("subprocess.run")
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=5)
        editor = VideoEditor()
        with pytest.raises(RuntimeError, match="超时"):
            editor._run_ffmpeg(["ffmpeg", "-i", "test.mp4"], timeout=5)


# ============================================================
# Probe (with mock)
# ============================================================

class TestProbe:
    def test_probe_success(self, mocker):
        ffprobe_output = {
            "streams": [
                {"index": 0, "codec_type": "video", "codec_name": "h264",
                 "width": 1920, "height": 1080, "r_frame_rate": "30/1",
                 "duration": "10.0", "pix_fmt": "yuv420p"},
                {"index": 1, "codec_type": "audio", "codec_name": "aac",
                 "duration": "10.0", "sample_rate": "48000", "channels": 2},
            ],
            "format": {"duration": "10.0", "bit_rate": "5000000", "format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
        }
        mock_result = mocker.MagicMock(stdout=json.dumps(ffprobe_output), stderr="")
        mocker.patch.object(VideoEditor, "_run_ffmpeg", return_value=mock_result)
        mocker.patch.object(VideoEditor, "_get_gop_size", return_value=30)

        editor = VideoEditor()
        probe = editor.probe("/tmp/video.mp4")

        assert probe.duration == 10.0
        assert probe.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
        assert probe.gop_size == 30
        assert probe.video_stream is not None
        assert probe.video_stream.codec_name == "h264"
        assert probe.video_stream.width == 1920
        assert probe.video_stream.fps == 30.0
        assert probe.audio_stream is not None
        assert probe.audio_stream.codec_name == "aac"

    def test_probe_failure_returns_empty(self, mocker):
        mocker.patch.object(VideoEditor, "_run_ffmpeg", side_effect=RuntimeError("fail"))
        editor = VideoEditor()
        probe = editor.probe("/tmp/video.mp4")
        assert probe.path == "/tmp/video.mp4"
        assert probe.duration == 0.0
        assert probe.streams == []


# ============================================================
# VideoEditor FFmpeg integration tests
# ============================================================

class TestVideoEditorFFmpeg:
    def test_edit_with_ffmpeg_success(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        editor = VideoEditor(EditorConfig(output_dir=str(tmp_path)))

        segments = [
            {"start_time": 0.0, "end_time": 5.0, "score": 0.9},
            {"start_time": 10.0, "end_time": 15.0, "score": 0.7},
        ]

        result = editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        assert result.source == "multimodal"
        assert len(result.segments) == 2
        assert result.output_path.startswith(str(tmp_path))
        assert "highlight_reel_" in result.output_path
        assert mock_run.call_count >= 4  # ffprobe + 2 cuts + 1 concat

    def test_edit_with_ffmpeg_empty_segments_raises(self):
        editor = VideoEditor()
        with pytest.raises(ValueError, match="segments 为空"):
            editor.edit_with_ffmpeg("/tmp/video.mp4", [])

    def test_edit_with_ffmpeg_single_segment(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        editor = VideoEditor(EditorConfig(output_dir=str(tmp_path)))

        segments = [{"start_time": 5.0, "end_time": 10.0, "score": 0.8}]

        result = editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        assert result.source == "multimodal"
        assert len(result.segments) == 1
        assert mock_run.call_count >= 3  # ffprobe + 1 cut + 1 concat

    def test_edit_with_ffmpeg_cleanup_temp_files(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        editor = VideoEditor(EditorConfig(output_dir=str(tmp_path)))

        segments = [{"start_time": 0.0, "end_time": 3.0, "score": 0.9}]

        editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        clip_files = list(Path(tmp_path).glob("_clip_*.mp4"))
        assert len(clip_files) == 0

    def test_force_reencode(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        cfg = EditorConfig(output_dir=str(tmp_path), force_reencode=True)
        editor = VideoEditor(cfg)
        segments = [{"start_time": 0.0, "end_time": 5.0, "score": 0.9}]

        editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        # Verify re-encode args present
        all_args = []
        for call in mock_run.call_args_list:
            all_args.extend(str(a) for a in call[0][0])
        all_str = " ".join(all_args)
        assert "-crf" in all_str
        assert "-preset" in all_str

    def test_output_format_respected(self, mocker, tmp_path):
        mock_run = mocker.patch("subprocess.run")
        cfg = EditorConfig(output_dir=str(tmp_path), output_format="mkv")
        editor = VideoEditor(cfg)
        segments = [{"start_time": 0.0, "end_time": 5.0, "score": 0.9}]

        result = editor.edit_with_ffmpeg("/tmp/video.mp4", segments)

        assert result.output_path.endswith(".mkv")
