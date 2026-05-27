from pathlib import Path

import pytest

from src.video_editor import EditResult, EditorConfig, VideoEditor


class TestEditorConfig:
    def test_defaults(self):
        cfg = EditorConfig()
        assert cfg.las_operator_id == "las_video_edit"
        assert cfg.las_mode == "detail"

    def test_custom(self):
        cfg = EditorConfig(
            output_dir="/tmp/out",
            las_mode="normal",
        )
        assert cfg.output_dir == "/tmp/out"
        assert cfg.las_mode == "normal"


class TestEditResult:
    def test_defaults(self):
        result = EditResult(output_path="/tmp/out.mp4")
        assert result.output_path == "/tmp/out.mp4"
        assert result.segments == []
        assert result.source == "las"

    def test_with_segments(self):
        seg_info = [{"start_time": 1.0, "end_time": 3.0, "score": 0.9}]
        result = EditResult(
            output_path="/tmp/out.mp4",
            segments=seg_info,
            source="las",
        )
        assert result.source == "las"
        assert len(result.segments) == 1


class TestVideoEditor:
    def test_edit_e2e_success(self, mocker, monkeypatch):
        monkeypatch.setenv("LAS_API_KEY", "test-key")

        mock_e2e = mocker.patch.object(
            VideoEditor, "_edit_with_las_e2e",
            return_value=EditResult(
                output_path="tos://bucket/output/clip_001.mp4",
                segments=[
                    {"start_time": 125.0, "end_time": 132.0, "score": 0.9, "label": "打斗高光"},
                    {"start_time": 161.0, "end_time": 170.0, "score": 0.9, "label": "打斗高光"},
                ],
                source="las",
            ),
        )

        editor = VideoEditor()
        result = editor.edit_e2e("/tmp/video.mp4", "剪辑打斗画面")

        assert result.source == "las"
        assert result.output_path == "tos://bucket/output/clip_001.mp4"
        assert len(result.segments) == 2
        mock_e2e.assert_called_once()

    def test_edit_e2e_raises_on_error(self, mocker):
        mocker.patch.object(
            VideoEditor, "_edit_with_las_e2e",
            side_effect=RuntimeError("LAS 不可用"),
        )

        editor = VideoEditor()
        with pytest.raises(RuntimeError, match="LAS 不可用"):
            editor.edit_e2e("/tmp/video.mp4", "剪辑")

    def test_las_e2e_submit_no_task_id(self, mocker, monkeypatch):
        monkeypatch.setenv("LAS_API_KEY", "test-key")

        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {"status": "submitted"}
        mock_resp.raise_for_status = mocker.MagicMock()
        mocker.patch("httpx.post", return_value=mock_resp)

        editor = VideoEditor()
        with pytest.raises(RuntimeError, match="LAS 未返回 task_id"):
            editor._edit_with_las_e2e("https://example.com/video.mp4", "剪辑")

    def test_las_client_lazy_init(self, monkeypatch):
        monkeypatch.setenv("LAS_API_KEY", "test-key")
        editor = VideoEditor()
        assert editor._las_client is None
        client = editor.las_client
        assert client is not None
        assert editor._las_client is not None


class TestVideoEditorResolveVideoUrl:
    def test_url_passthrough(self):
        editor = VideoEditor()
        url = "https://example.com/video.mp4"
        result = editor._resolve_video_url(url)
        assert result == url

    def test_http_url_passthrough(self):
        editor = VideoEditor()
        url = "http://example.com/video.mp4"
        result = editor._resolve_video_url(url)
        assert result == url

    def test_local_path_uses_tos_upload(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video content")

        mock_put = mocker.patch("tos.TosClientV2.put_object_from_file")

        editor = VideoEditor()
        result = editor._resolve_video_url(str(video))

        assert result.startswith("tos://")
        assert "/input/test/" in result
        assert result.endswith("/test.mp4")
        mock_put.assert_called_once()
