import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest
import tempfile

from src.video_fetcher import (
    ArkFileSource,
    LocalFileSource,
    TosSource,
    UrlSource,
    VideoFetcher,
    VideoMetadata,
)


class TestLocalFileSource:
    def test_resolve_existing_file(self, tmp_path):
        video = tmp_path / "test.mp4"
        video.touch()
        source = LocalFileSource(str(video))
        assert source.resolve() == str(video.absolute())

    def test_resolve_missing_file_raises(self):
        source = LocalFileSource("/nonexistent/video.mp4")
        with pytest.raises(FileNotFoundError, match="视频文件不存在"):
            source.resolve()


class TestUrlSource:
    def test_resolve_success(self, mocker):
        mock_run = mocker.patch("subprocess.run")
        mocker.patch("tempfile.mkdtemp", return_value="/tmp/video_dl_abc")
        mocker.patch.object(
            Path, "glob", return_value=[Path("/tmp/video_dl_abc/title.mp4")]
        )
        # mock _convert_to_mp4 to avoid ffmpeg call
        mocker.patch(
            "src.video_fetcher._convert_to_mp4",
            return_value="/tmp/video_dl_abc/title.mp4",
        )

        source = UrlSource("https://example.com/video")
        result = source.resolve()

        assert result.replace("\\", "/") == "/tmp/video_dl_abc/title.mp4"
        mock_run.assert_called_once()

    def test_resolve_download_failure(self, mocker):
        mocker.patch("tempfile.mkdtemp", return_value="/tmp/video_dl_abc")
        mocker.patch(
            "subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "yt-dlp", stderr="download error"),
        )

        source = UrlSource("https://example.com/video")
        with pytest.raises(RuntimeError, match="视频下载失败"):
            source.resolve()

    def test_resolve_no_output_file(self, mocker):
        mocker.patch("subprocess.run")
        mocker.patch("tempfile.mkdtemp", return_value="/tmp/video_dl_abc")
        mocker.patch.object(Path, "glob", return_value=[])

        source = UrlSource("https://example.com/video")
        with pytest.raises(RuntimeError, match="未找到输出文件"):
            source.resolve()


class TestTosSource:
    def test_resolve_missing_config_raises(self):
        source = TosSource("tos://bucket/video.mp4")
        with pytest.raises(RuntimeError, match="TOS 配置不完整"):
            source.resolve()

    def test_parse_tos_path(self):
        source = TosSource("tos://mybucket/path/to/video.mp4")
        bucket, key = source._parse_tos_path()
        assert bucket == "mybucket"
        assert key == "path/to/video.mp4"

    def test_parse_tos_path_invalid(self):
        source = TosSource("tos://nokey")
        with pytest.raises(ValueError, match="无效的 TOS 路径格式"):
            source._parse_tos_path()

    def test_resolve_success(self, mocker, tmp_path):
        mocker.patch("tempfile.mkdtemp", return_value=str(tmp_path))
        mocker.patch.object(TosSource, "_parse_tos_path", return_value=("bucket", "key.mp4"))
        mock_tos_client = mocker.patch("tos.TosClientV2")
        mock_instance = mock_tos_client.return_value
        mock_instance.get_object_to_file = mocker.MagicMock()

        source = TosSource("tos://bucket/key.mp4", access_key="ak", secret_key="sk")
        result = source.resolve()
        assert result.endswith("key.mp4")
        mock_instance.get_object_to_file.assert_called_once()


class TestVideoFetcher:
    def test_fetch_local_file(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake mp4 content")

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 300,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }[prop]
        mock_cap.read.side_effect = (
            [(True, np.zeros((1080, 1920, 3), dtype=np.uint8))] * 5 + [(False, None)]
        )

        mocker.patch("cv2.VideoCapture", return_value=mock_cap)
        mocker.patch("src.video_fetcher._convert_to_mp4", return_value=str(video))

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        meta = fetcher.fetch(LocalFileSource(str(video)))

        assert meta.duration == 10.0
        assert meta.fps == 30.0
        assert meta.width == 1920
        assert meta.height == 1080

    def test_fetch_cannot_open_video(self, mocker, tmp_path):
        video = tmp_path / "bad.mp4"
        video.write_bytes(b"fake mp4 content")

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = False
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="无法打开视频"):
            fetcher.fetch(LocalFileSource(str(video)))

    def test_convert_to_mp4(self, mocker, tmp_path):
        mocker.patch("subprocess.run")
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        result = fetcher._convert_to_mp4("/tmp/input.mov")
        assert result.endswith(".mp4")

    def test_fetch_empty_file_raises(self, tmp_path):
        video = tmp_path / "empty.mp4"
        video.write_text("")

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="视频文件为空"):
            fetcher.fetch(LocalFileSource(str(video)))

    def test_fetch_non_video_file_raises(self, mocker, tmp_path):
        video = tmp_path / "fake.mp4"
        video.write_text("this is not a video file")

        mock_cap = mocker.MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 0.0,
            cv2.CAP_PROP_FRAME_COUNT: 0,
            cv2.CAP_PROP_FRAME_WIDTH: 0,
            cv2.CAP_PROP_FRAME_HEIGHT: 0,
        }[prop]
        mocker.patch("cv2.VideoCapture", return_value=mock_cap)

        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(ValueError, match="无法解析视频参数|视频文件中无视频帧"):
            fetcher.fetch(LocalFileSource(str(video)))

    def test_convert_to_mp4_failure_raises(self, mocker, tmp_path):
        mocker.patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "ffmpeg", stderr="conversion error"))
        fetcher = VideoFetcher(output_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="视频转码失败"):
            fetcher._convert_to_mp4("/tmp/input.mov")


class TestArkFileSource:
    def test_resolve_success(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake mp4 content")

        mock_client = mocker.MagicMock()
        mock_client.upload_file.return_value = {"download_url": "https://ark-cn-beijing.volces.com/dl/test_video"}
        mocker.patch("src.ark_client.ArkClient", return_value=mock_client)

        source = ArkFileSource(str(video))
        result = source.resolve()

        assert result == "https://ark-cn-beijing.volces.com/dl/test_video"
        mock_client.upload_file.assert_called_once()

    def test_resolve_file_not_found(self):
        source = ArkFileSource("/nonexistent/video.mp4")
        with pytest.raises(FileNotFoundError, match="视频文件不存在"):
            source.resolve()

    def test_resolve_no_download_url(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake mp4 content")

        mock_client = mocker.MagicMock()
        mock_client.upload_file.return_value = {"id": "file-123"}
        mocker.patch("src.ark_client.ArkClient", return_value=mock_client)

        source = ArkFileSource(str(video))
        with pytest.raises(RuntimeError, match="未返回 download_url"):
            source.resolve()
