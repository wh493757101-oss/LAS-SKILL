import os

import pytest

from src.ark_client import ArkClient, ArkConfig


class TestArkConfig:
    def test_default_config(self):
        config = ArkConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.base_url == "https://ark.cn-beijing.volces.com/api/v3"
        assert config.model == "doubao-seed-2-0-pro"
        assert config.max_retries == 3

    def test_custom_config(self):
        config = ArkConfig(
            api_key="custom-key",
            base_url="https://custom.volces.com/api/v3",
            model="doubao-vision-pro",
            max_retries=5,
            timeout=60.0,
        )
        assert config.model == "doubao-vision-pro"
        assert config.max_retries == 5
        assert config.timeout == 60.0


class TestArkClient:
    def test_init_without_api_key_raises(self):
        config = ArkConfig(api_key="")
        with pytest.raises(ValueError, match="ARK_HIGHLIGHT_API_KEY"):
            ArkClient(config)

    def test_chat_success(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "hello"}}],
        }
        mock_resp.raise_for_status = mocker.MagicMock()
        mock_post = mocker.patch("httpx.post", return_value=mock_resp)

        client = ArkClient(ArkConfig(api_key="test-key"))
        result = client.chat([{"role": "user", "content": "hi"}])

        assert result["choices"][0]["message"]["content"] == "hello"
        mock_post.assert_called_once()

    def test_chat_retry_on_429(self, mocker):
        import httpx

        mock_429_resp = mocker.MagicMock()
        mock_429_resp.status_code = 429
        mock_429_resp.text = "rate limited"
        mock_429_err = httpx.HTTPStatusError(
            "rate limited", request=mocker.MagicMock(), response=mock_429_resp
        )

        mock_ok = mocker.MagicMock()
        mock_ok.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        mock_ok.raise_for_status = mocker.MagicMock()

        mock_post = mocker.patch("httpx.post", side_effect=[mock_429_err, mock_ok])
        mocker.patch("time.sleep")

        client = ArkClient(ArkConfig(api_key="test-key"))
        result = client.chat([{"role": "user", "content": "hi"}])

        assert result["choices"][0]["message"]["content"] == "ok"
        assert mock_post.call_count == 2

    def test_chat_with_response_format(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"score": 0.9}'}}],
        }
        mock_resp.raise_for_status = mocker.MagicMock()
        mock_post = mocker.patch("httpx.post", return_value=mock_resp)

        client = ArkClient(ArkConfig(api_key="test-key"))
        client.chat(
            [{"role": "user", "content": "analyze"}],
            response_format={"type": "json_object"},
        )

        call_args = mock_post.call_args[1]["json"]
        assert call_args["response_format"] == {"type": "json_object"}

    def test_chat_with_images(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "image analysis result"}}],
        }
        mock_resp.raise_for_status = mocker.MagicMock()
        mock_post = mocker.patch("httpx.post", return_value=mock_resp)

        mock_image = mocker.MagicMock()
        mock_image.mode = "RGB"
        mock_open = mocker.patch("PIL.Image.open", return_value=mock_image)
        mocker.patch("base64.b64encode", return_value=b"fake_base64")

        client = ArkClient(ArkConfig(api_key="test-key"))
        result = client.chat_with_images("describe", ["/fake/path.jpg"])

        assert "choices" in result
        mock_open.assert_called_once_with("/fake/path.jpg")

    def test_extract_json_from_string(self):
        client = ArkClient(ArkConfig(api_key="test-key"))
        response = {"choices": [{"message": {"content": '{"key": "value"}'}}]}
        result = client.extract_json(response)
        assert result == {"key": "value", "_usage": {}}

    def test_extract_json_from_code_block(self):
        client = ArkClient(ArkConfig(api_key="test-key"))
        response = {
            "choices": [
                {"message": {"content": '```json\n{"key": "value"}\n```'}}
            ]
        }
        result = client.extract_json(response)
        assert result == {"key": "value", "_usage": {}}

    def test_extract_json_from_dict_content(self):
        client = ArkClient(ArkConfig(api_key="test-key"))
        response = {"choices": [{"message": {"content": {"key": "value"}}}]}
        result = client.extract_json(response)
        assert result == {"key": "value", "_usage": {}}


class TestArkClientUploadFile:
    def test_upload_success(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video content")

        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {"download_url": "https://ark-cn-beijing.volces.com/dl/abc123"}
        mock_resp.raise_for_status = mocker.MagicMock()
        mock_post = mocker.patch("httpx.post", return_value=mock_resp)

        client = ArkClient(ArkConfig(api_key="test-key"))
        result = client.upload_file(str(video))

        assert result["download_url"] == "https://ark-cn-beijing.volces.com/dl/abc123"
        mock_post.assert_called_once()

    def test_upload_file_not_found(self):
        client = ArkClient(ArkConfig(api_key="test-key"))
        with pytest.raises(FileNotFoundError, match="文件不存在"):
            client.upload_file("/nonexistent/video.mp4")

    def test_upload_empty_file(self, tmp_path):
        video = tmp_path / "empty.mp4"
        video.write_text("")

        client = ArkClient(ArkConfig(api_key="test-key"))
        with pytest.raises(ValueError, match="文件为空"):
            client.upload_file(str(video))

    def test_upload_retry_on_429(self, mocker, tmp_path):
        import httpx

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video content")

        mock_429_resp = mocker.MagicMock()
        mock_429_resp.status_code = 429
        mock_429_resp.text = "rate limited"
        mock_429_err = httpx.HTTPStatusError(
            "rate limited", request=mocker.MagicMock(), response=mock_429_resp
        )

        mock_ok = mocker.MagicMock()
        mock_ok.json.return_value = {"download_url": "https://ark-cn-beijing.volces.com/dl/retry_ok"}
        mock_ok.raise_for_status = mocker.MagicMock()

        mock_post = mocker.patch("httpx.post", side_effect=[mock_429_err, mock_ok])
        mocker.patch("time.sleep")

        client = ArkClient(ArkConfig(api_key="test-key"))
        result = client.upload_file(str(video))

        assert result["download_url"] == "https://ark-cn-beijing.volces.com/dl/retry_ok"
        assert mock_post.call_count == 2

    def test_upload_http_error_raises(self, mocker, tmp_path):
        import httpx

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video content")

        mock_resp = mocker.MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        mock_err = httpx.HTTPStatusError(
            "internal error", request=mocker.MagicMock(), response=mock_resp
        )
        mocker.patch("httpx.post", side_effect=mock_err)

        client = ArkClient(ArkConfig(api_key="test-key"))
        with pytest.raises(RuntimeError, match="文件上传失败"):
            client.upload_file(str(video))


class TestArkClientChatWithVideo:
    def test_chat_with_video_success(self, mocker, tmp_path):
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video content")

        mock_chat = mocker.patch.object(
            ArkClient, "chat",
            return_value={"choices": [{"message": {"content": "video analysis"}}]},
        )

        client = ArkClient(ArkConfig(api_key="test-key"))
        result = client.chat_with_video("analyze this video", str(video))

        assert "choices" in result
        mock_chat.assert_called_once()
        call_args = mock_chat.call_args[1]
        messages = call_args["messages"]
        content = messages[0]["content"]
        assert content[0]["type"] == "video_url"
        assert content[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "analyze this video"

    def test_chat_with_video_file_not_found(self):
        client = ArkClient(ArkConfig(api_key="test-key"))
        with pytest.raises(FileNotFoundError):
            client.chat_with_video("prompt", "/nonexistent/video.mp4")
