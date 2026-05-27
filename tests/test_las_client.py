import pytest

from src.las_client import LasClient, LasConfig


class TestLasConfig:
    def test_default_config(self):
        config = LasConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.poll_interval == 5.0
        assert config.poll_timeout == 600.0
        assert config.max_retries == 3
        assert config.operator_id == "las_video_edit"
        assert config.operator_version == "v1"

    def test_custom_config(self):
        config = LasConfig(
            api_key="custom-key",
            poll_interval=5.0,
            poll_timeout=300.0,
            max_retries=1,
        )
        assert config.poll_interval == 5.0
        assert config.poll_timeout == 300.0


class TestLasClient:
    def test_init_without_api_key_raises(self):
        config = LasConfig(api_key="")
        with pytest.raises(ValueError, match="LAS_API_KEY"):
            LasClient(config)

    def test_submit_success(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {"task_id": "task-123", "status": "pending"}
        mock_resp.raise_for_status = mocker.MagicMock()
        mock_post = mocker.patch("httpx.post", return_value=mock_resp)

        client = LasClient(LasConfig(api_key="test-key"))
        result = client.submit("las_video_edit", {"video_url": "http://example.com/v.mp4"})

        assert result["task_id"] == "task-123"
        assert result["status"] == "pending"
        mock_post.assert_called_once()

    def test_submit_retry_on_error(self, mocker):
        import httpx

        mock_err_resp = mocker.MagicMock()
        mock_err_resp.status_code = 500
        mock_err_resp.text = "server error"
        mock_http_err = httpx.HTTPStatusError(
            "server error", request=mocker.MagicMock(), response=mock_err_resp
        )

        mock_ok = mocker.MagicMock()
        mock_ok.json.return_value = {"task_id": "task-456"}
        mock_ok.raise_for_status = mocker.MagicMock()

        mock_post = mocker.patch("httpx.post", side_effect=[mock_http_err, mock_ok])
        mocker.patch("time.sleep")

        client = LasClient(LasConfig(api_key="test-key"))
        result = client.submit("las_video_edit", {})

        assert result["task_id"] == "task-456"
        assert mock_post.call_count == 2

    def test_poll_success(self, mocker):
        mock_resp = mocker.MagicMock()
        mock_resp.json.return_value = {"task_id": "task-123", "status": "success", "output": {}}
        mock_resp.raise_for_status = mocker.MagicMock()
        mock_post = mocker.patch("httpx.post", return_value=mock_resp)

        client = LasClient(LasConfig(api_key="test-key"))
        result = client.poll("task-123")

        assert result["status"] == "success"
        mock_post.assert_called_once()

    def test_wait_for_completion_success(self, mocker):
        mock_pending = mocker.MagicMock()
        mock_pending.json.return_value = {"task_id": "task-123", "status": "processing"}
        mock_pending.raise_for_status = mocker.MagicMock()

        mock_done = mocker.MagicMock()
        mock_done.json.return_value = {"task_id": "task-123", "status": "success", "output": {"url": "..."}}
        mock_done.raise_for_status = mocker.MagicMock()

        mocker.patch("httpx.post", side_effect=[mock_pending, mock_done])
        mocker.patch("time.time", side_effect=[0, 0, 0, 100])
        mocker.patch("time.sleep")

        client = LasClient(LasConfig(api_key="test-key"))
        result = client.wait_for_completion("task-123")

        assert result["status"] == "success"
        assert result["output"] == {"url": "..."}

    def test_wait_for_completion_failed(self, mocker):
        mock_failed = mocker.MagicMock()
        mock_failed.json.return_value = {
            "task_id": "task-123",
            "status": "failed",
            "error": "处理失败",
        }
        mock_failed.raise_for_status = mocker.MagicMock()

        mocker.patch("httpx.post", return_value=mock_failed)
        mocker.patch("time.time", side_effect=[0, 100])
        mocker.patch("time.sleep")

        client = LasClient(LasConfig(api_key="test-key"))
        with pytest.raises(RuntimeError, match="LAS 任务失败"):
            client.wait_for_completion("task-123")

    def test_wait_for_completion_timeout(self, mocker):
        mock_pending = mocker.MagicMock()
        mock_pending.json.return_value = {"task_id": "task-123", "status": "processing"}
        mock_pending.raise_for_status = mocker.MagicMock()

        mocker.patch("httpx.post", return_value=mock_pending)
        mocker.patch("time.time", side_effect=[0, 1000])
        mocker.patch("time.sleep")

        client = LasClient(LasConfig(api_key="test-key", poll_timeout=10.0))
        with pytest.raises(TimeoutError, match="超时"):
            client.wait_for_completion("task-123")
