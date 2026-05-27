import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class LasConfig:
    api_key: str = field(default_factory=lambda: os.getenv("LAS_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LAS_BASE_URL", "https://operator.las.cn-guangzhou.volces.com/api/v1"))
    operator_id: str = "las_video_edit"
    operator_version: str = "v1"
    poll_interval: float = 5.0
    poll_timeout: float = 600.0
    max_retries: int = 3


class LasClient:
    def __init__(self, config: LasConfig | None = None):
        self.config = config or LasConfig()
        if not self.config.api_key:
            raise ValueError("LAS_API_KEY 未设置，请通过环境变量或 LasConfig 提供")
        self.call_count: int = 0
        self.retry_count: int = 0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def submit(
        self,
        operator_id: str,
        task_input: dict[str, Any],
        operator_version: str = "v1",
    ) -> dict[str, Any]:
        last_error: str | None = None
        self.call_count += 1
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                self.retry_count += 1
            try:
                resp = httpx.post(
                    f"{self.config.base_url}/submit",
                    headers=self._headers(),
                    json={
                        "operator_id": operator_id,
                        "operator_version": operator_version,
                        "data": task_input,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text[:500]}"
                logger.error("LAS submit failed: %s", last_error)
                if e.response.status_code in (429, 500):
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
            except httpx.RequestError as e:
                last_error = str(e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                raise

        raise RuntimeError(f"LAS 任务提交失败（已重试 {self.config.max_retries} 次）: {last_error}")

    def poll(self, task_id: str) -> dict[str, Any]:
        resp = httpx.post(
            f"{self.config.base_url}/poll",
            headers=self._headers(),
            json={
                "task_id": task_id,
                "operator_id": self.config.operator_id,
                "operator_version": self.config.operator_version,
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_for_completion(self, task_id: str) -> dict[str, Any]:
        deadline = time.time() + self.config.poll_timeout
        while time.time() < deadline:
            result = self.poll(task_id)
            meta = result.get("metadata", {})
            status = meta.get("task_status", result.get("status", ""))
            if status in ("COMPLETED", TaskStatus.SUCCESS):
                return result
            if status in ("FAILED", TaskStatus.FAILED, "TIMEOUT"):
                raise RuntimeError(
                    f"LAS 任务失败: code={meta.get('business_code', '?')} msg={meta.get('error_msg', '未知错误')}"
                )
            time.sleep(self.config.poll_interval)

        raise TimeoutError(f"LAS 任务 {task_id} 超时（{self.config.poll_timeout}s）")
