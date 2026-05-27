import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ArkConfig:
    api_key: str = ""
    base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    model: str = "doubao-seed-2-0-pro"
    max_retries: int = 3
    timeout: float = 120.0
    upload_timeout: float = 300.0


def _encode_image(image_path: str) -> str:
    img = Image.open(image_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = BytesIO()
    ext = Path(image_path).suffix.lower()
    fmt = "JPEG" if ext in (".jpg", ".jpeg") else "PNG"
    img.save(buf, format=fmt)
    return f"data:image/{fmt.lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"


class ArkClient:
    def __init__(self, config: ArkConfig | None = None):
        self.config = config or ArkConfig()
        if not self.config.api_key:
            raise ValueError("ARK_HIGHLIGHT_API_KEY 未设置，请通过环境变量或 ArkConfig 提供")
        self.call_count: int = 0
        self.retry_count: int = 0

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            body["response_format"] = response_format

        last_error: str | None = None
        self.call_count += 1
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                self.retry_count += 1
            try:
                resp = httpx.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=self.config.timeout,
                )
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                if e.response.status_code == 429:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
            except httpx.RequestError as e:
                last_error = str(e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                raise

        raise RuntimeError(f"Ark API 调用失败（已重试 {self.config.max_retries} 次）: {last_error}")

    def chat_with_images(
        self,
        text: str,
        image_paths: list[str],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(path)},
            })

        messages = [{"role": "user", "content": content}]
        return self.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def chat_with_video(
        self,
        text: str,
        video_path: str,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}，请检查文件路径后重试")

        with open(path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode()

        video_url = f"data:video/mp4;base64,{video_b64}"
        content: list[dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": text},
        ]
        return self.chat(
            messages=[{"role": "user", "content": content}],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def chat_with_video_omni(
        self,
        text: str,
        video_url: str,
        model: str | None = None,
        modalities: list[str] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """DashScope Qwen-Omni 视频+音频调用（stream + modalities）。

        video_url 可以是 HTTPS URL 或 base64 data URI（data:video/mp4;base64,...）。
        返回格式与 chat() 一致：{"choices": [{"message": {"content": "..."}}], "usage": {...}}。
        """
        if modalities is None:
            modalities = ["text"]

        content: list[dict[str, Any]] = [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": text},
        ]
        messages = [{"role": "user", "content": content}]

        model_id = model or self.config.model
        body: dict[str, Any] = {
            "model": model_id,
            "messages": messages,
            "modalities": modalities,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if "audio" in modalities:
            body["audio"] = {"voice": "Tina", "format": "wav"}
        if response_format:
            body["response_format"] = response_format

        self.call_count += 1
        last_error: str | None = None
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                self.retry_count += 1
            try:
                collected_text: str = ""
                collected_usage: dict[str, int] = {}
                with httpx.stream(
                    "POST",
                    f"{self.config.base_url}/chat/completions",
                    headers=self._headers(),
                    json=body,
                    timeout=self.config.timeout,
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if delta.get("content"):
                            collected_text += delta["content"]
                        usage = chunk.get("usage", {})
                        if usage:
                            collected_usage = usage
                return {
                    "choices": [{"message": {"content": collected_text, "role": "assistant"}}],
                    "usage": collected_usage,
                }
            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                if e.response.status_code == 429:
                    time.sleep(min(2 ** attempt, 30))
                    continue
                raise
            except httpx.RequestError as e:
                last_error = str(e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(1)
                    continue
                raise

        raise RuntimeError(f"Ark/Omni API 调用失败（已重试 {self.config.max_retries} 次）: {last_error}")

    def extract_json(self, response: dict[str, Any]) -> dict[str, Any]:
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = response.get("usage", {})
        if isinstance(content, dict):
            result = dict(content)
        elif isinstance(content, str):
            content = content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            result = json.loads(content)
        else:
            result = {}
        result["_usage"] = usage
        return result

    def upload_file(self, file_path: str) -> dict[str, Any]:
        """上传文件到 Ark Files API，返回包含 download_url 的 file object。

        Files API 文档: https://www.volcengine.com/docs/82379/1870405
        返回的 download_url 为 HTTPS 预签名链接，24 小时有效。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}，请检查文件路径后重试")
        if path.stat().st_size == 0:
            raise ValueError(f"文件为空（0 字节）: {file_path}，请检查文件后重试")

        with open(file_path, "rb") as f:
            last_error: str | None = None
            for attempt in range(self.config.max_retries):
                f.seek(0)
                try:
                    resp = httpx.post(
                        f"{self.config.base_url}/files",
                        headers={"Authorization": f"Bearer {self.config.api_key}"},
                        files={"file": (path.name, f, "application/octet-stream")},
                        data={"purpose": "user_data"},
                        timeout=self.config.upload_timeout,
                    )
                    resp.raise_for_status()
                    return resp.json()
                except httpx.HTTPStatusError as e:
                    last_error = f"HTTP {e.response.status_code}: {e.response.text}"
                    if e.response.status_code == 429:
                        time.sleep(min(2 ** attempt, 30))
                        continue
                    raise RuntimeError(f"文件上传失败，请稍后重试: {last_error}") from e
                except httpx.RequestError as e:
                    last_error = str(e)
                    if attempt < self.config.max_retries - 1:
                        time.sleep(1)
                        continue
                    raise RuntimeError(f"文件上传失败，请稍后重试: {last_error}") from e

        raise RuntimeError(
            f"文件上传失败（已重试 {self.config.max_retries} 次），请稍后重试: {last_error}"
        )
