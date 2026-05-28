import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .las_client import LasClient, LasConfig

logger = logging.getLogger(__name__)


@dataclass
class EditorConfig:
    output_dir: str = ""
    las_operator_id: str = field(default_factory=lambda: os.environ.get("LAS_OPERATOR_ID", "las_video_edit"))
    las_operator_version: str = "v1"
    output_tos_path: str = ""
    las_mode: str = "detail"


@dataclass
class EditTiming:
    """LAS 剪辑各阶段耗时（秒）。"""
    upload: float = 0.0
    las_inference: float = 0.0
    clip_export: float = 0.0


@dataclass
class EditResult:
    output_path: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    source: str = "las"
    session_tos_path: str = ""
    timing: EditTiming = field(default_factory=EditTiming)


class VideoEditor:
    def __init__(
        self,
        config: EditorConfig | None = None,
        las_client: LasClient | None = None,
    ):
        self.config = config or EditorConfig()
        self._las_client = las_client

    @property
    def las_client(self) -> LasClient:
        if self._las_client is None:
            self._las_client = LasClient()
        return self._las_client

    def edit_e2e(
        self,
        video_path: str,
        description: str,
    ) -> EditResult:
        """LAS 端到端：直接传用户需求给 LAS，由 LAS 完成识别+剪辑。失败直接抛出异常。"""
        return self._edit_with_las_e2e(video_path, description)

    def _edit_with_las_e2e(
        self,
        video_path: str,
        description: str,
    ) -> EditResult:
        t_upload_start = time.time()
        video_url = self._resolve_video_url(video_path)
        t_upload = time.time() - t_upload_start

        base_tos = self.config.output_tos_path or os.environ.get("TOS_OUTPUT_PATH", "")
        session_name = Path(video_path).stem + "_" + time.strftime("%Y%m%d_%H%M%S")
        output_tos_path = str(Path(base_tos.rstrip("/")) / session_name) + "/"
        task_input: dict[str, Any] = {
            "video_url": video_url,
            "task_description": description,
            "output_tos_path": output_tos_path,
            "mode": self.config.las_mode,
        }

        logger.info("LAS e2e submit: operator=%s mode=%s", self.config.las_operator_id, self.config.las_mode)

        t_infer_start = time.time()
        result = self.las_client.submit(
            self.config.las_operator_id,
            task_input,
            operator_version=self.config.las_operator_version,
        )
        task_id = result.get("metadata", {}).get("task_id", result.get("task_id", ""))
        if not task_id:
            raise RuntimeError("LAS 未返回 task_id")

        final = self.las_client.wait_for_completion(task_id)
        t_inference = time.time() - t_infer_start

        t_export_start = time.time()
        data = final.get("data", {})
        clips = data.get("clips", [])

        valid_clips = [
            c for c in clips
            if c.get("clip_url") and c.get("file_size", 0) > 1024
        ]

        output_url = ""
        segments: list[dict[str, Any]] = []
        for c in valid_clips:
            if not output_url:
                output_url = c["clip_url"]
            start_sec = _parse_time_to_seconds(c.get("start_time", "00:00:00"))
            end_sec = _parse_time_to_seconds(c.get("end_time", "00:00:00"))
            segments.append({
                "start_time": start_sec,
                "end_time": end_sec,
                "score": c.get("confidence", 0.5),
                "label": c.get("description", ""),
                "clip_url": c.get("clip_url", ""),
            })

        if not output_url:
            output_url = final.get("output", {}).get("url", "")

        t_export = time.time() - t_export_start

        return EditResult(
            output_path=output_url,
            segments=segments,
            source="las",
            session_tos_path=output_tos_path,
            timing=EditTiming(upload=t_upload, las_inference=t_inference, clip_export=t_export),
        )

    def _resolve_video_url(self, video_path: str) -> str:
        if video_path.startswith(("http://", "https://", "tos://")):
            return video_path

        import os as _os
        import tos as _tos
        from pathlib import Path as _Path

        _ak = _os.environ.get("TOS_ACCESS_KEY", "")
        _sk = _os.environ.get("TOS_SECRET_KEY", "")
        _bucket = "arkclaw-tos-2124145136-cn-guangzhou"
        _base_prefix = "arkclaw-tos-ci-yemqjzxa0w9t6r1y3a0v-lk0rj/video-highlight-bucket"

        _client = _tos.TosClientV2(_ak, _sk, "tos-cn-guangzhou.volces.com", "cn-guangzhou")
        _filename = _Path(video_path).name
        _folder = _Path(video_path).stem
        _tos_key = f"{_base_prefix}/input/{_folder}/{_filename}"
        _client.put_object_from_file(_bucket, _tos_key, video_path)
        logger.info("视频已上传到 TOS: tos://%s/%s", _bucket, _tos_key)
        return f"tos://{_bucket}/{_tos_key}"

def _parse_time_to_seconds(t: str) -> float:
    """Parse HH:MM:SS or HH:MM:SS.ss to seconds."""
    parts = t.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(t)
