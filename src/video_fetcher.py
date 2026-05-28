import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2

logger = logging.getLogger(__name__)

_FFMPEG_PATH: str | None = None
_YT_DLP_PATH: str | None = None


def _get_ffmpeg() -> str:
    global _FFMPEG_PATH
    if _FFMPEG_PATH:
        return _FFMPEG_PATH
    for candidate in [
        shutil.which("ffmpeg") or shutil.which("ffmpeg.exe") or "",
    ]:
        if candidate and Path(candidate).exists():
            _FFMPEG_PATH = candidate
            return candidate
    try:
        import imageio_ffmpeg
        exe: str | None = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            _FFMPEG_PATH = exe
            return _FFMPEG_PATH
    except ImportError:
        pass
    _FFMPEG_PATH = "ffmpeg"
    return _FFMPEG_PATH


def _get_ytdlp() -> str:
    global _YT_DLP_PATH
    if _YT_DLP_PATH:
        return _YT_DLP_PATH
    for candidate in [
        shutil.which("yt-dlp") or shutil.which("yt-dlp.exe") or "",
    ]:
        if candidate and Path(candidate).exists():
            _YT_DLP_PATH = candidate
            return candidate
    # 搜索用户级 Python Scripts 目录
    try:
        import site
        user_scripts = Path(site.getusersitepackages()).parent / "Scripts"
        ytdlp_exe = user_scripts / "yt-dlp.exe"
        if ytdlp_exe.exists():
            _YT_DLP_PATH = str(ytdlp_exe)
            return _YT_DLP_PATH
    except Exception:
        pass
    _YT_DLP_PATH = "yt-dlp"
    return _YT_DLP_PATH


@dataclass
class VideoMetadata:
    path: str
    duration: float
    fps: float
    width: int
    height: int


class VideoSource(Protocol):
    def resolve(self) -> str: ...


class LocalFileSource:
    def __init__(self, path: str):
        self.path = Path(path)

    def resolve(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.path}，请检查文件路径后重试")
        return str(self.path.absolute())


class UrlSource:
    def __init__(self, url: str):
        self.url = url

    def resolve(self) -> str:
        output_dir = Path(tempfile.mkdtemp(prefix="video_dl_"))
        cmd = [
            _get_ytdlp(),
            "--no-playlist",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o", str(output_dir / "%(title)s.%(ext)s"),
            self.url,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600,
                           env=_subprocess_env())
        except FileNotFoundError:
            raise RuntimeError(
                "yt-dlp 未安装或不在 PATH 中，请执行: pip install yt-dlp"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"视频下载超时，请检查网络后重试: {self.url}")
        except subprocess.CalledProcessError as e:
            err_msg = _decode_stderr(e)[:500]
            raise RuntimeError(f"视频下载失败，请检查链接是否有效后重试: {err_msg}") from e

        files = list(output_dir.glob("*"))
        if not files:
            raise RuntimeError("下载完成但未找到输出文件，请稍后重试")
        video_path = str(files[0])
        return video_path


class TosSource:
    def __init__(self, tos_path: str, endpoint: str | None = None, access_key: str | None = None, secret_key: str | None = None):
        self.tos_path = tos_path
        self.endpoint = endpoint or os.getenv("TOS_ENDPOINT", "")
        self.access_key = access_key or os.getenv("TOS_ACCESS_KEY", "")
        self.secret_key = secret_key or os.getenv("TOS_SECRET_KEY", "")

    def resolve(self) -> str:
        if not self.access_key or not self.secret_key:
            raise RuntimeError(
                "TOS 配置不完整，请设置环境变量 TOS_ACCESS_KEY / TOS_SECRET_KEY "
                "或通过 TosSource 构造参数传入"
            )
        output_dir = Path(tempfile.mkdtemp(prefix="tos_dl_"))
        bucket, key = self._parse_tos_path()
        output_path = output_dir / Path(key).name
        try:
            import tos as _tos
            _client = _tos.TosClientV2(
                self.access_key, self.secret_key,
                "tos-cn-guangzhou.volces.com", "cn-guangzhou",
            )
            _client.get_object_to_file(bucket, key, str(output_path))
        except ImportError:
            raise RuntimeError("TOS 下载需要 tos SDK，请执行: pip install tos")
        except Exception as e:
            raise RuntimeError(f"TOS 下载失败，请检查凭证和路径后重试: {e}") from e
        return str(output_path)

    def _parse_tos_path(self) -> tuple[str, str]:
        path = self.tos_path
        if path.startswith("tos://"):
            path = path[6:]
        elif path.startswith("s3://"):
            path = path[5:]
        parts = path.split("/", 1)
        if len(parts) != 2:
            raise ValueError(f"无效的 TOS 路径格式: {self.tos_path}，期望格式: tos://bucket/key")
        return parts[0], parts[1]


class ArkFileSource:
    """通过 Ark Files API 上传本地文件并获取 HTTPS 预签名下载链接。

    只需 ARK_API_KEY，无需 TOS 凭证。返回的 download_url 24 小时有效。
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def resolve(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"视频文件不存在: {self.path}，请检查文件路径后重试")

        from .ark_client import ArkClient, ArkConfig

        client = ArkClient(ArkConfig(api_key=os.environ.get("ARK_HIGHLIGHT_API_KEY", "")))
        result = client.upload_file(str(self.path.absolute()))
        download_url = result.get("download_url", "")
        if not download_url:
            raise RuntimeError("Files API 未返回 download_url，请稍后重试")
        logger.info("文件已上传到 Ark Files API: %s", download_url[:80])
        return download_url


def _subprocess_env() -> dict[str, str]:
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def _decode_stderr(e: subprocess.CalledProcessError) -> str:
    for field in (e.stderr, e.stdout):
        if not field:
            continue
        if isinstance(field, bytes):
            return field.decode("utf-8", errors="replace").strip()
        return str(field).strip()
    return ""


def _convert_to_mp4(src: str, output_dir: Path) -> str:
    src_path = Path(src)
    dst = str(output_dir / f"conv_{src_path.stem}.mp4")
    cmd = [
        _get_ffmpeg(), "-y",
        "-i", src,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac",
        dst,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300,
                       env=_subprocess_env())
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"视频转码超时（300s）: {src}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"视频转码失败: {_decode_stderr(e)}") from e
    return dst


class VideoFetcher:
    def __init__(self, output_dir: str | None = None):
        self.output_dir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="vf_"))

    def fetch(self, source: VideoSource) -> VideoMetadata:
        video_path = source.resolve()
        return self._preprocess(video_path)

    # 短视频剪辑场景限制
    MAX_DURATION_SECONDS = 300   # 5 分钟
    MAX_FILE_SIZE_BYTES = 500 * 1024 * 1024  # 500MB

    def _preprocess(self, video_path: str) -> VideoMetadata:
        file_path = Path(video_path)
        if not file_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}，请检查文件路径后重试")
        if file_path.stat().st_size == 0:
            raise ValueError(f"视频文件为空（0 字节）: {video_path}，请检查视频源后重试")

        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_path.stat().st_size > self.MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"视频文件过大（{file_size_mb:.0f}MB），建议裁剪到 {self.MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB 以内后重试。"
                f" 提示: 5 分钟 1080p 视频约 200-500MB"
            )

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}，请检查文件格式后重试")

        can_decode = False
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            if fps <= 0 or duration <= 0:
                raise ValueError(f"无法解析视频参数: fps={fps}, duration={duration}, frames={frame_count}，请检查视频文件后重试")
            if frame_count == 0:
                raise ValueError(f"视频文件中无视频帧: {video_path}，请检查视频源后重试")
            if duration > self.MAX_DURATION_SECONDS:
                raise ValueError(
                    f"视频时长过长（{duration:.0f}s），当前限制 {self.MAX_DURATION_SECONDS}s（5 分钟）以内。"
                    " 请裁剪视频后重试"
                )

            # 检测 OpenCV 是否能实际解码帧（av1 等编码 isOpened() 返回 True 但 read() 失败）
            can_decode, _ = cap.read()
        finally:
            cap.release()

        # 仅在需要时转码：OpenCV 无法解码（如 av1）或非 mp4 格式
        needs_transcode = not can_decode or file_path.suffix.lower() != ".mp4"
        if needs_transcode:
            video_path = _convert_to_mp4(video_path, self.output_dir)
            # 转码后重读元数据，确保 duration/fps/width/height 准确
            cap2 = cv2.VideoCapture(video_path)
            if cap2.isOpened():
                fps = cap2.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else duration
                cap2.release()

        return VideoMetadata(
            path=video_path,
            duration=duration,
            fps=fps,
            width=width,
            height=height,
        )

    def _convert_to_mp4(self, src: str) -> str:
        return _convert_to_mp4(src, self.output_dir)
