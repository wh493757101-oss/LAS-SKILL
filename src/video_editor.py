import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


# ============================================================
# FFmpeg / ffprobe binary resolution
# ============================================================

def _get_ffmpeg() -> str:
    for candidate in [
        shutil.which("ffmpeg") or shutil.which("ffmpeg.exe") or "",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    try:
        import imageio_ffmpeg
        exe: str | None = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).exists():
            return exe
    except ImportError:
        pass
    return "ffmpeg"


def _get_ffprobe() -> str:
    for candidate in [
        shutil.which("ffprobe") or shutil.which("ffprobe.exe") or "",
    ]:
        if candidate and Path(candidate).exists():
            return candidate
    try:
        import imageio_ffmpeg
        ffmpeg_exe: str | None = imageio_ffmpeg.get_ffmpeg_exe()
        if ffmpeg_exe:
            ffprobe_exe = str(Path(ffmpeg_exe).parent / "ffprobe")
            if Path(ffprobe_exe).exists():
                return ffprobe_exe
    except ImportError:
        pass
    return "ffprobe"


# ============================================================
# Data classes
# ============================================================

@dataclass
class EditorConfig:
    """视频剪辑配置，所有字段都有安全默认值，保持向后兼容。"""
    # -- 输出 --
    output_dir: str = ""
    concat_list_filename: str = "concat_list.txt"
    output_format: str = "mp4"

    # -- 编码参数 --
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 23
    preset: str = "medium"
    audio_bitrate: str = "128k"
    pixel_format: str = "yuv420p"

    # -- 行为控制 --
    force_reencode: bool = False
    auto_reencode: bool = True
    normalize_audio: bool = False
    add_transitions: bool = False
    transition_duration: float = 0.5
    keyframe_accurate: bool = True

    # -- 日志与超时 --
    log_ffmpeg_stderr: bool = True
    ffmpeg_timeout: int = 300


@dataclass
class EditTiming:
    detection: float = 0.0
    ffmpeg_concat: float = 0.0


@dataclass
class EditResult:
    output_path: str
    segments: list[dict[str, Any]] = field(default_factory=list)
    source: str = "multimodal"
    timing: EditTiming = field(default_factory=EditTiming)


@dataclass
class StreamInfo:
    """ffprobe 单条流信息。"""
    index: int = 0
    codec_type: str = ""
    codec_name: str = ""
    width: int = 0
    height: int = 0
    r_frame_rate: str = ""
    duration: float = 0.0
    bit_rate: int = 0
    sample_rate: int = 0
    channels: int = 0
    pix_fmt: str = ""

    @property
    def fps(self) -> float:
        if "/" in self.r_frame_rate:
            num, den = self.r_frame_rate.split("/", 1)
            try:
                return float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                return 0.0
        try:
            return float(self.r_frame_rate)
        except ValueError:
            return 0.0


@dataclass
class VideoProbe:
    """ffprobe 完整探测结果。"""
    path: str = ""
    duration: float = 0.0
    bit_rate: int = 0
    format_name: str = ""
    streams: list[StreamInfo] = field(default_factory=list)
    gop_size: int = 0

    @property
    def video_stream(self) -> StreamInfo | None:
        for s in self.streams:
            if s.codec_type == "video":
                return s
        return None

    @property
    def audio_stream(self) -> StreamInfo | None:
        for s in self.streams:
            if s.codec_type == "audio":
                return s
        return None


@dataclass(frozen=True)
class EncodingPreset:
    """预置编码质量档位。"""
    crf: int
    preset: str

    HIGH_QUALITY: ClassVar["EncodingPreset"]
    BALANCED: ClassVar["EncodingPreset"]
    COMPRESSED: ClassVar["EncodingPreset"]


EncodingPreset.HIGH_QUALITY = EncodingPreset(crf=18, preset="slow")
EncodingPreset.BALANCED = EncodingPreset(crf=23, preset="medium")
EncodingPreset.COMPRESSED = EncodingPreset(crf=28, preset="fast")


# ============================================================
# Codec ↔ container compatibility table
# ============================================================

_MP4_VIDEO_CODECS = frozenset({"h264", "hevc", "mpeg4", "av1"})
_MP4_AUDIO_CODECS = frozenset({"aac", "mp3", "ac3", "eac3", "alac"})
_WEBM_VIDEO_CODECS = frozenset({"vp8", "vp9", "av1"})
_WEBM_AUDIO_CODECS = frozenset({"opus", "vorbis"})


def _check_codec_consistency(probe: VideoProbe, output_format: str) -> bool:
    """检查源编码是否兼容目标容器 stream-copy。"""
    fmt = output_format.lower()
    vs = probe.video_stream
    audio = probe.audio_stream

    if fmt == "mp4":
        if vs and vs.codec_name not in _MP4_VIDEO_CODECS:
            return False
        if audio and audio.codec_name not in _MP4_AUDIO_CODECS:
            return False
        return True
    elif fmt == "webm":
        if vs and vs.codec_name not in _WEBM_VIDEO_CODECS:
            return False
        if audio and audio.codec_name not in _WEBM_AUDIO_CODECS:
            return False
        return True
    elif fmt == "mkv":
        return True  # Matroska 几乎支持所有编码
    return True  # 未知容器，假设兼容


# ============================================================
# VideoEditor
# ============================================================

class VideoEditor:
    def __init__(self, config: EditorConfig | None = None):
        self.config = config or EditorConfig()

    # ---- ffprobe ----------------------------------------------------------

    def probe(self, video_path: str) -> VideoProbe:
        """用 ffprobe 探测视频元数据。失败返回空 VideoProbe。"""
        cmd = [
            _get_ffprobe(), "-v", "quiet",
            "-print_format", "json",
            "-show_streams", "-show_format",
            video_path,
        ]
        try:
            result = self._run_ffmpeg(cmd, timeout=30)
            data = json.loads(result.stdout)
        except Exception as e:
            logger.warning("ffprobe 探测失败，使用降级模式: %s", e)
            return VideoProbe(path=video_path)

        streams = []
        for s in data.get("streams", []):
            streams.append(StreamInfo(
                index=s.get("index", 0),
                codec_type=s.get("codec_type", ""),
                codec_name=s.get("codec_name", ""),
                width=s.get("width", 0),
                height=s.get("height", 0),
                r_frame_rate=s.get("r_frame_rate", ""),
                duration=float(s.get("duration", 0)),
                bit_rate=int(s.get("bit_rate", 0)) if s.get("bit_rate") else 0,
                sample_rate=int(s.get("sample_rate", 0)) if s.get("sample_rate") else 0,
                channels=s.get("channels", 0),
                pix_fmt=s.get("pix_fmt", ""),
            ))

        fmt = data.get("format", {})
        probe = VideoProbe(
            path=video_path,
            duration=float(fmt.get("duration", 0)),
            bit_rate=int(fmt.get("bit_rate", 0)) if fmt.get("bit_rate") else 0,
            format_name=fmt.get("format_name", ""),
            streams=streams,
        )
        probe.gop_size = self._get_gop_size(video_path)
        return probe

    def _get_gop_size(self, video_path: str) -> int:
        """通过统计关键帧间隔估算 GOP 大小。限制前 1000 帧。"""
        cmd = [
            _get_ffprobe(), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "frame=key_frame",
            "-of", "csv",
            "-read_intervals", "%+#1000",
            video_path,
        ]
        try:
            result = self._run_ffmpeg(cmd, timeout=30)
            intervals: list[int] = []
            last_kf = -1
            for i, line in enumerate(result.stdout.strip().split("\n")):
                if line.strip().startswith("frame,1") or line.strip() in ("frame,1", "1"):
                    if last_kf >= 0:
                        intervals.append(i - last_kf)
                    last_kf = i
            if intervals:
                intervals.sort()
                return intervals[len(intervals) // 2]  # 中位数
        except Exception as e:
            logger.debug("GOP 探测失败: %s", e)
        return 0

    # ---- subprocess runner -------------------------------------------------

    def _run_ffmpeg(
        self, cmd: list[str], timeout: int | None = None
    ) -> subprocess.CompletedProcess:
        """统一 FFmpeg/ffprobe 子进程执行器。"""
        timeout = timeout if timeout is not None else self.config.ffmpeg_timeout
        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=timeout,
            )
            if self.config.log_ffmpeg_stderr and result.stderr:
                logger.debug("FFmpeg stderr [%s]: %s",
                             " ".join(cmd[:4]), result.stderr[:500])
            return result
        except subprocess.CalledProcessError as e:
            stderr = e.stderr or ""
            logger.error("FFmpeg 命令失败 (exit=%s): %s\nstderr: %s",
                         e.returncode, " ".join(cmd[:6]), stderr[:500])
            raise RuntimeError(
                f"FFmpeg 命令失败: {' '.join(cmd[:4])}...\n{stderr[-300:]}"
            ) from e
        except subprocess.TimeoutExpired as e:
            logger.error("FFmpeg 命令超时 (%ss): %s", timeout, " ".join(cmd[:4]))
            raise RuntimeError(
                f"FFmpeg 命令超时（{timeout}s）: {' '.join(cmd[:4])}..."
            ) from e

    # ---- keyframe alignment ------------------------------------------------

    @staticmethod
    def _align_cut_to_keyframe(
        timestamp: float, gop_size: int, fps: float, direction: str
    ) -> float:
        """将切割时间戳对齐到 GOP 边界。

        direction="start": 向下取整（确保从关键帧开始）
        direction="end":   向上取整（确保覆盖完整 GOP）
        """
        if gop_size <= 0 or fps <= 0:
            return timestamp
        gop_duration = gop_size / fps
        gop_index = int(timestamp / gop_duration)
        if direction == "start":
            return gop_index * gop_duration
        else:
            return (gop_index + 1) * gop_duration

    # ---- encoding args -----------------------------------------------------

    def _build_encode_args(self) -> list[str]:
        """根据 EditorConfig 构建编码参数。"""
        return [
            "-c:v", self.config.video_codec,
            "-crf", str(self.config.crf),
            "-preset", self.config.preset,
            "-c:a", self.config.audio_codec,
            "-b:a", self.config.audio_bitrate,
            "-pix_fmt", self.config.pixel_format,
            "-movflags", "+faststart",
        ]

    # ---- segment cutting ---------------------------------------------------

    def _cut_segment_streamcopy(
        self, video_path: str, start: float, end: float, output_path: str,
    ) -> None:
        """流拷贝模式裁剪片段（快速，关键帧精度）。"""
        duration = end - start
        cmd = [
            _get_ffmpeg(), "-y", "-hide_banner",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            output_path,
        ]
        self._run_ffmpeg(cmd)

    def _cut_segment_reencode(
        self, video_path: str, start: float, end: float, output_path: str,
    ) -> None:
        """重编码模式裁剪片段（帧精确，可加 fade 效果）。"""
        duration = end - start
        cmd = [
            _get_ffmpeg(), "-y", "-hide_banner",
            "-ss", str(start),
            "-to", str(end),
            "-i", video_path,
        ]
        cmd += self._build_encode_args()

        if self.config.add_transitions:
            td = self.config.transition_duration
            fade_out_start = max(0, duration - td)
            cmd += [
                "-vf", f"fade=t=in:d={td},fade=t=out:st={fade_out_start}:d={td}",
                "-af", f"afade=t=in:d={td},afade=t=out:st={fade_out_start}:d={td}",
            ]

        cmd.append(output_path)
        self._run_ffmpeg(cmd)

    # ---- audio normalization -----------------------------------------------

    def _normalize_audio_segments(
        self, clip_paths: list[str], probe: VideoProbe
    ) -> list[str]:
        """EBU R128 loudnorm 两遍归一化。无音频流则原样返回。"""
        if not probe.audio_stream:
            logger.info("视频无音频流，跳过归一化")
            return clip_paths

        normalized: list[str] = []
        for i, clip in enumerate(clip_paths):
            try:
                normalized.append(self._loudnorm_single(clip, i))
            except Exception as e:
                logger.warning("片段 #%d 音频归一化失败，使用原始片段: %s", i, e)
                normalized.append(clip)
        return normalized

    def _loudnorm_single(self, clip_path: str, index: int) -> str:
        """对单个片段执行 loudnorm 两遍处理。"""
        # Pass 1: 测量
        null_dev = "NUL" if os.name == "nt" else "/dev/null"
        cmd1 = [
            _get_ffmpeg(), "-y", "-hide_banner",
            "-i", clip_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", null_dev,
        ]
        result1 = self._run_ffmpeg(cmd1)
        measured = self._parse_loudnorm_json(result1.stderr)
        if not measured:
            raise RuntimeError("loudnorm 第一遍测量失败")

        # Pass 2: 应用
        output_dir = Path(clip_path).parent
        normalized_path = str(output_dir / f"_norm_{index:03d}.mp4")
        af = (
            f"loudnorm=I=-16:TP=-1.5:LRA=11:"
            f"measured_I={measured['input_i']}:"
            f"measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}"
        )
        cmd2 = [
            _get_ffmpeg(), "-y", "-hide_banner",
            "-i", clip_path,
            "-af", af,
            "-c:v", "copy",
            normalized_path,
        ]
        self._run_ffmpeg(cmd2)
        return normalized_path

    @staticmethod
    def _parse_loudnorm_json(stderr: str) -> dict[str, float] | None:
        """从 loudnorm JSON 输出中提取 measured 值。"""
        for line in stderr.split("\n"):
            line = line.strip()
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    keys = ("input_i", "input_tp", "input_lra", "input_thresh")
                    if all(k in data for k in keys):
                        return {k: float(data[k]) for k in keys}
                except (json.JSONDecodeError, ValueError):
                    continue
        return None

    # ---- concat ------------------------------------------------------------

    def _concat_streamcopy(
        self, clip_paths: list[str], output_path: str,
    ) -> None:
        """流拷贝 concat demuxer 拼接。"""
        concat_list = str(Path(output_path).parent / self.config.concat_list_filename)
        with open(concat_list, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        cmd = [
            _get_ffmpeg(), "-y", "-hide_banner",
            "-f", "concat", "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            output_path,
        ]
        self._run_ffmpeg(cmd)

    def _concat_with_transitions(
        self, clip_paths: list[str], output_path: str,
    ) -> None:
        """xfade + acrossfade 转场拼接。需要重编码。"""
        n = len(clip_paths)
        if n == 1:
            shutil.copy2(clip_paths[0], output_path)
            return

        # 先获取每个 clip 的时长
        durations: list[float] = []
        for cp in clip_paths:
            probe = self.probe(cp)
            durations.append(probe.duration if probe.duration > 0 else 1.0)

        td = self.config.transition_duration

        # 构建输入参数
        inputs: list[str] = []
        for cp in clip_paths:
            inputs += ["-i", cp]

        # 构建 filter_complex
        v_filters: list[str] = []
        a_filters: list[str] = []

        # 视频 xfade 链
        v_prev = "[0:v]"
        cumulative = durations[0]
        for i in range(1, n):
            offset = cumulative - td
            v_label = f"[v{i}]" if i < n - 1 else "[vout]"
            v_filters.append(
                f"{v_prev}[{i}:v]xfade=transition=fade:duration={td}:offset={offset}{v_label}"
            )
            v_prev = v_label
            cumulative += durations[i]

        # 音频 acrossfade 链
        a_prev = "[0:a]"
        for i in range(1, n):
            a_label = f"[a{i}]" if i < n - 1 else "[aout]"
            a_filters.append(
                f"{a_prev}[{i}:a]acrossfade=d={td}{a_label}"
            )
            a_prev = a_label

        filter_complex = ";".join(v_filters + a_filters)

        cmd = [
            _get_ffmpeg(), "-y", "-hide_banner",
        ] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[vout]", "-map", "[aout]",
        ] + self._build_encode_args() + [
            output_path,
        ]
        self._run_ffmpeg(cmd)

    # ---- main editing method -----------------------------------------------

    def edit_with_ffmpeg(
        self,
        video_path: str,
        segments: list[dict[str, Any]],
    ) -> EditResult:
        """使用 FFmpeg 拼接高光片段。

        根据 EditorConfig 自动选择流拷贝或重编码路径：
        - 默认：流拷贝（快速，关键帧精度）
        - auto_reencode=True + 编码不兼容 → 自动重编码
        - force_reencode=True → 强制重编码
        - add_transitions=True → 重编码 + xfade 转场
        - normalize_audio=True → loudnorm 两遍归一化
        """
        if not segments:
            raise ValueError("segments 为空，无法进行剪辑")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(
            output_dir / f"highlight_reel_{time.strftime('%Y%m%d_%H%M%S')}.{self.config.output_format}"
        )

        t_start = time.time()
        clip_paths: list[str] = []

        # 1. 探测源视频
        probe = self.probe(video_path)

        # 2. 决定编码策略
        needs_reencode = self.config.force_reencode or self.config.add_transitions
        if self.config.auto_reencode and not needs_reencode:
            if not _check_codec_consistency(probe, self.config.output_format):
                logger.info("源编码与目标容器不兼容，自动切换到重编码模式")
                needs_reencode = True

        if needs_reencode:
            logger.info("使用重编码模式 (codec=%s, crf=%s, preset=%s)",
                        self.config.video_codec, self.config.crf, self.config.preset)
        else:
            logger.info("使用流拷贝模式")

        # 3. 关键帧对齐（仅流拷贝模式需要）
        if self.config.keyframe_accurate and not needs_reencode:
            gop = probe.gop_size
            fps = probe.video_stream.fps if probe.video_stream else 0.0
            if gop > 0 and fps > 0:
                for seg in segments:
                    seg["start_time"] = self._align_cut_to_keyframe(
                        seg["start_time"], gop, fps, "start"
                    )
                    seg["end_time"] = self._align_cut_to_keyframe(
                        seg["end_time"], gop, fps, "end"
                    )

        # 4. 逐片段裁剪
        try:
            for i, seg in enumerate(segments):
                start = seg["start_time"]
                end = seg["end_time"]
                clip_path = str(output_dir / f"_clip_{i:03d}.mp4")

                if needs_reencode:
                    self._cut_segment_reencode(video_path, start, end, clip_path)
                else:
                    self._cut_segment_streamcopy(video_path, start, end, clip_path)
                clip_paths.append(clip_path)

            # 5. 音频归一化
            if self.config.normalize_audio:
                clip_paths = self._normalize_audio_segments(clip_paths, probe)

            # 6. 拼接
            if self.config.add_transitions:
                self._concat_with_transitions(clip_paths, output_path)
            else:
                self._concat_streamcopy(clip_paths, output_path)

        finally:
            # 7. 清理临时文件
            for cp in clip_paths:
                try:
                    Path(cp).unlink(missing_ok=True)
                except OSError:
                    pass
            # 清理归一化产生的中间文件
            for f in output_dir.glob("_norm_*.mp4"):
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
            concat_list = output_dir / self.config.concat_list_filename
            try:
                concat_list.unlink(missing_ok=True)
            except OSError:
                pass

        t_concat = time.time() - t_start

        return EditResult(
            output_path=output_path,
            segments=segments,
            source="multimodal",
            timing=EditTiming(ffmpeg_concat=t_concat),
        )
