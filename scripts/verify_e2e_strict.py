"""严格端到端验证 — 多模态识别 + FFmpeg 拼接 + LLM Judge 全路径，不降级"""
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
logger = logging.getLogger("e2e")

def _project_root():
    return Path(__file__).resolve().parent.parent

sys.path.insert(0, str(_project_root()))

_ENV_FILE = _project_root() / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def strict_fail(reason: str):
    logger.error("降级触发，验证终止: %s", reason)
    sys.exit(1)


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else str(_project_root() / "test_video_e2e.mp4")
    logger.info("测试视频: %s", video_path)

    # ---- Step 1: VideoFetcher ----
    from src.video_fetcher import LocalFileSource, VideoFetcher

    logger.info("=" * 60)
    logger.info("Step 1: VideoFetcher — 预处理")
    logger.info("=" * 60)

    fetcher = VideoFetcher(output_dir=str(_project_root() / "output" / "e2e_strict"))
    metadata = fetcher.fetch(LocalFileSource(video_path))
    logger.info("duration=%.1fs  fps=%.1f  %dx%d",
                metadata.duration, metadata.fps, metadata.width, metadata.height)

    # ---- Step 2: HighlightDetector — 多模态识别 ----
    from src.highlight_detector import DetectorConfig, HighlightDetector

    logger.info("=" * 60)
    logger.info("Step 2: HighlightDetector — 多模态高光识别")
    logger.info("=" * 60)

    detector = HighlightDetector(DetectorConfig())
    detection = detector.detect(metadata, description="测试端到端验证 — 剪辑精彩片段")

    if not detection.segments:
        strict_fail("多模态识别返回 0 个高光片段")

    logger.info("source=%s segments=%d", detection.source, len(detection.segments))
    for seg in detection.segments:
        logger.info("  %.1fs-%.1fs score=%.2f",
                    seg.start_time, seg.end_time, seg.combined_score)

    # ---- Step 3: VideoEditor — FFmpeg 拼接 ----
    from src.video_editor import EditorConfig, VideoEditor

    logger.info("=" * 60)
    logger.info("Step 3: VideoEditor — FFmpeg 拼接")
    logger.info("=" * 60)

    session_dir = str(_project_root() / "output" / "e2e_strict" / "session")
    editor_cfg = EditorConfig(output_dir=session_dir)
    editor = VideoEditor(editor_cfg)

    segments = [
        {
            "start_time": seg.start_time,
            "end_time": seg.end_time,
            "score": seg.combined_score,
            "label": getattr(seg, "label", ""),
        }
        for seg in detection.segments
    ]
    edit = editor.edit_with_ffmpeg(metadata.path, segments)

    if edit.source != "multimodal":
        strict_fail(f"剪辑源不是 multimodal: {edit.source}")

    logger.info("source=%s output=%s segments=%d", edit.source, edit.output_path, len(edit.segments))
    for seg in edit.segments:
        logger.info("  %.1fs-%.1fs score=%.2f label=%s",
                    seg.get("start_time", 0), seg.get("end_time", 0),
                    seg.get("score", 0), seg.get("label", ""))

    if not edit.segments:
        strict_fail("FFmpeg 拼接返回 0 个高光片段")

    # ---- Step 4: LLMJudge — 视频+音频打分 ----
    from evaluation.llm_judge import LLMJudge, JudgeConfig

    logger.info("=" * 60)
    logger.info("Step 4: LLM Judge — 视频+音频评分")
    logger.info("=" * 60)

    judge_cfg = JudgeConfig()
    logger.info("Judge model=%s base_url=%s", judge_cfg.model, judge_cfg.base_url)

    edited_video = edit.output_path if edit.output_path else metadata.path
    judge = LLMJudge(config=judge_cfg)
    score = judge.judge(
        category="测试",
        target="精彩集锦（节奏快）",
        style="快节奏",
        segments=edit.segments,
        video_path=edited_video,
        max_retries=2,
    )

    if score.error:
        strict_fail(f"LLM Judge 评分失败: {score.error}")

    logger.info("Judge 评分: 节奏感=%.1f 转场=%.1f 音画=%.1f 完整性=%.1f 契合度=%.1f 均分=%.1f",
                score.rhythm, score.transition_quality, score.audiovisual_sync,
                score.completeness, score.instruction_fit, score.average)
    logger.info("总体评价: %s", score.overall_comment)

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("全链路验证通过！")
    logger.info("  多模态识别: multimodal (%d segments)", len(detection.segments))
    logger.info("  输出: %s", edit.output_path)
    logger.info("  LLM Judge: %.1f/10  (%s)", score.average, score.overall_comment)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
