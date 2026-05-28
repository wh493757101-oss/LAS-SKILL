"""端到端验证脚本 — 测试完整 Pipeline：视频获取 → 多模态识别 → FFmpeg 拼接 → 评测"""
import os
import sys
import logging
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("verify")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def check_env():
    missing = []
    for key in ["ARK_HIGHLIGHT_API_KEY"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        logger.error("缺少环境变量: %s", ", ".join(missing))
        logger.error("请设置 .env 文件中的必要变量: ARK_HIGHLIGHT_API_KEY")
        return False
    logger.info("环境变量 OK")
    return True


def verify_fetch(video_path: str):
    from src.main import VideoHighlightPipeline, PipelineConfig
    from src.video_fetcher import LocalFileSource

    logger.info("=" * 60)
    logger.info("Step 1: 视频预处理验证（skip_edit=True）")
    logger.info("=" * 60)

    pipeline = VideoHighlightPipeline(PipelineConfig(output_dir="./output"))
    result = pipeline.run(
        LocalFileSource(video_path),
        description="帮我把精彩片段剪成60秒集锦，节奏要快",
        skip_edit=True,
    )

    if result.error:
        logger.error("预处理失败: %s", result.error)
        return None

    logger.info("视频时长: %.1fs", result.metadata.duration)
    logger.info("分辨率: %dx%d", result.metadata.width, result.metadata.height)
    logger.info("帧率: %.1f fps", result.metadata.fps)

    return result


def verify_editing(video_path: str):
    from src.main import VideoHighlightPipeline, PipelineConfig
    from src.video_fetcher import LocalFileSource

    logger.info("=" * 60)
    logger.info("Step 2: 完整 Pipeline 验证（多模态识别 + FFmpeg 拼接）")
    logger.info("=" * 60)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    pipeline = VideoHighlightPipeline(PipelineConfig(output_dir=output_dir))
    result = pipeline.run(
        LocalFileSource(video_path),
        description="帮我把精彩片段剪成60秒集锦，节奏要快",
        skip_edit=False,
    )

    if result.error:
        logger.error("Pipeline 失败: %s", result.error)
        return None

    logger.info("识别方式: %s", result.edit.source if result.edit else "N/A")
    logger.info("输出路径: %s", result.edit.output_path if result.edit else "N/A")
    if result.edit and result.edit.segments:
        for i, seg in enumerate(result.edit.segments):
            logger.info("  #%d: %.1fs - %.1fs (score=%.2f)",
                        i + 1, seg["start_time"], seg["end_time"], seg.get("score", 0))
    logger.info("JSON 导出:\n%s", pipeline.export_json(result))
    return result


def verify_evaluation():
    from evaluation.evaluator import HighlightEvaluator, compute_weighted_score
    from evaluation.llm_judge import JudgeReport, JudgeScore
    from evaluation.report import ReportConfig, ReportGenerator

    logger.info("=" * 60)
    logger.info("Step 3: 评测框架验证")
    logger.info("=" * 60)

    evaluator = HighlightEvaluator()
    results = [
        {
            "case_id": "verify_001",
            "category": "测试",
            "difficulty": "easy",
            "source_type": "local",
            "predicted": [{"start_time": 0.0, "end_time": 5.0}],
            "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
        },
    ]
    eval_report = evaluator.evaluate_all(results)
    logger.info(
        "量化评测: F1=%.3f, Precision=%.3f, Recall=%.3f",
        eval_report.overall_f1, eval_report.overall_precision, eval_report.overall_recall,
    )

    judge_report = JudgeReport(
        scores=[JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=5.0,
                           completeness=4.0, instruction_fit=4.0, overall_comment="测试通过")],
        overall_rhythm=4.0, overall_transition_quality=4.0, overall_audiovisual_sync=5.0,
        overall_completeness=4.0, overall_instruction_fit=4.0, overall_average=4.2,
    )

    weighted = compute_weighted_score(eval_report, judge_report)
    logger.info(
        "加权总分: %.4f (量化: %.4f, Judge: %.4f, 降级: %s)",
        weighted["weighted_score"], weighted["eval_score"],
        weighted["judge_score"], weighted["degraded"],
    )

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    gen = ReportGenerator(ReportConfig(output_dir=output_dir, save_charts=True))
    report_text = gen.generate(eval_report, judge_report, weighted)
    logger.info("报告已生成到 ./output/")
    print("\n" + report_text)
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/verify_e2e.py <视频文件路径> [--detect-only]")
        print("示例: python scripts/verify_e2e.py ./test_video.mp4")
        print("      python scripts/verify_e2e.py ./test_video.mp4 --detect-only")
        sys.exit(1)

    video_path = sys.argv[1]
    detect_only = "--detect-only" in sys.argv

    if not os.path.exists(video_path):
        logger.error("视频文件不存在: %s", video_path)
        sys.exit(1)

    if not check_env():
        sys.exit(1)

    result = verify_fetch(video_path)
    if result is None:
        sys.exit(1)

    if detect_only:
        logger.info("仅预处理模式，跳过剪辑和评测")
        return

    result = verify_editing(video_path)
    if result is None:
        logger.warning("剪辑步骤失败，但检测已通过")

    verify_evaluation()

    logger.info("=" * 60)
    logger.info("验证完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
