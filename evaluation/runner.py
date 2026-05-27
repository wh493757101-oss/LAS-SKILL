import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .evaluator import EvalReport, HighlightEvaluator, TestCaseLoader, compute_weighted_score
from .llm_judge import JudgeReport, LLMJudge
from .report import ReportConfig, ReportGenerator

logger = logging.getLogger(__name__)


@dataclass
class EvalRunConfig:
    test_cases_root: str = ""
    output_dir: str = ""
    iou_threshold: float = 0.5
    skip_llm_judge: bool = False
    skip_edit: bool = False
    case_filter: list[str] = field(default_factory=list)
    judge_weight: float = 0.5
    judge_max_retries: int = 3
    concurrency: int = 1
    concurrency_warmup: int = 0


class EvalRunner:
    def __init__(self, config: EvalRunConfig | None = None):
        self.config = config or EvalRunConfig()

    def run(self) -> tuple[EvalReport, JudgeReport, str]:
        loader = TestCaseLoader(self.config.test_cases_root)
        cases = loader.load_all()

        if self.config.case_filter:
            cases = [c for c in cases if c["case_id"] in self.config.case_filter]

        logger.info("加载 %d 个评测用例", len(cases))
        if not cases:
            return EvalReport(), JudgeReport(), ""

        if self.config.concurrency > 1:
            results = self._run_concurrent(cases)
        else:
            results: list[dict[str, Any]] = []
            for case in cases:
                logger.info("运行 case: %s", case["case_id"])
                result = self._run_case(case)
                results.append(result)

        # 并行执行量化评测和 LLM Judge
        evaluator = HighlightEvaluator(iou_threshold=self.config.iou_threshold)
        judge_report = JudgeReport()

        if self.config.skip_llm_judge:
            eval_report = evaluator.evaluate_all(results)
            judge_report.degraded = True
        else:
            with ThreadPoolExecutor(max_workers=2) as executor:
                eval_future = executor.submit(evaluator.evaluate_all, results)

                def _run_judge():
                    judge = LLMJudge()
                    judge_cases = self._build_judge_cases(results)
                    if judge_cases:
                        return judge.judge_all(judge_cases, max_retries=self.config.judge_max_retries)
                    jr = JudgeReport()
                    jr.degraded = True
                    return jr

                judge_future = executor.submit(_run_judge)
                eval_report = eval_future.result()
                judge_report = judge_future.result()

        # 计算加权总分
        weighted = compute_weighted_score(
            eval_report, judge_report,
            weight_eval=1.0 - self.config.judge_weight,
            weight_judge=self.config.judge_weight,
        )

        report_gen = ReportGenerator(
            ReportConfig(
                output_dir=self.config.output_dir,
                save_charts=bool(self.config.output_dir),
                save_json=bool(self.config.output_dir),
            )
        )
        report_text = report_gen.generate(eval_report, judge_report, weighted)

        return eval_report, judge_report, report_text

    def _run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        import tracemalloc

        from src.main import PipelineConfig, VideoHighlightPipeline
        from src.video_fetcher import LocalFileSource, TosSource, UrlSource

        source_type = case.get("source_type", "local")
        instruction = case.get("instruction", {})
        description = instruction.get("prompt", "")

        pipeline = VideoHighlightPipeline(
            PipelineConfig(output_dir=self.config.output_dir)
        )

        if source_type == "tos":
            tos_path = case.get("tos_path", "")
            source = TosSource(tos_path)
        elif source_type == "remote":
            source_url = case.get("source_url", "")
            source = UrlSource(source_url)
        else:
            video_path = case.get("video_path", "")
            source = LocalFileSource(video_path)

        tracemalloc.start()
        result = pipeline.run(
            source,
            description=description,
            skip_edit=self.config.skip_edit,
        )
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if result.edit and result.edit.segments:
            predicted = [
                {
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "score": seg.get("score", 0.5),
                }
                for seg in result.edit.segments
            ]

        return {
            "case_id": case["case_id"],
            "category": case["category"],
            "difficulty": case["difficulty"],
            "source_type": case["source_type"],
            "predicted": predicted,
            "ground_truth": case["ground_truth"],
            "target": description,
            "style": instruction.get("style", ""),
            "usage": {},
            "video_duration": result.metadata.duration,
            "elapsed_time": result.elapsed_time,
            "api_calls": 1,
            "api_retries": 0,
            "memory_peak_mb": peak_bytes / (1024 * 1024),
            "memory_avg_mb": 0.0,
        }

    def _run_concurrent(self, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """并发压测模式：多线程并行执行 case，测量吞吐量。"""
        n = len(cases)
        concurrency = self.config.concurrency
        warmup = self.config.concurrency_warmup

        logger.info("并发压测: %d cases, 并发度=%d, 预热=%d", n, concurrency, warmup)

        results: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(self._run_case, case): case
                for case in cases
            }
            for future in as_completed(future_map):
                try:
                    results.append(future.result())
                except Exception as e:
                    case = future_map[future]
                    logger.error("并发 case %s 异常: %s", case["case_id"], e)
                    results.append({
                        "case_id": case["case_id"],
                        "category": case.get("category", ""),
                        "difficulty": case.get("difficulty", ""),
                        "source_type": case.get("source_type", "local"),
                        "predicted": [],
                        "ground_truth": case.get("ground_truth", []),
                        "target": "",
                        "style": "",
                        "error": str(e),
                    })

        elapsed = time.perf_counter() - t0
        effective = n - warmup
        throughput = effective / elapsed if elapsed > 0 else 0.0

        # 将吞吐量数据注入到每个 result 中，便于 CostStats 汇总
        for r in results:
            r["concurrency"] = concurrency
            r["concurrent_total_elapsed"] = elapsed
            r["concurrent_throughput"] = throughput

        logger.info(
            "并发压测完成: 总耗时=%.1fs, 有效 case=%d, 吞吐量=%.2f case/s (并发度=%d)",
            elapsed, effective, throughput, concurrency,
        )

        return results

    def _build_judge_cases(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        judge_cases: list[dict[str, Any]] = []
        for r in results:
            if not r.get("predicted"):
                continue
            judge_cases.append({
                "category": r.get("category", ""),
                "target": r.get("target", ""),
                "style": r.get("style", ""),
                "segments": r["predicted"],
                "video_path": r.get("video_path", ""),
            })
        return judge_cases
