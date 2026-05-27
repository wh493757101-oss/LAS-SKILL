import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SegmentMatch:
    predicted_start: float
    predicted_end: float
    gt_start: float
    gt_end: float
    iou: float
    hit: bool
    mae_start: float = 0.0
    mae_end: float = 0.0
    mae_avg: float = 0.0


@dataclass
class CaseScore:
    case_id: str
    category: str
    difficulty: str
    source_type: str
    iou_scores: list[float] = field(default_factory=list)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    hit_rate_1: float = 0.0
    hit_rate_3: float = 0.0
    mae: float = 0.0
    iou_distribution: dict[str, int] = field(default_factory=dict)
    matched_pairs: list[SegmentMatch] = field(default_factory=list)
    error: str | None = None
    degraded: bool = False


@dataclass
class CostStats:
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    video_duration: float = 0.0
    tokens_per_minute: float = 0.0
    api_calls: int = 0
    api_retries: int = 0
    total_elapsed: float = 0.0
    avg_elapsed: float = 0.0
    processing_ratio: float = 0.0
    memory_peak_mb: float = 0.0
    memory_avg_mb: float = 0.0
    concurrent_throughput: float = 0.0
    concurrency: int = 0


@dataclass
class EvalReport:
    scores: list[CaseScore] = field(default_factory=list)
    overall_iou: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    overall_hit_rate_1: float = 0.0
    overall_hit_rate_3: float = 0.0
    overall_mae: float = 0.0
    iou_distribution: dict[str, int] = field(default_factory=lambda: {
        "excellent": 0, "qualified": 0, "unqualified": 0,
    })
    exception_rate: float = 0.0
    exception_count: int = 0
    total_count: int = 0
    degraded_count: int = 0
    degradation_rate: float = 0.0
    cost: CostStats = field(default_factory=CostStats)
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_source: dict[str, dict[str, Any]] = field(default_factory=dict)


class HighlightEvaluator:
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def compute_iou(
        self, pred_start: float, pred_end: float, gt_start: float, gt_end: float
    ) -> float:
        intersection = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
        union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    def match_segments(
        self,
        predicted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
    ) -> list[SegmentMatch]:
        matches: list[SegmentMatch] = []
        gt_used: set[int] = set()

        for pred in predicted:
            best_iou = 0.0
            best_gt_idx = -1
            best_gt: dict[str, Any] | None = None

            for j, gt in enumerate(ground_truth):
                if j in gt_used:
                    continue
                iou = self.compute_iou(
                    pred["start_time"], pred["end_time"],
                    gt["start_time"], gt["end_time"],
                )
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
                    best_gt = gt

            hit = best_iou >= self.iou_threshold
            if hit and best_gt is not None:
                gt_used.add(best_gt_idx)

            mae_start = abs(pred["start_time"] - best_gt["start_time"]) if best_gt else 0.0
            mae_end = abs(pred["end_time"] - best_gt["end_time"]) if best_gt else 0.0

            matches.append(SegmentMatch(
                predicted_start=pred["start_time"],
                predicted_end=pred["end_time"],
                gt_start=best_gt["start_time"] if best_gt else 0.0,
                gt_end=best_gt["end_time"] if best_gt else 0.0,
                iou=best_iou,
                hit=hit,
                mae_start=mae_start,
                mae_end=mae_end,
                mae_avg=(mae_start + mae_end) / 2,
            ))

        return matches

    def compute_hit_rate(
        self,
        predicted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        k: int,
    ) -> float:
        if not ground_truth or not predicted:
            return 0.0
        top_k = predicted[:k]
        hits = 0
        for pred in top_k:
            for gt in ground_truth:
                iou = self.compute_iou(
                    pred["start_time"], pred["end_time"],
                    gt["start_time"], gt["end_time"],
                )
                if iou >= self.iou_threshold:
                    hits += 1
                    break
        return hits / min(k, len(predicted))

    def classify_iou(self, iou: float) -> str:
        if iou >= 0.8:
            return "excellent"
        elif iou >= 0.5:
            return "qualified"
        return "unqualified"

    def score_case(
        self,
        case_id: str,
        predicted: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
        category: str = "",
        difficulty: str = "",
        source_type: str = "local",
        degraded: bool = False,
    ) -> CaseScore:
        if not ground_truth:
            return CaseScore(
                case_id=case_id,
                category=category,
                difficulty=difficulty,
                source_type=source_type,
                error="ground_truth 为空",
            )

        if not predicted:
            return CaseScore(
                case_id=case_id,
                category=category,
                difficulty=difficulty,
                source_type=source_type,
                precision=0.0,
                recall=0.0,
                f1=0.0,
            )

        matches = self.match_segments(predicted, ground_truth)
        hit_count = sum(1 for m in matches if m.hit)
        iou_scores = [m.iou for m in matches]

        precision = hit_count / len(predicted) if predicted else 0.0
        recall = hit_count / len(ground_truth) if ground_truth else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        hit_rate_1 = self.compute_hit_rate(predicted, ground_truth, 1)
        hit_rate_3 = self.compute_hit_rate(predicted, ground_truth, 3)

        mae = 0.0
        hit_matches = [m for m in matches if m.hit]
        if hit_matches:
            mae = sum(m.mae_avg for m in hit_matches) / len(hit_matches)

        iou_dist: dict[str, int] = {"excellent": 0, "qualified": 0, "unqualified": 0}
        for iou in iou_scores:
            iou_dist[self.classify_iou(iou)] += 1

        return CaseScore(
            case_id=case_id,
            category=category,
            difficulty=difficulty,
            source_type=source_type,
            iou_scores=iou_scores,
            precision=precision,
            recall=recall,
            f1=f1,
            hit_rate_1=hit_rate_1,
            hit_rate_3=hit_rate_3,
            mae=mae,
            iou_distribution=iou_dist,
            matched_pairs=matches,
            degraded=degraded,
        )

    def evaluate_all(self, results: list[dict[str, Any]]) -> EvalReport:
        report = EvalReport()
        if not results:
            return report

        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        total_iou = 0.0
        total_hit1 = 0.0
        total_hit3 = 0.0
        total_mae = 0.0
        mae_count = 0
        iou_count = 0

        cat_scores: dict[str, list[float]] = {}
        dif_scores: dict[str, list[float]] = {}
        src_scores: dict[str, list[float]] = {}
        cat_deg_counts: dict[str, int] = {}
        dif_deg_counts: dict[str, int] = {}
        src_deg_counts: dict[str, int] = {}
        cat_counts: dict[str, int] = {}
        dif_counts: dict[str, int] = {}
        src_counts: dict[str, int] = {}

        for r in results:
            score = self.score_case(
                case_id=r.get("case_id", ""),
                predicted=r.get("predicted", []),
                ground_truth=r.get("ground_truth", []),
                category=r.get("category", ""),
                difficulty=r.get("difficulty", ""),
                source_type=r.get("source_type", "local"),
                degraded=r.get("degraded", False),
            )
            report.scores.append(score)

            if score.error:
                continue

            total_precision += score.precision
            total_recall += score.recall
            total_f1 += score.f1
            total_hit1 += score.hit_rate_1
            total_hit3 += score.hit_rate_3
            if score.mae > 0 or score.matched_pairs:
                total_mae += score.mae
                mae_count += 1
            for iou in score.iou_scores:
                total_iou += iou
                iou_count += 1
                report.iou_distribution[self.classify_iou(iou)] += 1

            cat_scores.setdefault(score.category, []).append(score.f1)
            dif_scores.setdefault(score.difficulty, []).append(score.f1)
            src_scores.setdefault(score.source_type, []).append(score.f1)
            cat_counts[score.category] = cat_counts.get(score.category, 0) + 1
            dif_counts[score.difficulty] = dif_counts.get(score.difficulty, 0) + 1
            src_counts[score.source_type] = src_counts.get(score.source_type, 0) + 1
            if score.degraded:
                cat_deg_counts[score.category] = cat_deg_counts.get(score.category, 0) + 1
                dif_deg_counts[score.difficulty] = dif_deg_counts.get(score.difficulty, 0) + 1
                src_deg_counts[score.source_type] = src_deg_counts.get(score.source_type, 0) + 1

        n = len([s for s in report.scores if not s.error]) or 1
        report.overall_precision = total_precision / n
        report.overall_recall = total_recall / n
        report.overall_f1 = total_f1 / n
        report.overall_iou = total_iou / iou_count if iou_count > 0 else 0.0
        report.overall_hit_rate_1 = total_hit1 / n
        report.overall_hit_rate_3 = total_hit3 / n
        report.overall_mae = total_mae / mae_count if mae_count > 0 else 0.0

        report.exception_count = len([s for s in report.scores if s.error])
        report.total_count = len(report.scores)
        report.exception_rate = report.exception_count / report.total_count if report.total_count > 0 else 0.0
        valid_scores = [s for s in report.scores if not s.error]
        report.degraded_count = sum(1 for s in valid_scores if s.degraded)
        report.degradation_rate = report.degraded_count / len(valid_scores) if valid_scores else 0.0

        report.cost = self._aggregate_costs(results)

        report.by_category = {
            k: {
                "f1": sum(v) / len(v),
                "count": len(v),
                "degradation_rate": cat_deg_counts.get(k, 0) / cat_counts.get(k, 1),
            }
            for k, v in cat_scores.items()
        }
        report.by_difficulty = {
            k: {
                "f1": sum(v) / len(v),
                "count": len(v),
                "degradation_rate": dif_deg_counts.get(k, 0) / dif_counts.get(k, 1),
            }
            for k, v in dif_scores.items()
        }
        report.by_source = {
            k: {
                "f1": sum(v) / len(v) if v else 0.0,
                "count": len(v),
                "degradation_rate": src_deg_counts.get(k, 0) / src_counts.get(k, 1) if src_counts.get(k, 0) > 0 else 0.0,
            }
            for k, v in src_scores.items()
        }

        return report

    def _aggregate_costs(self, results: list[dict[str, Any]]) -> CostStats:
        total_prompt = 0
        total_completion = 0
        total_duration = 0.0
        total_elapsed = 0.0
        api_calls = 0
        api_retries = 0
        memory_peaks: list[float] = []
        memory_avgs: list[float] = []
        for r in results:
            usage = r.get("usage", {})
            total_prompt += usage.get("prompt_tokens", 0)
            total_completion += usage.get("completion_tokens", 0)
            total_duration += r.get("video_duration", 0.0)
            total_elapsed += r.get("elapsed_time", 0.0)
            api_calls += r.get("api_calls", 0)
            api_retries += r.get("api_retries", 0)
            mem_peak = r.get("memory_peak_mb", 0.0)
            mem_avg = r.get("memory_avg_mb", 0.0)
            if mem_peak > 0:
                memory_peaks.append(mem_peak)
            if mem_avg > 0:
                memory_avgs.append(mem_avg)

        total_tokens = total_prompt + total_completion
        tokens_per_minute = total_tokens / (total_duration / 60.0) if total_duration > 0 else 0.0
        n = len(results) or 1
        avg_elapsed = total_elapsed / n
        processing_ratio = total_elapsed / total_duration if total_duration > 0 else 0.0
        memory_peak_mb = max(memory_peaks) if memory_peaks else 0.0
        memory_avg_mb = sum(memory_avgs) / len(memory_avgs) if memory_avgs else 0.0

        return CostStats(
            total_tokens=total_tokens,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            video_duration=total_duration,
            tokens_per_minute=tokens_per_minute,
            api_calls=api_calls,
            api_retries=api_retries,
            total_elapsed=total_elapsed,
            avg_elapsed=avg_elapsed,
            processing_ratio=processing_ratio,
            memory_peak_mb=memory_peak_mb,
            memory_avg_mb=memory_avg_mb,
            concurrent_throughput=results[0].get("concurrent_throughput", 0.0) if results else 0.0,
            concurrency=results[0].get("concurrency", 0) if results else 0,
        )


def compute_weighted_score(
    eval_report: EvalReport,
    judge_report: Any,
    weight_eval: float = 0.5,
    weight_judge: float = 0.5,
) -> dict[str, Any]:
    eval_score = eval_report.overall_f1
    judge_normalized = 0.0
    degraded = getattr(judge_report, "degraded", False)

    if not degraded and hasattr(judge_report, "overall_average") and judge_report.overall_average > 0:
        judge_normalized = judge_report.overall_average / 10.0

    if degraded:
        weighted = eval_score
    else:
        weighted = eval_score * weight_eval + judge_normalized * weight_judge

    return {
        "eval_score": round(eval_score, 4),
        "judge_score": round(judge_normalized, 4),
        "weighted_score": round(weighted, 4),
        "degraded": degraded,
    }


class TestCaseLoader:
    def __init__(self, test_cases_root: str):
        self.root = Path(test_cases_root)

    def _load_cases_from_dir(
        self, dir_name: str, source_type: str
    ) -> list[dict[str, Any]]:
        cases_yaml = self.root / dir_name / "cases.yaml"
        if not cases_yaml.exists():
            return []

        with open(cases_yaml, encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error("YAML 解析失败 [%s]: %s", cases_yaml, e)
                return []

        cases: list[dict[str, Any]] = []
        for c in data.get("cases", []):
            case_dir = self.root / dir_name / c["id"]
            instruction_path = case_dir / "instruction.json"
            gt_path = case_dir / "ground_truth.json"

            instruction = {}
            if instruction_path.exists():
                try:
                    instruction = json.loads(instruction_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning("instruction.json 解析失败 [%s]: %s", c["id"], e)
                    instruction = {"prompt": "", "parse_error": str(e)}

            ground_truth = []
            if gt_path.exists():
                try:
                    gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
                    ground_truth = gt_data.get("highlights", [])
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning("ground_truth.json 解析失败 [%s]: %s", c["id"], e)
                    ground_truth = []

            case: dict[str, Any] = {
                "case_id": c["id"],
                "category": c["category"],
                "difficulty": c["difficulty"],
                "source_type": source_type,
                "video_path": str(case_dir / c.get("video_file", "video.mp4")),
                "instruction": instruction,
                "ground_truth": ground_truth,
            }

            if source_type == "remote":
                case["source_url"] = c.get("source_url", "")

            cases.append(case)

        return cases

    def load_local_cases(self) -> list[dict[str, Any]]:
        return self._load_cases_from_dir("open_data", "local")

    def load_remote_cases(self) -> list[dict[str, Any]]:
        return self._load_cases_from_dir("self-built_data", "remote")

    def load_all(self) -> list[dict[str, Any]]:
        return self.load_local_cases() + self.load_remote_cases()
