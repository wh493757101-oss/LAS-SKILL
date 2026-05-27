import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    output_dir: str = ""
    save_charts: bool = True
    save_json: bool = True


class ReportGenerator:
    def __init__(self, config: ReportConfig | None = None):
        self.config = config or ReportConfig()

    def generate(
        self,
        eval_report: Any,
        judge_report: Any,
        weighted: dict[str, Any] | None = None,
    ) -> str:
        lines: list[str] = []

        lines.append("=" * 70)
        lines.append("视频高光剪辑 — 评测报告")
        lines.append("=" * 70)

        lines.append("")
        lines.append("## 一、时间戳 IoU 评测")
        lines.append("")
        lines.append(f"  整体 IoU:       {eval_report.overall_iou:.3f}")
        lines.append(f"  整体 Precision: {eval_report.overall_precision:.3f}")
        lines.append(f"  整体 Recall:    {eval_report.overall_recall:.3f}")
        lines.append(f"  整体 F1:        {eval_report.overall_f1:.3f}")
        lines.append(f"  Hit Rate @1:    {eval_report.overall_hit_rate_1:.3f}")
        lines.append(f"  Hit Rate @3:    {eval_report.overall_hit_rate_3:.3f}")
        lines.append(f"  MAE (时间偏差): {eval_report.overall_mae:.2f}s")
        lines.append("")
        lines.append("  tIoU 分布:")
        lines.append(f"    优秀 (≥0.8): {eval_report.iou_distribution.get('excellent', 0)}")
        lines.append(f"    合格 (≥0.5): {eval_report.iou_distribution.get('qualified', 0)}")
        lines.append(f"    不合格 (<0.5): {eval_report.iou_distribution.get('unqualified', 0)}")
        lines.append("")
        lines.append(f"  异常率:    {eval_report.exception_rate:.1%} ({eval_report.exception_count}/{eval_report.total_count})")
        lines.append(f"  降级率:    {eval_report.degradation_rate:.1%} ({eval_report.degraded_count} cases)")
        lines.append("")
        lines.append("## 性能 & 成本")
        lines.append("")
        lines.append(f"  总 Token:          {eval_report.cost.total_tokens:,}")
        lines.append(f"  Prompt Token:      {eval_report.cost.prompt_tokens:,}")
        lines.append(f"  Completion Token:  {eval_report.cost.completion_tokens:,}")
        lines.append(f"  API 调用次数:      {eval_report.cost.api_calls}")
        lines.append(f"  API 重试次数:      {eval_report.cost.api_retries}")
        lines.append(f"  视频总时长:        {eval_report.cost.video_duration:.1f}s")
        lines.append(f"  Token/分钟:        {eval_report.cost.tokens_per_minute:,.0f}")
        lines.append(f"  总处理耗时:        {eval_report.cost.total_elapsed:.1f}s")
        lines.append(f"  平均耗时/case:     {eval_report.cost.avg_elapsed:.1f}s")
        lines.append(f"  处理倍速:          {eval_report.cost.processing_ratio:.2f}x")
        if eval_report.cost.memory_peak_mb > 0:
            lines.append(f"  内存峰值:          {eval_report.cost.memory_peak_mb:.1f} MB")
            lines.append(f"  内存均值:          {eval_report.cost.memory_avg_mb:.1f} MB")
        if eval_report.cost.concurrency > 1:
            lines.append(f"  并发度:            {eval_report.cost.concurrency}")
            lines.append(f"  并发吞吐量:        {eval_report.cost.concurrent_throughput:.2f} case/s")
        lines.append("")

        if eval_report.by_category:
            lines.append("")
            lines.append("  按视频类型:")
            for cat, stats in eval_report.by_category.items():
                deg = stats.get("degradation_rate", 0.0)
                lines.append(f"    {cat}: F1={stats['f1']:.3f}, 降级率={deg:.0%} (n={stats['count']})")

        if eval_report.by_difficulty:
            lines.append("")
            lines.append("  按难度:")
            for dif, stats in eval_report.by_difficulty.items():
                deg = stats.get("degradation_rate", 0.0)
                lines.append(f"    {dif}: F1={stats['f1']:.3f}, 降级率={deg:.0%} (n={stats['count']})")

        if eval_report.by_source:
            lines.append("")
            lines.append("  按来源:")
            for src, stats in eval_report.by_source.items():
                deg = stats.get("degradation_rate", 0.0)
                lines.append(f"    {src}: F1={stats['f1']:.3f}, 降级率={deg:.0%} (n={stats['count']})")

        lines.append("")
        lines.append("  各用例详情:")
        lines.append(f"  {'ID':<12} {'类型':<8} {'难度':<8} {'来源':<6} {'Prec':<8} {'Recall':<8} {'F1':<8} {'HR@1':<8} {'MAE':<8}")
        lines.append("  " + "-" * 78)
        for score in eval_report.scores:
            if score.error:
                lines.append(f"  {score.case_id:<12} {'-':<8} {'-':<8} {'-':<6} [SKIP] {score.error}")
            else:
                lines.append(
                    f"  {score.case_id:<12} {score.category:<8} {score.difficulty:<8} "
                    f"{score.source_type:<6} {score.precision:<8.3f} {score.recall:<8.3f} "
                    f"{score.f1:<8.3f} {score.hit_rate_1:<8.3f} {score.mae:<8.2f}"
                )

        lines.append("")
        lines.append("## 二、LLM Judge 主观评测")
        lines.append("")

        if hasattr(judge_report, 'degraded') and judge_report.degraded:
            lines.append("  [降级] LLM Judge 不可用，已降级为纯量化评测")
        elif hasattr(judge_report, 'overall_average'):
            lines.append(f"  节奏感:       {judge_report.overall_rhythm:.2f} / 10.0")
            lines.append(f"  内容完整性:   {judge_report.overall_completeness:.2f} / 10.0")
            lines.append(f"  精彩程度:     {judge_report.overall_excitement:.2f} / 10.0")
            lines.append(f"  指令契合度:   {judge_report.overall_instruction_fit:.2f} / 10.0")
            lines.append(f"  综合均分:     {judge_report.overall_average:.2f} / 10.0")

            lines.append("")
            lines.append("  各用例 LLM 评价:")
            for i, score in enumerate(judge_report.scores):
                if score.error:
                    lines.append(f"    #{i + 1}: [ERROR] {score.error}")
                else:
                    lines.append(
                        f"    #{i + 1}: {score.average:.1f}/10.0 — {score.overall_comment}"
                    )

        # 加权总分
        if weighted:
            lines.append("")
            lines.append("## 三、加权总分")
            lines.append("")
            if weighted.get("degraded"):
                lines.append("  状态: LLM Judge 不可用，总分仅基于量化评测")
                lines.append(f"  加权总分: {weighted['weighted_score']:.4f} / 1.0 (纯量化)")
            else:
                eval_part = weighted["eval_score"] * 0.5
                judge_part = weighted["judge_score"] * 0.5
                lines.append(f"  量化评测分 (F1): {weighted['eval_score']:.4f} × 0.5 = {eval_part:.4f}")
                lines.append(f"  LLM Judge 分:    {weighted['judge_score']:.4f} × 0.5 = {judge_part:.4f}")
                lines.append(f"  ─────────────────────────────────")
                lines.append(f"  加权总分:        {weighted['weighted_score']:.4f} / 1.0")

        lines.append("")
        lines.append("=" * 70)

        report_text = "\n".join(lines)

        if self.config.output_dir:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            if self.config.save_json:
                json_path = out_dir / "report.json"
                json_path.write_text(
                    self._build_json(eval_report, judge_report, weighted),
                    encoding="utf-8",
                )
                logger.info("JSON 报告已保存: %s", json_path)

            text_path = out_dir / "report.txt"
            text_path.write_text(report_text, encoding="utf-8")
            logger.info("文本报告已保存: %s", text_path)

            if self.config.save_charts:
                self._save_charts(eval_report, judge_report, out_dir)

            self._upload_to_tos(out_dir)

        return report_text

    def _upload_to_tos(self, out_dir: Path) -> None:
        import os as _os

        _ak = _os.environ.get("TOS_ACCESS_KEY", "")
        _sk = _os.environ.get("TOS_SECRET_KEY", "")
        if not _ak or not _sk:
            logger.warning("TOS 凭证未配置，跳过报告上传")
            return

        try:
            import tos as _tos

            _bucket = "arkclaw-tos-2124145136-cn-guangzhou"
            _base_prefix = "arkclaw-tos-ci-yemqjzxa0w9t6r1y3a0v-lk0rj/video-highlight-bucket"
            _folder = out_dir.name
            _client = _tos.TosClientV2(_ak, _sk, "tos-cn-guangzhou.volces.com", "cn-guangzhou")

            for _f in out_dir.iterdir():
                if _f.is_file():
                    _tos_key = f"{_base_prefix}/output/{_folder}/{_f.name}"
                    _client.put_object_from_file(_bucket, _tos_key, str(_f))
                    logger.info("报告已上传到 TOS: tos://%s/%s", _bucket, _tos_key)
        except Exception as e:
            logger.warning("TOS 报告上传失败: %s", e)

    def _build_json(self, eval_report: Any, judge_report: Any, weighted: dict[str, Any] | None = None) -> str:
        data: dict[str, Any] = {
            "iou_eval": {
                "overall_iou": eval_report.overall_iou,
                "overall_precision": eval_report.overall_precision,
                "overall_recall": eval_report.overall_recall,
                "overall_f1": eval_report.overall_f1,
                "overall_hit_rate_1": round(eval_report.overall_hit_rate_1, 3),
                "overall_hit_rate_3": round(eval_report.overall_hit_rate_3, 3),
                "overall_mae": round(eval_report.overall_mae, 2),
                "iou_distribution": eval_report.iou_distribution,
                "exception_rate": round(eval_report.exception_rate, 3),
                "exception_count": eval_report.exception_count,
                "total_count": eval_report.total_count,
                "cost": {
                    "total_tokens": eval_report.cost.total_tokens,
                    "prompt_tokens": eval_report.cost.prompt_tokens,
                    "completion_tokens": eval_report.cost.completion_tokens,
                    "video_duration": eval_report.cost.video_duration,
                    "tokens_per_minute": round(eval_report.cost.tokens_per_minute, 1),
                    "api_calls": eval_report.cost.api_calls,
                    "api_retries": eval_report.cost.api_retries,
                    "total_elapsed": round(eval_report.cost.total_elapsed, 1),
                    "avg_elapsed": round(eval_report.cost.avg_elapsed, 1),
                    "processing_ratio": round(eval_report.cost.processing_ratio, 2),
                    "memory_peak_mb": round(eval_report.cost.memory_peak_mb, 1),
                    "memory_avg_mb": round(eval_report.cost.memory_avg_mb, 1),
                    "concurrency": eval_report.cost.concurrency,
                    "concurrent_throughput": round(eval_report.cost.concurrent_throughput, 2),
                },
                "degradation_rate": round(eval_report.degradation_rate, 3),
                "by_category": {
                    k: {
                        "f1": round(v["f1"], 3),
                        "count": v["count"],
                        "degradation_rate": round(v.get("degradation_rate", 0), 3),
                    }
                    for k, v in eval_report.by_category.items()
                },
                "by_difficulty": {
                    k: {
                        "f1": round(v["f1"], 3),
                        "count": v["count"],
                        "degradation_rate": round(v.get("degradation_rate", 0), 3),
                    }
                    for k, v in eval_report.by_difficulty.items()
                },
                "by_source": {
                    k: {
                        "f1": round(v["f1"], 3),
                        "count": v["count"],
                        "degradation_rate": round(v.get("degradation_rate", 0), 3),
                    }
                    for k, v in eval_report.by_source.items()
                },
                "cases": [
                    {
                        "case_id": s.case_id,
                        "category": s.category,
                        "difficulty": s.difficulty,
                        "source_type": s.source_type,
                        "precision": round(s.precision, 3),
                        "recall": round(s.recall, 3),
                        "f1": round(s.f1, 3),
                        "hit_rate_1": round(s.hit_rate_1, 3),
                        "hit_rate_3": round(s.hit_rate_3, 3),
                        "mae": round(s.mae, 2),
                        "iou_distribution": s.iou_distribution,
                        "error": s.error,
                    }
                    for s in eval_report.scores
                ],
            },
        }

        if hasattr(judge_report, 'overall_average'):
            data["llm_judge"] = {
                "overall_rhythm": round(judge_report.overall_rhythm, 2),
                "overall_completeness": round(judge_report.overall_completeness, 2),
                "overall_excitement": round(judge_report.overall_excitement, 2),
                "overall_instruction_fit": round(judge_report.overall_instruction_fit, 2),
                "overall_average": round(judge_report.overall_average, 2),
                "degraded": getattr(judge_report, "degraded", False),
                "cases": [
                    {
                        "rhythm": s.rhythm,
                        "completeness": s.completeness,
                        "excitement": s.excitement,
                        "instruction_fit": s.instruction_fit,
                        "average": round(s.average, 1),
                        "comment": s.overall_comment,
                        "error": s.error,
                    }
                    for s in judge_report.scores
                ],
            }

        if weighted:
            data["weighted_score"] = weighted

        return json.dumps(data, ensure_ascii=False, indent=2)

    def _save_charts(
        self,
        eval_report: Any,
        judge_report: Any,
        out_dir: Path,
    ) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            logger.warning("matplotlib 未安装，跳过图表生成")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. F1 by category
        if eval_report.by_category:
            ax = axes[0][0]
            cats = list(eval_report.by_category.keys())
            f1s = [eval_report.by_category[c]["f1"] for c in cats]
            colors = plt.cm.Set2(np.linspace(0, 1, len(cats)))
            ax.bar(cats, f1s, color=colors)
            ax.set_title("F1 by Category")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            for i, v in enumerate(f1s):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        # 2. F1 by difficulty
        if eval_report.by_difficulty:
            ax = axes[0][1]
            difs = list(eval_report.by_difficulty.keys())
            f1s = [eval_report.by_difficulty[d]["f1"] for d in difs]
            colors = ["#4CAF50" if d == "easy" else "#FF9800" if d == "medium" else "#F44336" for d in difs]
            ax.bar(difs, f1s, color=colors)
            ax.set_title("F1 by Difficulty")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            for i, v in enumerate(f1s):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        # 3. F1 by source
        if eval_report.by_source:
            ax = axes[1][0]
            srcs = list(eval_report.by_source.keys())
            f1s = [eval_report.by_source[s]["f1"] for s in srcs]
            ax.bar(srcs, f1s, color=["#2196F3", "#FF5722"])
            ax.set_title("F1 by Source")
            ax.set_ylabel("F1")
            ax.set_ylim(0, 1)
            for i, v in enumerate(f1s):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        # 4. LLM Judge radar
        if hasattr(judge_report, 'overall_average') and judge_report.overall_average > 0:
            ax = axes[1][1]
            ax.set_title("LLM Judge")
            dims = ["Rhythm", "Completeness", "Excitement", "Fit"]
            values = [
                judge_report.overall_rhythm,
                judge_report.overall_completeness,
                judge_report.overall_excitement,
                judge_report.overall_instruction_fit,
            ]
            x = np.arange(len(dims))
            ax.bar(x, values, color=plt.cm.Set3(np.linspace(0, 1, 4)))
            ax.set_xticks(x)
            ax.set_xticklabels(dims)
            ax.set_ylabel("Score")
            ax.set_ylim(0, 10)
            for i, v in enumerate(values):
                ax.text(i, v + 0.1, f"{v:.1f}", ha="center")

        plt.tight_layout()
        chart_path = out_dir / "charts.png"
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("图表已保存: %s", chart_path)
