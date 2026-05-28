import json
from pathlib import Path

import pytest

from evaluation.evaluator import (
    MULTI_IOU_THRESHOLDS,
    CaseScore,
    CostStats,
    EvalReport,
    HighlightEvaluator,
    SegmentMatch,
    TestCaseLoader,
    compute_weighted_score,
    parse_target_duration,
)
from evaluation.llm_judge import (
    JudgeReport,
    JudgeScore,
    LLMJudge,
    SegmentJudgeScore,
    VideoJudgeScore,
    format_judge_report,
)
from evaluation.report import ReportConfig, ReportGenerator
from evaluation.runner import EvalRunConfig, EvalRunner


class TestHighlightEvaluator:
    def test_compute_iou_perfect_match(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(0.0, 5.0, 0.0, 5.0)
        assert iou == 1.0

    def test_compute_iou_no_overlap(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(0.0, 2.0, 3.0, 5.0)
        assert iou == 0.0

    def test_compute_iou_partial_overlap(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(0.0, 4.0, 2.0, 6.0)
        assert 0.0 < iou < 1.0

    def test_compute_iou_zero_union(self):
        evaluator = HighlightEvaluator()
        iou = evaluator.compute_iou(1.0, 1.0, 1.0, 1.0)
        assert iou == 0.0

    def test_match_segments_perfect(self):
        evaluator = HighlightEvaluator(iou_threshold=0.5)
        predicted = [{"start_time": 0.0, "end_time": 5.0}]
        ground_truth = [{"start_time": 0.0, "end_time": 5.0}]

        matches = evaluator.match_segments(predicted, ground_truth)
        assert len(matches) == 1
        assert matches[0].hit is True
        assert matches[0].iou == 1.0

    def test_match_segments_no_hit(self):
        evaluator = HighlightEvaluator(iou_threshold=0.5)
        predicted = [{"start_time": 0.0, "end_time": 2.0}]
        ground_truth = [{"start_time": 5.0, "end_time": 7.0}]

        matches = evaluator.match_segments(predicted, ground_truth)
        assert len(matches) == 1
        assert matches[0].hit is False

    def test_match_segments_one_to_one_greedy(self):
        evaluator = HighlightEvaluator(iou_threshold=0.5)
        predicted = [
            {"start_time": 0.0, "end_time": 4.0},
            {"start_time": 6.0, "end_time": 10.0},
        ]
        ground_truth = [
            {"start_time": 0.5, "end_time": 3.5},
            {"start_time": 6.5, "end_time": 9.5},
        ]

        matches = evaluator.match_segments(predicted, ground_truth)
        assert len(matches) == 2
        assert all(m.hit for m in matches)

    def test_score_case_basic(self):
        evaluator = HighlightEvaluator()
        predicted = [{"start_time": 0.0, "end_time": 5.0}]
        ground_truth = [{"start_time": 0.0, "end_time": 5.0}]

        score = evaluator.score_case(
            "case_001", predicted, ground_truth,
            category="体育", difficulty="easy", source_type="local",
            video_duration=60.0,
        )

        assert score.case_id == "case_001"
        assert score.precision == 1.0
        assert score.recall == 1.0
        assert score.f1 == 1.0
        assert score.hit_rate_1 == 1.0
        assert score.hit_rate_3 == 1.0
        assert score.mae == 0.0
        assert score.iou_distribution["excellent"] == 1
        assert score.segment_count_deviation == 0.0
        assert score.total_duration_ratio == pytest.approx(5.0 / 60.0)
        assert score.instruction_duration_fit == 1.0
        assert score.map_50 == 1.0
        assert score.map_75 == 1.0
        assert score.avg_map == 1.0
        assert score.kendall_tau is None
        assert score.spearman_rho is None
        assert score.error is None

    def test_score_case_empty_gt(self):
        evaluator = HighlightEvaluator()
        score = evaluator.score_case("case_001", [], [])
        assert score.error == "ground_truth 为空"

    def test_score_case_empty_pred(self):
        evaluator = HighlightEvaluator()
        score = evaluator.score_case(
            "case_001", [],
            [{"start_time": 0.0, "end_time": 5.0}],
        )
        assert score.precision == 0.0
        assert score.recall == 0.0

    def test_evaluate_all(self):
        evaluator = HighlightEvaluator()
        results = [
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
            {
                "case_id": "case_002",
                "category": "游戏",
                "difficulty": "medium",
                "source_type": "remote",
                "predicted": [{"start_time": 2.0, "end_time": 6.0}],
                "ground_truth": [{"start_time": 2.0, "end_time": 6.0}],
            },
        ]

        report = evaluator.evaluate_all(results)
        assert report.overall_f1 == 1.0
        assert report.overall_micro_f1 == 1.0
        assert report.overall_hit_rate_1 == 1.0
        assert report.overall_hit_rate_3 == 1.0
        assert report.overall_mae == 0.0
        assert report.overall_segment_count_deviation == 0.0
        assert report.overall_map_50 == 1.0
        assert report.overall_map_75 == 1.0
        assert report.overall_avg_map == 1.0
        assert report.iou_distribution["excellent"] == 2
        assert report.exception_rate == 0.0
        assert len(report.scores) == 2
        assert "体育" in report.by_category
        assert "游戏" in report.by_category
        assert "easy" in report.by_difficulty
        assert "medium" in report.by_difficulty
        assert "local" in report.by_source
        assert "remote" in report.by_source

    def test_evaluate_all_empty(self):
        evaluator = HighlightEvaluator()
        report = evaluator.evaluate_all([])
        assert report.overall_f1 == 0.0

    def test_hit_rate(self):
        evaluator = HighlightEvaluator(iou_threshold=0.5)
        predicted = [
            {"start_time": 10.0, "end_time": 12.0},
            {"start_time": 0.0, "end_time": 4.0},
            {"start_time": 5.0, "end_time": 8.0},
        ]
        ground_truth = [
            {"start_time": 0.0, "end_time": 4.0},
            {"start_time": 5.0, "end_time": 8.0},
        ]
        assert evaluator.compute_hit_rate(predicted, ground_truth, 1) == 0.0
        assert evaluator.compute_hit_rate(predicted, ground_truth, 3) == 2.0 / 3.0

    def test_mae_computation(self):
        evaluator = HighlightEvaluator()
        predicted = [{"start_time": 1.0, "end_time": 6.0}]
        ground_truth = [{"start_time": 0.0, "end_time": 5.0}]
        score = evaluator.score_case("case_001", predicted, ground_truth)
        assert score.mae == 1.0

    def test_iou_distribution(self):
        evaluator = HighlightEvaluator()
        predicted = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 6.0, "end_time": 9.0},
            {"start_time": 10.0, "end_time": 12.0},
        ]
        ground_truth = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 5.5, "end_time": 9.5},
            {"start_time": 50.0, "end_time": 60.0},
        ]
        score = evaluator.score_case("case_001", predicted, ground_truth)
        assert score.iou_distribution["excellent"] == 1
        assert score.iou_distribution["qualified"] >= 1
        assert score.iou_distribution["unqualified"] == 1

    def test_exception_rate(self):
        evaluator = HighlightEvaluator()
        results = [
            {"case_id": "c1", "predicted": [], "ground_truth": [], "category": "边界", "difficulty": "edge", "source_type": "local", "usage": {}, "video_duration": 0.0},
            {"case_id": "c2", "predicted": [{"start_time": 0, "end_time": 5}], "ground_truth": [{"start_time": 0, "end_time": 5}], "category": "体育", "difficulty": "easy", "source_type": "local", "usage": {}, "video_duration": 120.0},
        ]
        report = evaluator.evaluate_all(results)
        assert report.exception_count == 1
        assert report.exception_rate == 0.5

    def test_cost_aggregation(self):
        evaluator = HighlightEvaluator()
        results = [
            {
                "case_id": "c1",
                "predicted": [{"start_time": 0, "end_time": 5}],
                "ground_truth": [{"start_time": 0, "end_time": 5}],
                "category": "体育", "difficulty": "easy", "source_type": "local",
                "usage": {"prompt_tokens": 5000, "completion_tokens": 500},
                "video_duration": 60.0,
            },
            {
                "case_id": "c2",
                "predicted": [{"start_time": 0, "end_time": 5}],
                "ground_truth": [{"start_time": 0, "end_time": 5}],
                "category": "体育", "difficulty": "easy", "source_type": "local",
                "usage": {"prompt_tokens": 3000, "completion_tokens": 300},
                "video_duration": 120.0,
            },
        ]
        report = evaluator.evaluate_all(results)
        assert report.cost.total_tokens == 8800
        assert report.cost.prompt_tokens == 8000
        assert report.cost.completion_tokens == 800
        assert report.cost.video_duration == 180.0
        assert report.cost.tokens_per_minute == pytest.approx(2933.33, rel=0.01)


class TestParseTargetDuration:
    def test_seconds(self):
        assert parse_target_duration("剪辑60秒精彩集锦") == 60.0

    def test_minutes(self):
        assert parse_target_duration("剪1分钟高潮片段") == 60.0

    def test_english_s(self):
        assert parse_target_duration("make a 30s highlight") == 30.0

    def test_english_min(self):
        assert parse_target_duration("cut to 2min reel") == 120.0

    def test_no_duration(self):
        assert parse_target_duration("把精彩片段剪出来") is None

    def test_empty(self):
        assert parse_target_duration("") is None


class TestMultiIoUMap:
    def test_compute_map_perfect_match(self):
        evaluator = HighlightEvaluator()
        predicted = [{"start_time": 0.0, "end_time": 5.0}]
        ground_truth = [{"start_time": 0.0, "end_time": 5.0}]
        map_50, map_75, avg_map = evaluator.compute_multi_iou_map(predicted, ground_truth)
        assert map_50 == 1.0
        assert map_75 == 1.0
        assert avg_map == 1.0

    def test_compute_map_no_match(self):
        evaluator = HighlightEvaluator()
        predicted = [{"start_time": 0.0, "end_time": 2.0}]
        ground_truth = [{"start_time": 5.0, "end_time": 7.0}]
        map_50, map_75, avg_map = evaluator.compute_multi_iou_map(predicted, ground_truth)
        assert map_50 == 0.0
        assert map_75 == 0.0
        assert avg_map == 0.0

    def test_compute_map_partial(self):
        evaluator = HighlightEvaluator()
        predicted = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 10.0, "end_time": 15.0},
        ]
        ground_truth = [
            {"start_time": 0.0, "end_time": 5.0},
            {"start_time": 50.0, "end_time": 55.0},
        ]
        map_50, map_75, avg_map = evaluator.compute_multi_iou_map(predicted, ground_truth)
        assert map_50 == 0.75  # 第1个命中，第2个未命中 → AP = (1/1 + 1/2)/2
        assert map_75 == 0.75
        assert 0.0 < avg_map < 1.0

    def test_compute_map_empty(self):
        evaluator = HighlightEvaluator()
        assert evaluator.compute_multi_iou_map([], []) == (0.0, 0.0, 0.0)
        assert evaluator.compute_multi_iou_map(
            [{"start_time": 0, "end_time": 5}], [],
        ) == (0.0, 0.0, 0.0)

    def test_multi_iou_thresholds_range(self):
        assert MULTI_IOU_THRESHOLDS[0] == 0.5
        assert MULTI_IOU_THRESHOLDS[-1] == 0.95
        assert len(MULTI_IOU_THRESHOLDS) == 10


class TestRankCorrelation:
    def test_rank_correlation_perfect(self):
        evaluator = HighlightEvaluator()
        predicted = [
            {"start_time": 0.0, "end_time": 2.0, "score": 0.1},
            {"start_time": 3.0, "end_time": 5.0, "score": 0.9},
            {"start_time": 6.0, "end_time": 8.0, "score": 0.5},
        ]
        ground_truth = [
            {"start_time": 0.0, "end_time": 2.0, "score": 0.1},
            {"start_time": 3.0, "end_time": 5.0, "score": 0.9},
            {"start_time": 6.0, "end_time": 8.0, "score": 0.5},
        ]
        tau, rho = evaluator.compute_rank_correlation(predicted, ground_truth)
        assert tau == 1.0
        assert rho == 1.0

    def test_rank_correlation_insufficient_data(self):
        evaluator = HighlightEvaluator()
        predicted = [{"start_time": 0.0, "end_time": 2.0, "score": 0.5}]
        ground_truth = [{"start_time": 0.0, "end_time": 2.0, "score": 0.5}]
        tau, rho = evaluator.compute_rank_correlation(predicted, ground_truth)
        assert tau is None
        assert rho is None

    def test_rank_correlation_no_scores(self):
        evaluator = HighlightEvaluator()
        predicted = [
            {"start_time": 0.0, "end_time": 2.0},
            {"start_time": 3.0, "end_time": 5.0},
            {"start_time": 6.0, "end_time": 8.0},
        ]
        ground_truth = [
            {"start_time": 0.0, "end_time": 2.0},
            {"start_time": 3.0, "end_time": 5.0},
            {"start_time": 6.0, "end_time": 8.0},
        ]
        tau, rho = evaluator.compute_rank_correlation(predicted, ground_truth)
        assert tau is None
        assert rho is None

    def test_rank_correlation_mismatched_lengths(self):
        evaluator = HighlightEvaluator()
        predicted = [
            {"start_time": 0.0, "end_time": 2.0, "score": 0.3},
            {"start_time": 3.0, "end_time": 5.0, "score": 0.7},
            {"start_time": 6.0, "end_time": 8.0, "score": 0.5},
        ]
        ground_truth = [
            {"start_time": 0.0, "end_time": 2.0, "score": 0.3},
            {"start_time": 3.0, "end_time": 5.0, "score": 0.7},
        ]
        tau, rho = evaluator.compute_rank_correlation(predicted, ground_truth)
        assert tau == 1.0
        assert rho == pytest.approx(1.0)


class TestLLMJudge:
    def test_build_prompt(self):
        judge = LLMJudge()
        segments = [
            {"start_time": 0.0, "end_time": 5.0, "score": 0.9, "label": "进球"},
        ]
        prompt = judge.build_prompt("体育", "进球集锦", "快节奏", segments)

        assert "体育" in prompt
        assert "进球集锦" in prompt
        assert "快节奏" in prompt
        assert "0.0s - 5.0s" in prompt
        assert "进球" in prompt

    def test_build_prompt_no_target_style(self):
        judge = LLMJudge()
        segments: list = []
        prompt = judge.build_prompt("体育", "", "", segments)
        assert "精彩集锦" in prompt
        assert "无特定要求" in prompt

    def test_judge_score_average(self):
        score = JudgeScore(rhythm=4.0, transition_quality=3.0, audiovisual_sync=5.0, completeness=4.0, instruction_fit=4.0)
        assert score.average == 4.0

    def test_judge_score_error(self):
        score = JudgeScore(error="API 不可用")
        assert score.error == "API 不可用"
        assert score.average == 0.0

    def test_judge_all_empty(self):
        judge = LLMJudge()
        report = judge.judge_all([])
        assert report.overall_average == 0.0
        assert report.segment_average == 0.0
        assert report.video_average == 0.0

    def test_format_judge_report(self):
        report = JudgeReport(
            scores=[
                JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, completeness=4.0, instruction_fit=4.0, overall_comment="剪辑质量优秀"),
            ],
            segment_scores=[
                SegmentJudgeScore(content_completeness=4.0, segment_quality=5.0, instruction_fit=4.0, overall_comment="片段质量不错"),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, content_completeness=4.0, instruction_fit=4.0, overall_comment="剪辑质量优秀"),
            ],
            overall_rhythm=4.0,
            overall_transition_quality=4.0,
            overall_audiovisual_sync=4.0,
            overall_completeness=4.0,
            overall_instruction_fit=4.0,
            overall_average=4.0,
            segment_content_completeness=4.0,
            segment_quality=5.0,
            segment_instruction_fit=4.0,
            segment_average=4.33,
            video_rhythm=4.0,
            video_transition_quality=4.0,
            video_audiovisual_sync=4.0,
            video_content_completeness=4.0,
            video_instruction_fit=4.0,
            video_average=4.0,
        )
        formatted = format_judge_report(report)
        assert "Segment Judge" in formatted
        assert "Video Judge" in formatted
        assert "剪辑质量优秀" in formatted

    def test_format_judge_report_with_error(self):
        report = JudgeReport(
            scores=[JudgeScore(error="API 不可用")],
            segment_scores=[SegmentJudgeScore(error="API 不可用")],
            video_scores=[VideoJudgeScore(error="API 不可用")],
        )
        formatted = format_judge_report(report)
        assert "[ERROR]" in formatted
        assert "API 不可用" in formatted


class TestReportGenerator:
    def test_generate_text_report(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])

        judge_report = JudgeReport(
            scores=[
                JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, completeness=4.0, instruction_fit=4.0, overall_comment="不错"),
            ],
            segment_scores=[
                SegmentJudgeScore(content_completeness=4.0, segment_quality=5.0, instruction_fit=4.0, overall_comment="片段不错"),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, content_completeness=4.0, instruction_fit=4.0, overall_comment="集锦不错"),
            ],
            overall_rhythm=4.0,
            overall_transition_quality=4.0,
            overall_audiovisual_sync=4.0,
            overall_completeness=4.0,
            overall_instruction_fit=4.0,
            overall_average=4.0,
            segment_content_completeness=4.0,
            segment_quality=5.0,
            segment_instruction_fit=4.0,
            segment_average=4.33,
            video_rhythm=4.0,
            video_transition_quality=4.0,
            video_audiovisual_sync=4.0,
            video_content_completeness=4.0,
            video_instruction_fit=4.0,
            video_average=4.0,
        )

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=False))
        text = gen.generate(eval_report, judge_report)

        assert "视频高光剪辑" in text
        assert "case_001" in text
        assert "Segment Judge" in text
        assert "Video Judge" in text
        assert "不错" in text

    def test_generate_json_report(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])

        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, completeness=4.0, instruction_fit=4.0, overall_comment="不错")],
            segment_scores=[
                SegmentJudgeScore(content_completeness=4.0, segment_quality=5.0, instruction_fit=4.0, overall_comment="片段不错"),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, content_completeness=4.0, instruction_fit=4.0, overall_comment="集锦不错"),
            ],
            overall_rhythm=4.0,
            overall_transition_quality=4.0,
            overall_audiovisual_sync=4.0,
            overall_completeness=4.0,
            overall_instruction_fit=4.0,
            overall_average=4.0,
            segment_content_completeness=4.0,
            segment_quality=5.0,
            segment_instruction_fit=4.0,
            segment_average=4.33,
            video_rhythm=4.0,
            video_transition_quality=4.0,
            video_audiovisual_sync=4.0,
            video_content_completeness=4.0,
            video_instruction_fit=4.0,
            video_average=4.0,
        )

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=False))
        gen.generate(eval_report, judge_report)

        json_path = tmp_path / "report.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["iou_eval"]["overall_f1"] == 1.0
        assert data["segment_judge"]["average"] == 4.33
        assert data["video_judge"]["average"] == 4.0
        assert data["llm_judge"]["overall_average"] == 4.0

    def test_generate_charts(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])

        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, completeness=4.0, instruction_fit=4.0, overall_comment="不错")],
            segment_scores=[
                SegmentJudgeScore(content_completeness=4.0, segment_quality=5.0, instruction_fit=4.0, overall_comment="片段不错"),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, content_completeness=4.0, instruction_fit=4.0, overall_comment="集锦不错"),
            ],
            overall_rhythm=4.0,
            overall_transition_quality=4.0,
            overall_audiovisual_sync=4.0,
            overall_completeness=4.0,
            overall_instruction_fit=4.0,
            overall_average=4.0,
            video_rhythm=4.0,
            video_transition_quality=4.0,
            video_audiovisual_sync=4.0,
            video_content_completeness=4.0,
            video_instruction_fit=4.0,
            video_average=4.0,
        )

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=True))
        gen.generate(eval_report, judge_report)

        chart_path = tmp_path / "charts.png"
        assert chart_path.exists()

    def test_report_config_defaults(self):
        cfg = ReportConfig()
        assert cfg.save_charts is True
        assert cfg.save_json is True


class TestTestCaseLoader:
    def test_load_local_cases(self, tmp_path):
        local_dir = tmp_path / "open_data"
        local_dir.mkdir()
        cases_yaml = local_dir / "cases.yaml"
        cases_yaml.write_text("""
cases:
  - id: "case_001"
    category: 体育
    difficulty: easy
    description: "测试用例"
    video_file: "video.mp4"
    instruction:
      prompt: "帮我把这个足球视频的进球剪成60秒集锦，节奏要快"
""", encoding="utf-8")

        case_dir = local_dir / "case_001"
        case_dir.mkdir()
        (case_dir / "instruction.json").write_text(
            '{"prompt": "帮我把这个足球视频的进球剪成60秒集锦，节奏要快"}',
            encoding="utf-8",
        )
        (case_dir / "ground_truth.json").write_text(
            '{"highlights": [{"start_time": 0.0, "end_time": 5.0, "label": "进球", "score": 0.95}]}',
            encoding="utf-8",
        )

        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_local_cases()

        assert len(cases) == 1
        assert cases[0]["case_id"] == "case_001"
        assert cases[0]["category"] == "体育"
        assert cases[0]["source_type"] == "local"
        assert len(cases[0]["ground_truth"]) == 1

    def test_load_remote_cases(self, tmp_path):
        remote_dir = tmp_path / "self-built_data"
        remote_dir.mkdir()
        cases_yaml = remote_dir / "cases.yaml"
        cases_yaml.write_text("""
cases:
  - id: "case_021"
    category: 体育
    difficulty: medium
    description: "远程测试"
    source_url: "https://example.com/video.mp4"
    instruction:
      target: "精彩回合"
      duration_limit: "60秒以内"
      style: ""
""", encoding="utf-8")

        case_dir = remote_dir / "case_021"
        case_dir.mkdir()
        (case_dir / "instruction.json").write_text(
            '{"target": "精彩回合", "duration_limit": "60秒以内", "style": ""}',
            encoding="utf-8",
        )
        (case_dir / "ground_truth.json").write_text(
            '{"highlights": []}',
            encoding="utf-8",
        )

        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_remote_cases()

        assert len(cases) == 1
        assert cases[0]["case_id"] == "case_021"
        assert cases[0]["source_type"] == "remote"
        assert cases[0]["source_url"] == "https://example.com/video.mp4"

    def test_load_all(self, tmp_path):
        local_dir = tmp_path / "open_data"
        local_dir.mkdir()
        (local_dir / "cases.yaml").write_text("cases: []", encoding="utf-8")

        remote_dir = tmp_path / "self-built_data"
        remote_dir.mkdir()
        (remote_dir / "cases.yaml").write_text("cases: []", encoding="utf-8")

        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_all()
        assert cases == []

    def test_load_missing_yaml(self, tmp_path):
        loader = TestCaseLoader(str(tmp_path))
        cases = loader.load_local_cases()
        assert cases == []


class TestEvalRunner:
    def test_run_no_cases(self, tmp_path):
        local_dir = tmp_path / "open_data"
        local_dir.mkdir()
        (local_dir / "cases.yaml").write_text("cases: []", encoding="utf-8")

        remote_dir = tmp_path / "self-built_data"
        remote_dir.mkdir()
        (remote_dir / "cases.yaml").write_text("cases: []", encoding="utf-8")

        runner = EvalRunner(EvalRunConfig(test_cases_root=str(tmp_path)))
        eval_report, judge_report, text = runner.run()
        assert eval_report.overall_f1 == 0.0
        assert judge_report.overall_average == 0.0

    def test_run_with_mock_pipeline(self, mocker, tmp_path):
        local_dir = tmp_path / "open_data"
        local_dir.mkdir()
        cases_yaml = local_dir / "cases.yaml"
        cases_yaml.write_text("""
cases:
  - id: "case_001"
    category: 体育
    difficulty: easy
    description: "测试"
    video_file: "video.mp4"
    instruction:
      prompt: "测试指令"
""", encoding="utf-8")

        case_dir = local_dir / "case_001"
        case_dir.mkdir()
        (case_dir / "instruction.json").write_text(
            '{"prompt": "测试指令"}', encoding="utf-8",
        )
        (case_dir / "ground_truth.json").write_text(
            '{"highlights": [{"start_time": 0.0, "end_time": 5.0, "score": 0.9}]}',
            encoding="utf-8",
        )
        (case_dir / "video.mp4").write_bytes(b"fake mp4")

        mocker.patch.object(
            EvalRunner, "_run_case",
            return_value={
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [
                    {"start_time": 1.0, "end_time": 4.0, "score": 0.8},
                ],
                "ground_truth": [
                    {"start_time": 0.0, "end_time": 5.0, "score": 0.9},
                ],
                "target": "测试指令",
                "style": "",
            },
        )

        runner = EvalRunner(EvalRunConfig(
            test_cases_root=str(tmp_path),
            skip_llm_judge=True,
        ))
        eval_report, judge_report, text = runner.run()

        assert len(eval_report.scores) == 1
        assert eval_report.scores[0].case_id == "case_001"
        assert eval_report.scores[0].precision > 0
        assert "IoU" in text

    def test_case_filter(self, mocker, tmp_path):
        local_dir = tmp_path / "open_data"
        local_dir.mkdir()
        cases_yaml = local_dir / "cases.yaml"
        cases_yaml.write_text("""
cases:
  - id: "case_001"
    category: 体育
    difficulty: easy
    description: "测试1"
    video_file: "video.mp4"
    instruction:
      prompt: "指令1"
  - id: "case_002"
    category: 户外
    difficulty: medium
    description: "测试2"
    video_file: "video.mp4"
    instruction:
      prompt: "指令2"
""", encoding="utf-8")

        for cid in ["case_001", "case_002"]:
            case_dir = local_dir / cid
            case_dir.mkdir()
            (case_dir / "instruction.json").write_text(
                f'{{"prompt": "指令"}}', encoding="utf-8",
            )
            (case_dir / "ground_truth.json").write_text(
                '{"highlights": []}', encoding="utf-8",
            )
            (case_dir / "video.mp4").write_bytes(b"fake mp4")

        mocker.patch.object(EvalRunner, "_run_case", return_value={
            "case_id": "case_001",
            "category": "体育",
            "difficulty": "easy",
            "source_type": "local",
            "predicted": [],
            "ground_truth": [],
            "target": "",
            "style": "",
        })

        runner = EvalRunner(EvalRunConfig(
            test_cases_root=str(tmp_path),
            skip_llm_judge=True,
            case_filter=["case_001"],
        ))
        eval_report, _, _ = runner.run()
        assert len(eval_report.scores) == 1
        assert eval_report.scores[0].case_id == "case_001"


class TestWeightedScore:
    def test_normal(self):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])
        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, completeness=4.0, instruction_fit=4.0)],
            segment_scores=[
                SegmentJudgeScore(content_completeness=4.0, segment_quality=4.0, instruction_fit=4.0),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, content_completeness=4.0, instruction_fit=4.0),
            ],
            overall_rhythm=4.0,
            overall_transition_quality=4.0,
            overall_audiovisual_sync=4.0,
            overall_completeness=4.0,
            overall_instruction_fit=4.0,
            overall_average=4.0,
            segment_content_completeness=4.0,
            segment_quality=4.0,
            segment_instruction_fit=4.0,
            segment_average=4.0,
            video_rhythm=4.0,
            video_transition_quality=4.0,
            video_audiovisual_sync=4.0,
            video_content_completeness=4.0,
            video_instruction_fit=4.0,
            video_average=4.0,
        )

        result = compute_weighted_score(eval_report, judge_report)
        assert result["eval_score"] == 1.0
        assert result["judge_score"] == pytest.approx(0.4)
        assert result["segment_judge_score"] == pytest.approx(0.4)
        assert result["video_judge_score"] == pytest.approx(0.4)
        assert result["weighted_score"] == pytest.approx(0.7)
        assert result["degraded"] is False

    def test_degraded(self):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])
        judge_report = JudgeReport(degraded=True, segment_degraded=True, video_degraded=True)

        result = compute_weighted_score(eval_report, judge_report)
        assert result["eval_score"] == 1.0
        assert result["judge_score"] == 0.0
        assert result["weighted_score"] == 1.0
        assert result["degraded"] is True

    def test_custom_weights(self):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])
        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=10.0, transition_quality=10.0, audiovisual_sync=10.0, completeness=10.0, instruction_fit=10.0)],
            segment_scores=[
                SegmentJudgeScore(content_completeness=10.0, segment_quality=10.0, instruction_fit=10.0),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=10.0, transition_quality=10.0, audiovisual_sync=10.0, content_completeness=10.0, instruction_fit=10.0),
            ],
            overall_rhythm=10.0,
            overall_transition_quality=10.0,
            overall_audiovisual_sync=10.0,
            overall_completeness=10.0,
            overall_instruction_fit=10.0,
            overall_average=10.0,
            segment_content_completeness=10.0,
            segment_quality=10.0,
            segment_instruction_fit=10.0,
            segment_average=10.0,
            video_rhythm=10.0,
            video_transition_quality=10.0,
            video_audiovisual_sync=10.0,
            video_content_completeness=10.0,
            video_instruction_fit=10.0,
            video_average=10.0,
        )

        result = compute_weighted_score(eval_report, judge_report, weight_eval=0.6, weight_judge=0.4)
        assert result["weighted_score"] == 1.0 * 0.6 + 1.0 * 0.4

    def test_judge_all_degraded(self):
        judge = LLMJudge()
        report = judge.judge_all([])
        assert report.degraded is False

        report2 = JudgeReport(degraded=True, segment_degraded=True, video_degraded=True)
        assert report2.degraded is True


class TestLLMJudgeRetry:
    def test_judge_retry_success_after_failure(self, mocker):
        judge = LLMJudge()
        mock_client = mocker.MagicMock()
        mock_client.chat.side_effect = [
            RuntimeError("临时错误"),
            RuntimeError("临时错误"),
            {
                "choices": [{
                    "message": {
                        "content": '{"节奏感": 4, "转场质量": 4, "音画同步": 5, "内容完整性": 4, "指令契合度": 4, "主要优点": "不错"}'
                    }
                }]
            },
        ]
        mock_client.extract_json.side_effect = lambda r: {
            "节奏感": 4, "转场质量": 4, "音画同步": 5, "内容完整性": 4, "指令契合度": 4, "主要优点": "不错"
        }
        judge._ark_client = mock_client

        score = judge.judge("体育", "测试", "", [], max_retries=3)
        assert score.error is None
        assert score.rhythm == 4.0
        assert score.audiovisual_sync == 5.0
        assert mock_client.chat.call_count == 3

    def test_judge_retry_all_fail(self, mocker):
        judge = LLMJudge()
        mock_client = mocker.MagicMock()
        mock_client.chat.side_effect = RuntimeError("API 不可用")
        judge._ark_client = mock_client

        score = judge.judge("体育", "测试", "", [], max_retries=3)
        assert score.error is not None
        assert "API 不可用" in score.error
        assert mock_client.chat.call_count == 3

    def test_judge_all_with_retries_sets_degraded(self, mocker):
        judge = LLMJudge()
        mock_client = mocker.MagicMock()
        mock_client.chat.side_effect = RuntimeError("全部失败")
        mock_client.extract_json.side_effect = RuntimeError("全部失败")
        judge._ark_client = mock_client

        cases = [
            {"category": "体育", "target": "测试", "style": "", "segments": [{"start_time": 0, "end_time": 5, "score": 0.9}]},
        ]
        report = judge.judge_all(cases, max_retries=2)
        assert report.degraded is True
        assert report.segment_degraded is True
        assert report.video_degraded is True
        assert report.overall_average == 0.0
        assert len(report.scores) == 1
        assert report.scores[0].error is not None


class TestReportWithWeightedScore:
    def test_generate_with_weighted(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])
        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, completeness=4.0, instruction_fit=4.0, overall_comment="不错")],
            segment_scores=[
                SegmentJudgeScore(content_completeness=4.0, segment_quality=4.0, instruction_fit=4.0, overall_comment="片段不错"),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, content_completeness=4.0, instruction_fit=4.0, overall_comment="集锦不错"),
            ],
            overall_rhythm=4.0,
            overall_transition_quality=4.0,
            overall_audiovisual_sync=4.0,
            overall_completeness=4.0,
            overall_instruction_fit=4.0,
            overall_average=4.0,
            segment_content_completeness=4.0,
            segment_quality=4.0,
            segment_instruction_fit=4.0,
            segment_average=4.0,
            video_rhythm=4.0,
            video_transition_quality=4.0,
            video_audiovisual_sync=4.0,
            video_content_completeness=4.0,
            video_instruction_fit=4.0,
            video_average=4.0,
        )
        weighted = {"eval_score": 1.0, "judge_score": 0.4, "segment_judge_score": 0.4, "video_judge_score": 0.4, "weighted_score": 0.7, "degraded": False}

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=False))
        text = gen.generate(eval_report, judge_report, weighted)

        assert "加权总分" in text
        assert "0.7" in text
        assert "Segment Judge" in text
        assert "Video Judge" in text

    def test_generate_with_degraded_weighted(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])
        judge_report = JudgeReport(degraded=True, segment_degraded=True, video_degraded=True)
        weighted = {"eval_score": 1.0, "judge_score": 0.0, "segment_judge_score": 0.0, "video_judge_score": 0.0, "weighted_score": 1.0, "degraded": True}

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=False))
        text = gen.generate(eval_report, judge_report, weighted)

        assert "降级" in text
        assert "纯量化" in text

    def test_json_report_includes_weighted_score(self, tmp_path):
        evaluator = HighlightEvaluator()
        eval_report = evaluator.evaluate_all([
            {
                "case_id": "case_001",
                "category": "体育",
                "difficulty": "easy",
                "source_type": "local",
                "predicted": [{"start_time": 0.0, "end_time": 5.0}],
                "ground_truth": [{"start_time": 0.0, "end_time": 5.0}],
            },
        ])
        judge_report = JudgeReport(
            scores=[JudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, completeness=4.0, instruction_fit=4.0, overall_comment="不错")],
            segment_scores=[
                SegmentJudgeScore(content_completeness=4.0, segment_quality=4.0, instruction_fit=4.0, overall_comment="片段不错"),
            ],
            video_scores=[
                VideoJudgeScore(rhythm=4.0, transition_quality=4.0, audiovisual_sync=4.0, content_completeness=4.0, instruction_fit=4.0, overall_comment="集锦不错"),
            ],
            overall_rhythm=4.0,
            overall_transition_quality=4.0,
            overall_audiovisual_sync=4.0,
            overall_completeness=4.0,
            overall_instruction_fit=4.0,
            overall_average=4.0,
            segment_content_completeness=4.0,
            segment_quality=4.0,
            segment_instruction_fit=4.0,
            segment_average=4.0,
            video_rhythm=4.0,
            video_transition_quality=4.0,
            video_audiovisual_sync=4.0,
            video_content_completeness=4.0,
            video_instruction_fit=4.0,
            video_average=4.0,
        )
        weighted = {"eval_score": 1.0, "judge_score": 0.4, "segment_judge_score": 0.4, "video_judge_score": 0.4, "weighted_score": 0.7, "degraded": False}

        gen = ReportGenerator(ReportConfig(output_dir=str(tmp_path), save_charts=False))
        gen.generate(eval_report, judge_report, weighted)

        json_path = tmp_path / "report.json"
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "weighted_score" in data
        assert data["weighted_score"]["weighted_score"] == 0.7
        assert data["segment_judge"]["average"] == 4.0
        assert data["video_judge"]["average"] == 4.0
        assert data["llm_judge"]["degraded"] is False
