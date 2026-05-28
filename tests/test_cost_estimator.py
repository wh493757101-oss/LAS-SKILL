import pytest

from src.cost_estimator import estimate_ark_cost


class TestEstimateArkCost:
    def test_default_pricing(self):
        cost = estimate_ark_cost(60.0)
        # 60s video: ~2000 + 6000 = 8000 prompt tokens, 500 completion tokens
        # input: 8000/1000 * 0.015 = 0.12, output: 500/1000 * 0.06 = 0.03
        expected = round(0.12 + 0.03, 4)
        assert cost == expected

    def test_short_video(self):
        cost = estimate_ark_cost(10.0)
        # 10s video: ~2000 + 1000 = 3000 prompt tokens, 500 completion tokens
        # input: 3000/1000 * 0.015 = 0.045, output: 0.03
        expected = round(0.045 + 0.03, 4)
        assert cost == expected

    def test_long_video(self):
        cost = estimate_ark_cost(300.0)
        # 300s video: ~2000 + 30000 = 32000 prompt tokens, 500 completion tokens
        # input: 32000/1000 * 0.015 = 0.48, output: 0.03
        expected = round(0.48 + 0.03, 4)
        assert cost == expected

    def test_explicit_tokens(self):
        cost = estimate_ark_cost(
            60.0,
            estimated_prompt_tokens=10000,
            estimated_completion_tokens=1000,
        )
        expected = round((10000 / 1000) * 0.015 + (1000 / 1000) * 0.06, 4)
        assert cost == expected

    def test_custom_model(self):
        cost = estimate_ark_cost(60.0, model="doubao-seed-2-0-pro")
        expected = round(0.12 + 0.03, 4)
        assert cost == expected
