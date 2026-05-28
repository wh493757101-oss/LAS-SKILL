"""Ark API token-based cost estimation."""

ARK_PRICING: dict[str, dict[str, float]] = {
    "doubao-seed-2-0-pro": {"input": 0.015, "output": 0.06},
    "doubao-vision-pro-32k": {"input": 0.015, "output": 0.06},
    "default": {"input": 0.015, "output": 0.06},
}


def estimate_ark_cost(
    video_duration_seconds: float,
    model: str = "default",
    estimated_prompt_tokens: int | None = None,
    estimated_completion_tokens: int | None = None,
) -> float:
    """Estimate Ark API cost based on video duration and model pricing.

    If token counts are not provided, estimate based on video duration:
    - ~2000 prompt tokens per request (fixed overhead)
    - ~100 tokens per second of video (video encoding overhead)
    - ~500 completion tokens (typical JSON response)

    Returns cost in RMB (yuan).
    """
    pricing = ARK_PRICING.get(model, ARK_PRICING["default"])

    if estimated_prompt_tokens is None:
        estimated_prompt_tokens = 2000 + int(video_duration_seconds * 100)
    if estimated_completion_tokens is None:
        estimated_completion_tokens = 500

    input_cost = (estimated_prompt_tokens / 1000) * pricing["input"]
    output_cost = (estimated_completion_tokens / 1000) * pricing["output"]

    return round(input_cost + output_cost, 4)
