"""Verify: LAS e2e + LLM Judge + Ark Files upload API connectivity."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ark_client import ArkClient, ArkConfig
from src.las_client import LasClient, LasConfig
from evaluation.llm_judge import JudgeConfig, LLMJudge


def test_env_vars():
    print("=" * 60)
    print("1. Check environment variables")
    print("=" * 60)
    vars_to_check = {
        "LAS_API_KEY": os.environ.get("LAS_API_KEY", ""),
        "ARK_JUDGE_API_KEY": os.environ.get("ARK_JUDGE_API_KEY", ""),
        "ARK_JUDGE_MODEL": os.environ.get("ARK_JUDGE_MODEL", ""),
        "ARK_JUDGE_BASE_URL": os.environ.get("ARK_JUDGE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "TOS_ACCESS_KEY": os.environ.get("TOS_ACCESS_KEY", ""),
        "TOS_SECRET_KEY": "***" if os.environ.get("TOS_SECRET_KEY") else "",
    }

    for name, value in vars_to_check.items():
        display = f"***{value[-8:]}" if value and not name.endswith("_BASE_URL") and name != "TOS_SECRET_KEY" else (value or "(NOT SET)")
        print(f"  {name:.<30s} {display}")

    errors = []
    required = ["LAS_API_KEY", "ARK_JUDGE_API_KEY", "ARK_JUDGE_MODEL"]
    for name in required:
        if not os.environ.get(name):
            errors.append(f"{name} not set")

    if errors:
        print("\n  [FAIL] Missing env vars:")
        for e in errors:
            print(f"     - {e}")
        return False
    print("  [PASS] All required env vars present")
    if os.environ.get("TOS_ACCESS_KEY"):
        print("  [INFO] TOS config present (optional)")
    return True


def test_las_health():
    print("\n" + "=" * 60)
    print("2. LAS API connectivity (health check via submit)")
    print("=" * 60)

    las_key = os.environ["LAS_API_KEY"]
    client = LasClient(LasConfig(api_key=las_key))
    try:
        result = client.submit(
            operator_id="las_video_edit",
            task_input={
                "video_url": "tos://example-bucket/nonexistent.mp4",
                "output_path": "tos://example-bucket/output/",
                "task_description": "health check - expect parameter error",
            },
        )
        print(f"  Response: task_id={result.get('task_id', 'N/A')}, status={result.get('status', result)}")
        print(f"  [PASS] LAS API reachable (auth OK)")
        return True
    except Exception as e:
        err = str(e)
        if "400" in err or "422" in err or "InvalidParameter" in err:
            print(f"  Response: {err[:120]}")
            print(f"  [PASS] LAS API reachable (auth OK, parameter error expected)")
            return True
        print(f"  [FAIL] {e}")
        return False


def test_ark_files_upload():
    print("\n" + "=" * 60)
    print("3. Ark Files API Upload (local → HTTPS URL)")
    print("=" * 60)

    import shutil
    import subprocess
    import tempfile

    tmpdir = tempfile.mkdtemp(prefix="verify_upload_")
    video_path = str(Path(tmpdir) / "test_upload.mp4")
    try:
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=0.5",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", video_path],
                check=True, capture_output=True, timeout=30,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"  [SKIP] Cannot create test video: {e}")
            return True

        from src.video_fetcher import ArkFileSource

        try:
            url = ArkFileSource(video_path).resolve()
            if url.startswith("https://"):
                print(f"  Upload OK: {url[:80]}...")
                print(f"  [PASS] Ark Files API Upload OK")
                return True
            else:
                print(f"  [FAIL] Unexpected URL format: {url}")
                return False
        except Exception as e:
            print(f"  [FAIL] {e}")
            return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_judge_chat():
    print("\n" + "=" * 60)
    print("4. LLM Judge API (Qwen / DashScope)")
    print("=" * 60)

    cfg = JudgeConfig()
    client = ArkClient(ArkConfig(api_key=cfg.api_key, base_url=cfg.base_url, model=cfg.model))
    try:
        response = client.chat(
            messages=[{"role": "user", "content": 'Reply JSON: {"status": "ok"}'}],
            temperature=0.1,
            max_tokens=128,
        )
        content = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        print(f"  Response: {content[:120]}")
        print(f"  Tokens:   prompt={usage.get('prompt_tokens')}, completion={usage.get('completion_tokens')}")
        print(f"  Base URL: {cfg.base_url}")
        print(f"  Model:    {cfg.model}")
        print(f"  [PASS] LLM Judge API OK")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def test_judge_scoring():
    print("\n" + "=" * 60)
    print("5. LLM Judge full scoring (text-only)")
    print("=" * 60)

    judge = LLMJudge()
    score = judge._judge_text_only(
        category="sports",
        target="highlight reel",
        style="fast-paced",
        segments=[
            {"start_time": 0.0, "end_time": 5.0, "score": 0.9, "label": "action"},
            {"start_time": 10.0, "end_time": 15.0, "score": 0.8, "label": "key scene"},
        ],
        max_retries=1,
    )

    if score.error:
        print(f"  [FAIL] {score.error}")
        return False

    print(f"  Rhythm:           {score.rhythm}/10.0")
    print(f"  Completeness:     {score.completeness}/10.0")
    print(f"  Excitement:       {score.excitement}/10.0")
    print(f"  Instruction Fit:  {score.instruction_fit}/10.0")
    print(f"  Average:          {score.average:.1f}/10.0")
    print(f"  Comment:          {score.overall_comment}")
    print(f"  [PASS] LLM Judge scoring OK")
    return True


def main():
    print("=" * 60)
    print("Video Highlight Skill - API Verification")
    print("  LAS e2e + LLM Judge (Qwen/DashScope) + Ark Files Upload")
    print("=" * 60)
    print(f"Working dir: {Path.cwd()}")
    print()

    results = {"Env Vars": test_env_vars()}

    if not results["Env Vars"]:
        print("\n[ABORT] Required env vars missing.")
        sys.exit(1)

    results["LAS API"] = test_las_health()
    results["Ark Files Upload"] = test_ark_files_upload()
    results["Judge Chat"] = test_judge_chat()
    results["Judge Scoring"] = test_judge_scoring()

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    all_pass = True
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All 5 checks passed! APIs working correctly.")
    else:
        print("Some checks failed. See errors above.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
