"""API 连通性检查 — Ark API + Judge API + Files 上传"""
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("connectivity")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    load_dotenv(_ENV_FILE)


def check_env():
    checks = {
        "ARK_HIGHLIGHT_API_KEY": os.environ.get("ARK_HIGHLIGHT_API_KEY", ""),
        "ARK_JUDGE_API_KEY": os.environ.get("ARK_JUDGE_API_KEY", ""),
        "ARK_JUDGE_MODEL": os.environ.get("ARK_JUDGE_MODEL", ""),
        "ARK_JUDGE_BASE_URL": os.environ.get("ARK_JUDGE_BASE_URL", ""),
    }
    logger.info("环境变量:")
    for k, v in checks.items():
        status = "OK" if v else "MISSING"
        logger.info("  %s: %s", k, status)
    return all(checks.values())


def test_ark_chat():
    from src.ark_client import ArkClient, ArkConfig

    logger.info("=" * 40)
    logger.info("测试 Ark Chat API...")
    try:
        client = ArkClient(ArkConfig(
            api_key=os.environ["ARK_HIGHLIGHT_API_KEY"],
            model=os.environ.get("ARK_HIGHLIGHT_MODEL", "doubao-seed-2-0-pro"),
        ))
        resp = client.chat(
            messages=[{"role": "user", "content": "回复 OK"}],
            temperature=0.0,
            max_tokens=10,
        )
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("Ark Chat 响应: %s", content[:50])
        return True
    except Exception as e:
        logger.error("Ark Chat 失败: %s", e)
        return False


def test_ark_files_upload():
    from src.video_fetcher import ArkFileSource

    logger.info("=" * 40)
    logger.info("测试 Ark Files 上传...")
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42")
            tmp_path = f.name

        source = ArkFileSource(tmp_path)
        url = source.resolve()
        logger.info("上传成功: %s", url[:80])
        os.unlink(tmp_path)
        return True
    except Exception as e:
        logger.error("Ark Files 上传失败: %s", e)
        return False


def test_judge_chat():
    from evaluation.llm_judge import LLMJudge, JudgeConfig

    logger.info("=" * 40)
    logger.info("测试 Judge Chat API...")
    try:
        judge = LLMJudge(config=JudgeConfig())
        resp = judge.ark_client.chat(
            messages=[{"role": "user", "content": '输出 {"test": true}'}],
            temperature=0.0,
            max_tokens=50,
        )
        parsed = judge.ark_client.extract_json(resp)
        logger.info("Judge Chat 响应: %s", json.dumps(parsed, ensure_ascii=False)[:100])
        return True
    except Exception as e:
        logger.error("Judge Chat 失败: %s", e)
        return False


def main():
    logger.info("API 连通性检查")
    logger.info("=" * 60)

    if not check_env():
        logger.warning("部分环境变量缺失，后续测试可能失败")

    results = {
        "ark_chat": test_ark_chat(),
        "ark_files_upload": test_ark_files_upload(),
        "judge_chat": test_judge_chat(),
    }

    logger.info("=" * 60)
    logger.info("结果汇总:")
    for name, ok in results.items():
        logger.info("  %s: %s", name, "PASS" if ok else "FAIL")

    if all(results.values()):
        logger.info("全部通过！")
    else:
        logger.warning("部分检查失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
