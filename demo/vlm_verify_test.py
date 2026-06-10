"""VLM 复核独立测试 —— 验证视觉模型对属性的二次判断.

对一张已有图片，用豆包视觉模型复核指定的属性列表。

用法（在仓库根目录）:
    # 手动指定属性列表
    python -m demo.vlm_verify_test --image path/to/img.png --missing "属性1,属性2,属性3"

    # 中文复核
    python -m demo.vlm_verify_test --image path/to/img.png --missing "红色标注,蓝色虚线" --lang zh
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

from src.config import VLM_VISION_MODEL
from src.evaluate.vlm_verifier import VLMVerifier
from src.models.schemas import AttributeScore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def test_manual_missing(
    image_path: str,
    missing_attrs: list[str],
    lang: str = "en",
) -> int:
    """手动指定 missing 属性，直接跑 VLM 复核（跳过 CLIP）."""
    _print_separator("VLM 复核独立测试 (手动指定 missing)")

    print(f"图片: {image_path}")
    print(f"手动指定 missing 属性 ({len(missing_attrs)}):")
    for i, attr in enumerate(missing_attrs, 1):
        print(f"  {i}. {attr}")

    # 构造假的 AttributeScore（score 填 0 占位）
    missing = [AttributeScore(attribute=attr, score=0.0) for attr in missing_attrs]

    _print_separator("VLM 复核中...")
    print(f"VLM 模型: {VLM_VISION_MODEL}")
    print(f"复核语言: {lang}")

    verifier = VLMVerifier()
    t0 = time.perf_counter()
    try:
        still_missing, promoted = asyncio.run(verifier.verify_missing(
            image_path=image_path,
            missing=missing,
            original_prompt="",  # 手动模式无原始 prompt
            lang=lang,
        ))
    except Exception as exc:
        logger.error("VLM verify failed: %s", exc)
        print(f"\n[FAIL] VLM 复核失败: {exc}")
        return 1
    vlm_elapsed = time.perf_counter() - t0

    # 结果
    _print_separator("复核结果")
    print(f"VLM 复核完成 ({vlm_elapsed:.1f}s)")

    print(f"\n  维持 missing ({len(still_missing)}):")
    if still_missing:
        for a in still_missing:
            print(f"       ❌ {a.attribute}")
    else:
        print(f"       (无)")

    print(f"\n  提升为 weak ({len(promoted)}):")
    if promoted:
        for a in promoted:
            print(f"       🔺 {a.attribute} → VLM 判为可见")
    else:
        print(f"       (无)")

    total = len(missing_attrs)
    promote_rate = len(promoted) / total * 100 if total else 0
    print(f"\n  {'─' * 50}")
    print(f"  汇总: {total} 个属性 → {len(promoted)} 个提升 ({promote_rate:.0f}%)")
    print(f"  {'─' * 50}")

    _print_separator("测试完成")
    return 0


# ═══════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="VLM 复核独立测试 — 验证豆包视觉模型对属性的二次判断",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 手动指定属性列表
  python -m picture.demo.vlm_verify_test --image path/to/img.png --missing "红色标注,蓝色虚线,绿色箭头"

  # 中文复核
  python -m picture.demo.vlm_verify_test --image path/to/img.png --missing "海洋,蒸发箭头" --lang zh
""",
    )

    parser.add_argument(
        "--image",
        required=True,
        help="待测试图片路径",
    )
    parser.add_argument(
        "--missing",
        required=True,
        help="手动指定属性列表，逗号分隔",
    )
    parser.add_argument(
        "--lang",
        default="en",
        choices=["zh", "en"],
        help="复核语言: en 英文 / zh 中文（默认 en）",
    )

    args = parser.parse_args(argv)

    # 检查图片存在
    if not Path(args.image).exists():
        print(f"错误: 图片不存在: {args.image}")
        sys.exit(1)

    missing_attrs = [a.strip() for a in args.missing.split(",") if a.strip()]
    if not missing_attrs:
        print("错误: --missing 参数为空")
        sys.exit(1)
    sys.exit(test_manual_missing(args.image, missing_attrs, lang=args.lang))


if __name__ == "__main__":
    main()
