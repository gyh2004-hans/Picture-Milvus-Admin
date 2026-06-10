"""分区同步脚本 —— 确保数据库中所有分类分区存在.

以 config.CATEGORIES 为权威数据源，
扫描 Milvus 现有分区，创建缺失的分类分区。

用法:
  python scripts/sync_partitions.py           # 同步（创建缺失分区）
  python scripts/sync_partitions.py --dry-run # 仅预览，不创建
  python scripts/sync_partitions.py --list    # 仅列出当前状态
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 确保项目根在 sys.path
_project_dir = Path(__file__).resolve().parent.parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from src.config import CATEGORIES, CATEGORY_PARTITION_MAP
from src.milvus.vector_store import DEFAULT_PARTITION, VectorStore, get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("sync_partitions")


def print_status(store: VectorStore) -> None:
    """打印当前分区状态."""
    stats = store.get_stats_by_category()
    print()
    print("=" * 70)
    print("  当前分区状态")
    print("=" * 70)
    print(f"  {'分类':<12s} {'分区名':<12s} {'记录数':<8s} {'状态'}")
    print("-" * 70)

    existing = set()
    try:
        existing = set(store._backend.list_partitions()) if store._backend else set()
    except Exception:
        pass

    for cat in CATEGORIES:
        part_name = CATEGORY_PARTITION_MAP.get(cat, cat)
        count = stats.get(cat, 0)
        exists = "[OK]" if part_name in existing else "[MISSING]"
        print(f"  {cat:<12s} {part_name:<18s} {count:<8d} {exists}")

    default_count = stats.get("_default", 0)
    default_exists = "[OK]" if DEFAULT_PARTITION in existing else "[MISSING]"
    print(f"  {'_default':<12s} {DEFAULT_PARTITION:<18s} {default_count:<8d} {default_exists}")
    print("-" * 70)
    print(f"  {'合计':>26s} {stats.get('total', 0):<8d}")
    print("=" * 70)


def sync(store: VectorStore, dry_run: bool = False) -> dict:
    """同步分区 —— 创建 config.CATEGORIES 中缺失的分区.

    Args:
        store: 已连接的 VectorStore.
        dry_run: True 则仅打印不创建.

    Returns:
        {created: [partition_names], skipped_existing: [partition_names], errors: [...]}
    """
    if store._backend is None:
        store.connect()
    assert store._backend is not None

    result = {"created": [], "skipped_existing": [], "errors": []}

    existing_partitions = set()
    try:
        existing_partitions = set(store._backend.list_partitions())
        logger.info("现有分区: %s", sorted(existing_partitions))
    except Exception as exc:
        logger.error("无法列出分区: %s", exc)
        result["errors"].append(f"list_partitions: {exc}")
        return result

    for cat in CATEGORIES:
        part_name = CATEGORY_PARTITION_MAP.get(cat, cat)
        if part_name in existing_partitions:
            logger.info("  [OK] 已存在: %s (分类: %s)", part_name, cat)
            result["skipped_existing"].append(part_name)
        else:
            if dry_run:
                logger.info("  [DRY-RUN] 将创建: %s (分类: %s)", part_name, cat)
            else:
                try:
                    store._ensure_partition(part_name)
                    logger.info("  [NEW] 已创建: %s (分类: %s)", part_name, cat)
                    result["created"].append(part_name)
                except Exception as exc:
                    logger.error("  [FAIL] 创建失败: %s (分类: %s) error=%s", part_name, cat, exc)
                    result["errors"].append(f"{part_name}: {exc}")

    # 确保 _default 分区存在
    if DEFAULT_PARTITION not in existing_partitions:
        if dry_run:
            logger.info("  [DRY-RUN] 将创建: %s", DEFAULT_PARTITION)
        else:
            try:
                store._ensure_partition(DEFAULT_PARTITION)
                logger.info("  [NEW] 已创建: %s", DEFAULT_PARTITION)
                result["created"].append(DEFAULT_PARTITION)
            except Exception as exc:
                logger.error("  [FAIL] 创建失败: %s error=%s", DEFAULT_PARTITION, exc)
                result["errors"].append(f"{DEFAULT_PARTITION}: {exc}")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="同步 Milvus 分区 —— 以 config.CATEGORIES 为权威数据源",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="仅预览，不实际创建分区",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="仅列出当前分区状态并退出",
    )
    args = parser.parse_args()

    # 打印配置
    print(f"权威分类列表 (config.CATEGORIES): {len(CATEGORIES)} 个")
    for i, cat in enumerate(CATEGORIES, 1):
        part = CATEGORY_PARTITION_MAP.get(cat, cat)
        marker = " ← 分区名不同" if part != cat else ""
        print(f"  {i:>2}. {cat} → {part}{marker}")

    # 连接 Milvus
    store = get_vector_store()
    try:
        store.connect()
        logger.info("已连接 Milvus: backend=%s", store.backend_type)
    except Exception as exc:
        logger.error("连接 Milvus 失败: %s", exc)
        print("\n请确保 Milvus 已启动:")
        print("  docker compose up -d")
        sys.exit(1)

    if args.list:
        print_status(store)
        return

    # 同步
    result = sync(store, dry_run=args.dry_run)

    # ── 报告 ──
    print()
    print("=" * 70)
    print("  分区同步报告")
    print("=" * 70)
    print(f"  模式:          {'DRY-RUN' if args.dry_run else 'EXEC'}")
    print(f"  期望分区数:    {len(CATEGORIES) + 1} (含 _default)")
    print(f"  已存在（跳过）: {len(result['skipped_existing'])}")
    print(f"  新创建:        {len(result['created'])}")
    print(f"  失败:          {len(result['errors'])}")
    if result["created"]:
        print(f"  新分区:        {', '.join(result['created'])}")
    if result["errors"]:
        print(f"  错误:          {', '.join(result['errors'])}")
    print("=" * 70)

    # 同步后列出最终状态
    print_status(store)


if __name__ == "__main__":
    main()
