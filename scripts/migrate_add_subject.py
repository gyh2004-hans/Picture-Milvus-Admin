#!/usr/bin/env python3
"""存量数据迁移脚本 —— 为无 subject 的存量记录标注学科.

迁移策略:
  1. 扫描 Milvus collection 中 subject 为空或 null 的记录
  2. 尝试用 LLM 从 prompt 自动推断学科
  3. 无法推断的标记为 _default 分区（保持不变）
  4. 提供 dry-run 模式预览变更

用法:
  python scripts/migrate_add_subject.py                    # dry-run 预览
  python scripts/migrate_add_subject.py --execute          # 执行迁移
  python scripts/migrate_add_subject.py --subject math     # 批量标记为数学
  python scripts/migrate_add_subject.py --execute --batch-size 50  # 分批执行
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

# 确保 picture2 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milvus.vector_store import DEFAULT_PARTITION, SUBJECT_PARTITION_MAP, VectorStore, get_vector_store

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("migrate_add_subject")

# 学科关键词映射（用于正则推断）
SUBJECT_KEYWORDS: dict[str, list[str]] = {
    "chinese": ["语文", "作文", "阅读", "文言文", "诗歌", "古诗", "汉字", "拼音", "课文", "写作", "阅读理解"],
    "math": ["数学", "几何", "代数", "函数", "方程", "三角", "圆", "概率", "统计", "数列", "导数", "积分", "向量", "坐标"],
    "english": ["英语", "英文", "单词", "语法", "时态", "阅读理解", "完形填空", "听力", "口语", "作文", "翻译"],
    "physics": ["物理", "力学", "电学", "光学", "磁场", "电场", "牛顿", "速度", "加速度", "能量", "动量", "电路", "电磁"],
    "chemistry": ["化学", "元素", "反应", "分子", "原子", "方程式", "酸碱", "氧化", "还原", "有机", "无机", "实验"],
    "biology": ["生物", "细胞", "基因", "DNA", "蛋白质", "光合", "呼吸", "遗传", "进化", "生态", "植物", "动物", "人体"],
    "history": ["历史", "朝代", "战争", "革命", "帝国", "皇帝", "古代", "近代", "文明", "改革", "条约", "起义"],
    "geography": ["地理", "地图", "气候", "地形", "山脉", "河流", "海洋", "板块", "经纬", "气压", "降水", "火山", "地震", "人文"],
    "politics": ["政治", "国家", "政府", "法律", "宪法", "民主", "权利", "义务", "制度", "政策", "公民", "选举", "社会"],
}


def infer_subject_from_prompt(prompt: str) -> str | None:
    """从 prompt 文本推断学科.

    策略: 关键词正则匹配，匹配最多关键词的学科胜出.
    """
    if not prompt:
        return None

    scores: dict[str, int] = {}
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # 使用正则匹配（忽略大小写）
            count = len(re.findall(re.escape(kw), prompt, re.IGNORECASE))
            score += count
        if score > 0:
            scores[subject] = score

    if not scores:
        return None

    # 匹配最多关键词的学科胜出
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best]
    # 置信度：需要至少 2 个关键词，或总分 ≥ 2
    if best_score < 2:
        return None
    return best


async def migrate(
    execute: bool = False,
    force_subject: str | None = None,
    batch_size: int = 50,
    dry_run: bool = True,
) -> dict:
    """执行存量数据迁移.

    Args:
        execute: False 则仅预览不写入.
        force_subject: 强制设置所有无 subject 记录为该学科.
        batch_size: 分批大小.
        dry_run: True 时仅打印变更不写入.

    Returns:
        {total_scanned, migrated, failed, by_subject: {subject: count}}
    """
    store = get_vector_store()
    store.connect()
    backend = store._backend
    if backend is None:
        logger.error("Migration: failed to connect to Milvus")
        return {"total_scanned": 0, "migrated": 0, "failed": 0, "by_subject": {}}

    backend_label = store.backend_type
    logger.info(
        "Migration: connected to backend=%s total=%d",
        backend_label, store.count(),
    )

    by_subject: dict[str, int] = {}
    result = {
        "total_scanned": store.count(),
        "migrated": 0,
        "failed": 0,
        "by_subject": by_subject,
    }

    if store.count() == 0:
        logger.info("Migration: collection is empty, nothing to do")
        return result

    # Milvus Lite / Server: query entities with empty subject
    if backend_label != "local_numpy":
        try:
            # Query all entities to check subject field
            from pymilvus import MilvusClient
            client = backend._client  # type: ignore[union-attr]
            if client is None:
                logger.error("Migration: MilvusClient is None")
                return result

            # Query entities where subject is empty or missing
            # For dynamic fields, we query all and filter
            all_entities = client.query(
                collection_name="image_embeddings",
                filter="id >= 0",
                output_fields=["id", "prompt", "subject"],
                limit=batch_size * 10,
            )
            logger.info("Migration: queried %d entities", len(all_entities))

            for entity in all_entities:
                entity_id = entity.get("id")
                existing_subject = entity.get("subject", "")
                prompt = entity.get("prompt", "")

                if existing_subject and existing_subject.strip():
                    continue  # 已有学科标注，跳过

                # 推断学科
                if force_subject:
                    new_subject = force_subject
                else:
                    new_subject = infer_subject_from_prompt(prompt)

                if new_subject is None:
                    logger.debug("Migration: id=%d cannot infer subject, keeping _default", entity_id)
                    continue

                partition = SUBJECT_PARTITION_MAP.get(new_subject, DEFAULT_PARTITION)
                logger.info(
                    "Migration: %s id=%d subject='%s' → '%s' (partition=%s) prompt='%s'",
                    "DRY-RUN" if dry_run else "EXEC",
                    entity_id,
                    existing_subject or "(empty)",
                    new_subject,
                    partition,
                    prompt[:60],
                )

                if execute:
                    try:
                        ok = backend.update_metadata(entity_id, {"subject": new_subject})
                        if ok:
                            result["migrated"] += 1
                            result["by_subject"][new_subject] = \
                                result["by_subject"].get(new_subject, 0) + 1
                        else:
                            result["failed"] += 1
                    except Exception as exc:
                        logger.error("Migration: id=%d update failed: %s", entity_id, exc)
                        result["failed"] += 1
                else:
                    result["migrated"] += 1
                    result["by_subject"][new_subject] = \
                        result["by_subject"].get(new_subject, 0) + 1

        except Exception as exc:
            logger.error("Migration: query failed: %s", exc)
            return result
    else:
        logger.warning(
            "Migration: local_numpy backend detected. "
            "Numpy backend records don't have persistent subject field. "
            "Use --force-subject to set subject for all in-memory records."
        )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="存量数据迁移: 为无 subject 的 Milvus 记录标注学科",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="执行迁移（默认 dry-run 仅预览）",
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="强制设置所有无 subject 记录为该学科",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="分批大小（默认 50）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="仅预览，不写入（默认）",
    )
    args = parser.parse_args()

    # 验证 subject 参数
    if args.subject and args.subject not in SUBJECT_PARTITION_MAP:
        logger.error(
            "Invalid subject: '%s'. Valid: %s",
            args.subject, list(SUBJECT_PARTITION_MAP.keys()),
        )
        sys.exit(1)

    execute = args.execute
    if execute:
        args.dry_run = False
        logger.warning("⚠️  EXECUTE mode: changes will be written to Milvus!")
        confirm = input("Continue? [y/N]: ")
        if confirm.strip().lower() != "y":
            logger.info("Migration cancelled")
            return

    result = asyncio.run(migrate(
        execute=execute,
        force_subject=args.subject,
        batch_size=args.batch_size,
        dry_run=not execute,
    ))

    # ── 报告 ──
    print()
    print("=" * 60)
    print("  Migration Report")
    print("=" * 60)
    print(f"  Mode:          {'EXECUTE' if execute else 'DRY-RUN'}")
    print(f"  Force subject: {args.subject or 'auto-infer'}")
    print(f"  Total scanned: {result['total_scanned']}")
    print(f"  Migrated:      {result['migrated']}")
    print(f"  Failed:        {result['failed']}")
    if result["by_subject"]:
        print("  By subject:")
        for subj, cnt in sorted(result["by_subject"].items()):
            print(f"    {subj}: {cnt}")
    print("=" * 60)


if __name__ == "__main__":
    main()
