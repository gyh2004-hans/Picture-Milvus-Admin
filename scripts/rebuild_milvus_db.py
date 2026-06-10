"""一次性脚本: 从 upstream_*_summary.json 重建 Milvus 数据库.

milvus_lite 3.0 存在中文分区名的 Unicode 路径 bug（faiss 索引构建失败）。
此脚本使用拼音分区名重建数据库，解决 load_collection 失败问题。

运行: 在 picture2/ 目录内执行 python scripts/rebuild_milvus_db.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# 确保 picture2/ 在 sys.path
_picture2_dir = Path(__file__).resolve().parent.parent
if str(_picture2_dir) not in sys.path:
    sys.path.insert(0, str(_picture2_dir))

from src.config import EMBEDDING_DIM, STORAGE_DIR
from src.milvus.vector_store import (
    get_vector_store,
    DEFAULT_PARTITION,
    SUBJECT_PARTITION_MAP,
    VectorStore,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rebuild_milvus_db")


def _subject_from_filename(filename: str) -> Optional[str]:
    """从 upstream_{subject}_*_summary.json 提取学科英文名.

    Examples:
        upstream_math_derivative_meaning_summary.json → "math"
        upstream_en_tenses_timeline_summary.json → "en" (→ "english")
        upstream_cn_poetry_scene_summary.json → "cn" (→ "chinese")
        upstream_geo_atmosphere_circulation_summary.json → "geo" (→ "geography")
        upstream_hist_renaissance_summary.json → "hist" (→ "history")
        upstream_chem_electrolysis_summary.json → "chem" (→ "chemistry")
        upstream_physics_newton_laws_summary.json → "physics"
        upstream_pol_dialectical_materialism_summary.json → "pol" (→ "politics")
    """
    stem = Path(filename).stem  # upstream_math_derivative_meaning_summary
    parts = stem.split("_")
    if len(parts) < 2 or parts[0] != "upstream":
        return None
    subj_abbr = parts[1].lower()
    # 缩写 → 标准英文名
    _abbr_map = {
        "cn": "chinese", "math": "math", "en": "english",
        "physics": "physics", "chem": "chemistry", "bio": "biology",
        "hist": "history", "geo": "geography", "pol": "politics",
        # 全名直接通过
        "chinese": "chinese", "english": "english", "biology": "biology",
        "history": "history", "geography": "geography", "politics": "politics",
        "chemistry": "chemistry",
    }
    return _abbr_map.get(subj_abbr)


def main() -> None:
    """主流程: 扫描 → 解析 → 入库."""
    storage = Path(STORAGE_DIR)
    if not storage.exists():
        logger.error("STORAGE_DIR not found: %s", storage)
        sys.exit(1)

    # 1. 扫描所有 upstream_*_summary.json
    summary_files = sorted(storage.glob("upstream_*_summary.json"))
    logger.info("Found %d summary files", len(summary_files))
    if not summary_files:
        logger.error("No summary files found in %s", storage)
        sys.exit(1)

    # 2. 删除旧数据库（如果存在）
    from src.milvus.vector_store import MILVUS_LITE_DB_PATH

    old_db = Path(MILVUS_LITE_DB_PATH)
    if old_db.exists():
        import shutil
        shutil.rmtree(old_db)
        logger.info("Deleted old milvus_lite.db")

    # 3. 初始化 VectorStore（创建新 DB + pinyin 分区）
    store = get_vector_store()
    store.connect()
    logger.info("VectorStore connected | backend=%s", store.backend_type)

    # 4. 逐文件解析并入库
    imported = 0
    skipped = 0
    zero_vec = [0.0] * EMBEDDING_DIM

    for sf in summary_files:
        subject = _subject_from_filename(sf.name)
        if subject is None:
            logger.warning("Cannot determine subject from filename: %s", sf.name)
            skipped += 1
            continue

        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON: %s error=%s", sf.name, exc)
            skipped += 1
            continue

        # 提取最终生成的图片信息
        final_image_path = data.get("final_image_path", "")
        final_prompt = data.get("final_prompt", "")
        final_score = data.get("final_score", 0.0)

        # 从 history 中取最后一轮的 optimized_prompt
        history = data.get("history", [])
        optimized_prompt = None
        if history:
            last_iter = history[-1]
            optimized_prompt = last_iter.get("optimized_prompt")

        if not final_image_path:
            logger.warning("No final_image_path in %s, skipping", sf.name)
            skipped += 1
            continue

        # 验证图片文件存在
        img_path = Path(final_image_path)
        if not img_path.exists():
            logger.warning("Image not found: %s (from %s)", final_image_path, sf.name)
            skipped += 1
            continue

        # 构造 ImageRecord 并入库
        from src.models.schemas import ImageRecord

        record = ImageRecord(
            image_id=0,  # auto_id
            prompt=final_prompt,
            optimized_prompt=optimized_prompt,
            score=final_score,
            image_path=str(img_path),
            embedding=zero_vec.copy(),
            subject=subject,
            category="",
            tags=[],
        )

        try:
            img_id = store.insert(record, subject=subject)
            imported += 1
            if imported % 10 == 0:
                logger.info("Progress: %d/%d imported", imported, len(summary_files))
        except Exception as exc:
            logger.error("Insert failed for %s: %s", sf.name, exc)
            skipped += 1

    logger.info(
        "DONE: imported=%d skipped=%d total_summaries=%d "
        "db_records=%d backend=%s",
        imported, skipped, len(summary_files),
        store.count(), store.backend_type,
    )


if __name__ == "__main__":
    main()
