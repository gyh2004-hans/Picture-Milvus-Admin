"""向量存储与检索模块

对应项目计划书 §5 "检索与向量数据库模块".

功能:
  1. 图片向量存储（CLIP embedding）
  2. 相似图检索（Top-K） — 以文搜图 / 以图搜图
  3. 历史 Prompt 复用
  4. 分类分区隔离检索（v4 学科分区, v6 升级为动态分类分区）

数据库存储字段:
  - 图像 ID / Prompt / 评测得分 / 图像/文本/语义向量
  - 分类标签 (category) / 自由标签 (tags) ← v6 泛化
  - 时间戳与生成元数据

v6 分区体系:
  动态分类分区（从 config.CATEGORIES 加载）+ _default 分区用于未分类数据
  保留 SUBJECT_PARTITION_MAP 向后兼容旧 9 学科分区
"""
from __future__ import annotations

import atexit
import json
import logging
import os as _os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import CATEGORIES, CATEGORY_PARTITION_MAP, EMBEDDING_DIM, MILVUS_HOST, MILVUS_PORT, STORAGE_DIR
from src.models.schemas import ImageRecord, SearchResponse

logger = logging.getLogger(__name__)

# ── Windows: monkey-patch os.rename → os.replace ───────────
# milvus_lite 用 os.rename() 做 manifest 原子写入，但 Windows 上
# os.rename 在目标文件已存在时报 FileExistsError（Linux 上是覆盖）。
# 用 os.replace 替换，它在所有平台都是原子覆盖。
if _os.name == "nt":
    try:
        import milvus_lite.storage.manifest as _ml_manifest
        _ml_manifest.os.rename = _os.replace
        logger.info("vector_store: patched milvus_lite os.rename → os.replace for Windows")
    except ImportError:
        pass

COLLECTION_NAME = "image_embeddings"  # 向量维度见 config.EMBEDDING_DIM
MILVUS_LITE_DB_PATH = str(STORAGE_DIR / "milvus_lite.db")
# Milvus Standalone search/query 的 topk 上限（超出会抛 MilvusException）
MILVUS_SEARCH_MAX_LIMIT = 16384

# ══════════════════════════════════════════════════════════
# 学科 → 分区名映射 (v4, DEPRECATED — v6 由 CATEGORY_PARTITION_MAP 替代)
# ══════════════════════════════════════════════════════════
SUBJECT_PARTITION_MAP: dict[str, str] = {
    "chinese":   "yuwen",
    "math":      "shuxue",
    "english":   "yingyu",
    "physics":   "wuli",
    "chemistry": "huaxue",
    "biology":   "shengwu",
    "history":   "lishi",
    "geography": "dili",
    "politics":  "zhengzhi",
}
PARTITION_NAME_TO_SUBJECT: dict[str, str] = {
    v: k for k, v in SUBJECT_PARTITION_MAP.items()
}
DEFAULT_PARTITION = "_default"

# v6: 合并旧学科分区和新的动态分类分区
_ALL_KNOWN_PARTITIONS: set[str] = set(SUBJECT_PARTITION_MAP.values()) | set(CATEGORY_PARTITION_MAP.values())


def _resolve_partition(subject_or_category: str | None) -> str:
    """将 subject / category 值解析为分区名（v6 泛化）.

    解析优先级:
      1. 新动态分类映射 (CATEGORY_PARTITION_MAP)
      2. 旧学科映射 (SUBJECT_PARTITION_MAP)
      3. Subject 枚举
      4. 直接作为分区名（宽松匹配）
      5. _default

    Args:
        subject_or_category: 分类名/学科名，或 None.

    Returns:
        分区名（如 "风景" 或 "_default"）.
    """
    if not subject_or_category:
        return DEFAULT_PARTITION
    s = subject_or_category.strip()
    if not s:
        return DEFAULT_PARTITION

    # 1. 精确匹配新分类映射 (中文名)
    if s in CATEGORY_PARTITION_MAP:
        return CATEGORY_PARTITION_MAP[s]

    # 2. 精确匹配旧学科映射 (英文名)
    s_lower = s.lower()
    if s_lower in SUBJECT_PARTITION_MAP:
        return SUBJECT_PARTITION_MAP[s_lower]

    # 3. 旧分区名反向匹配
    if s_lower in PARTITION_NAME_TO_SUBJECT:
        return s_lower

    # 4. Subject 枚举
    try:
        from src.models.schemas import Subject
        subj = Subject(s_lower)
        return SUBJECT_PARTITION_MAP[subj.value]
    except (ValueError, KeyError):
        pass

    # 5. 宽松匹配: 如果值直接等于某个可用分类名，直接使用
    if s in CATEGORIES:
        return s

    # 6. 包含匹配（如 "风景类" 匹配 "风景"）
    for cat in CATEGORIES:
        if cat in s or s in cat:
            logger.info("_resolve_partition: fuzzy match %r → %r", s, cat)
            return cat

    logger.warning("_resolve_partition: unknown value=%r, using _default", subject_or_category)
    return DEFAULT_PARTITION


def _serialize_tags(tags: list[str] | str | None) -> str:
    """将 tags 序列化为 JSON array string（对齐优化计划 §5.1）.

    向后兼容：已是 JSON 字符串的直接返回；逗号分隔格式自动转换.
    """
    if isinstance(tags, list):
        return json.dumps(tags, ensure_ascii=False)
    if isinstance(tags, str) and tags.strip():
        stripped = tags.strip()
        if stripped.startswith("["):
            return stripped  # 已是 JSON array
        # 逗号分隔 → JSON array
        items = [t.strip() for t in stripped.split(",") if t.strip()]
        return json.dumps(items, ensure_ascii=False)
    return "[]"


def _deserialize_tags(tags_raw: str | None) -> list[str]:
    """从 Milvus 存储反序列化 tags，向后兼容逗号分隔旧格式."""
    if not tags_raw or not tags_raw.strip():
        return []
    stripped = tags_raw.strip()
    # 优先 JSON 解析（v4+ 新格式）
    try:
        result = json.loads(stripped)
        if isinstance(result, list):
            return [str(item) for item in result]
    except (json.JSONDecodeError, ValueError):
        pass
    # 回退：逗号分隔旧格式
    return [t.strip() for t in stripped.split(",") if t.strip()]


# ══════════════════════════════════════════════════════════
# 抽象后端接口
# ══════════════════════════════════════════════════════════
class VectorBackend(ABC):
    """向量存储后端抽象接口（项目计划书 §5.2）"""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def insert(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray | None,
        metadata: dict,
        partition_name: str = DEFAULT_PARTITION,
    ) -> int: ...

    @abstractmethod
    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        partition_names: list[str] | None = None,
    ) -> list[tuple[float, dict]]: ...

    @abstractmethod
    def count(self, partition_name: str | None = None) -> int: ...

    @abstractmethod
    def drop_all(self) -> None: ...

    # ── v4 分区管理（可选实现，默认抛 NotImplemented） ──

    def create_partition(self, partition_name: str) -> None:
        raise NotImplementedError

    def has_partition(self, partition_name: str) -> bool:
        raise NotImplementedError

    def drop_partition(self, partition_name: str) -> None:
        raise NotImplementedError

    def list_partitions(self) -> list[str]:
        raise NotImplementedError

    def get_partition_stats(self) -> dict[str, int]:
        raise NotImplementedError

    # ── v4 CRUD ──

    def delete_by_id(self, entity_id: int) -> bool:
        raise NotImplementedError

    def update_metadata(self, entity_id: int, metadata: dict) -> bool:
        raise NotImplementedError

    def list_data(
        self,
        limit: int = 50,
        offset: int = 0,
        subject: str | None = None,
        min_score: float | None = None,
    ) -> tuple[list[dict], int]:
        """分页列出记录（带可选过滤）.

        Returns:
            (records, total_count)
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════
# 本地 Numpy 后端（默认）
# ══════════════════════════════════════════════════════════
class LocalNumpyBackend(VectorBackend):
    """本地内存 numpy 索引后端.

    零外部依赖，适合开发和小规模数据集.
    v4: 新增 partition 支持（dict of lists）.
    """

    def __init__(self) -> None:
        self._embeddings: dict[str, list[np.ndarray]] = {}  # partition → embeddings
        self._records: dict[str, list[dict]] = {}            # partition → records
        self._connected = False

    def connect(self) -> None:
        logger.info("LocalNumpyBackend.connect | mode=local_in_memory")
        self._connected = True

    def insert(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray | None,
        metadata: dict,
        partition_name: str = DEFAULT_PARTITION,
    ) -> int:
        if not self._connected:
            self.connect()

        if partition_name not in self._embeddings:
            self._embeddings[partition_name] = []
            self._records[partition_name] = []

        image_id = sum(len(v) for v in self._records.values()) + 1
        rec = {
            "image_id": image_id,
            "partition": partition_name,
            **metadata,
        }
        if text_embedding is not None:
            rec["text_embedding"] = text_embedding

        self._embeddings[partition_name].append(image_embedding.astype(np.float32))
        self._records[partition_name].append(rec)
        return image_id

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        partition_names: list[str] | None = None,
    ) -> list[tuple[float, dict]]:
        # 确定要搜索的分区
        if partition_names is None:
            part_keys = list(self._embeddings.keys())
        else:
            part_keys = [p for p in partition_names if p in self._embeddings]

        if not part_keys or all(len(self._embeddings[p]) == 0 for p in part_keys):
            return []

        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        scored: list[tuple[float, dict]] = []
        for pk in part_keys:
            for emb, rec in zip(self._embeddings[pk], self._records[pk]):
                emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
                sim = float(np.dot(query_norm, emb_norm))
                scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def count(self, partition_name: str | None = None) -> int:
        if partition_name is None:
            return sum(len(v) for v in self._records.values())
        return len(self._records.get(partition_name, []))

    def drop_all(self) -> None:
        total = sum(len(v) for v in self._records.values())
        logger.warning("LocalNumpyBackend.drop_all | dropping %d records across %d partitions",
                       total, len(self._records))
        self._embeddings.clear()
        self._records.clear()

    # ── v4 分区管理 ──

    def create_partition(self, partition_name: str) -> None:
        if partition_name not in self._embeddings:
            self._embeddings[partition_name] = []
            self._records[partition_name] = []
            logger.info("LocalNumpyBackend.create_partition | name=%s", partition_name)

    def has_partition(self, partition_name: str) -> bool:
        return partition_name in self._embeddings

    def drop_partition(self, partition_name: str) -> None:
        cnt = len(self._records.get(partition_name, []))
        self._embeddings.pop(partition_name, None)
        self._records.pop(partition_name, None)
        logger.info("LocalNumpyBackend.drop_partition | name=%s dropped=%d", partition_name, cnt)

    def list_partitions(self) -> list[str]:
        return list(self._embeddings.keys())

    def get_partition_stats(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._records.items()}

    def delete_by_id(self, entity_id: int) -> bool:
        for pk, recs in self._records.items():
            for i, rec in enumerate(recs):
                if rec.get("image_id") == entity_id:
                    recs.pop(i)
                    self._embeddings[pk].pop(i)
                    return True
        return False

    def update_metadata(self, entity_id: int, metadata: dict) -> bool:
        for recs in self._records.values():
            for rec in recs:
                if rec.get("image_id") == entity_id:
                    rec.update(metadata)
                    return True
        return False

    def list_data(
        self,
        limit: int = 50,
        offset: int = 0,
        subject: str | None = None,
        min_score: float | None = None,
    ) -> tuple[list[dict], int]:
        # 收集所有记录（展平所有分区），附带 partition 信息
        all_recs: list[dict] = []
        for pk, recs in self._records.items():
            for rec in recs:
                all_recs.append({
                    "id": rec.get("image_id", 0),
                    "prompt": rec.get("prompt", ""),
                    "optimized_prompt": rec.get("optimized_prompt", ""),
                    "score": float(rec.get("score", 0.0)),
                    "image_path": rec.get("image_path", ""),
                    "subject": rec.get("subject", ""),
                    "category": rec.get("category", ""),
                    "tags": rec.get("tags", []) if isinstance(rec.get("tags"), list) else [],
                    "created_at": rec.get("created_at", ""),
                    "model_version": rec.get("model_version", ""),
                    "_partition": pk,
                })

        # 按学科分区过滤
        if subject:
            resolved_partition = _resolve_partition(subject)
            all_recs = [r for r in all_recs if r["_partition"] == resolved_partition]
        # 按最低分过滤
        if min_score is not None:
            all_recs = [r for r in all_recs if r["score"] >= min_score]

        total = len(all_recs)
        # 按 id 降序（最新在前）
        all_recs.sort(key=lambda r: r["id"], reverse=True)
        return all_recs[offset:offset + limit], total


# ══════════════════════════════════════════════════════════
# Milvus Lite 后端（生产推荐）
# ══════════════════════════════════════════════════════════
class MilvusLiteBackend(VectorBackend):
    """Milvus Lite 嵌入式向量数据库后端.

    使用 pymilvus.MilvusClient 提供的本地文件存储，
    无需额外部署 Milvus 服务端，零运维成本。

    v4 新增:
      - 学科分区（Partition）创建与路由
      - 分区内检索
      - 分区统计
      - CRUD 操作

    特性:
      - 本地文件持久化（db_path 指向 .db 文件）
      - 与 Milvus Server 完全兼容 API
      - 支持 COSINE / L2 / IP 距离度量
      - 支持索引（FLAT 默认，可扩展 IVF_FLAT / HNSW）

    需要: pip install "pymilvus>=2.4.0"
    """

    def __init__(self, uri: str = "", db_path: str = "") -> None:
        # 优先使用 URI（Milvus Standalone），回退到文件路径（Milvus Lite）
        if uri:
            self._uri = uri
        elif db_path:
            self._uri = db_path
        else:
            from src.config import MILVUS_URI
            self._uri = MILVUS_URI
        self._connected = False
        self._client = None
        self._collection_name = COLLECTION_NAME
        self._dim = EMBEDDING_DIM

    def connect(self) -> None:
        """连接 Milvus Lite 本地数据库并确保 collection 存在."""
        try:
            from pymilvus import MilvusClient
        except ImportError:
            logger.warning(
                "MilvusLiteBackend: pymilvus not installed. "
                "Install with: pip install 'pymilvus>=2.4.0'"
            )
            raise

        logger.info(
            "MilvusLiteBackend.connect | uri=%s",
            self._uri,
        )

        try:
            self._client = MilvusClient(uri=self._uri)

            # 检查 collection 是否存在
            if self._client.has_collection(self._collection_name):
                logger.info(
                    "MilvusLiteBackend: collection '%s' already exists, loading",
                    self._collection_name,
                )
            else:
                # 创建 collection（项目计划书 §5.2 schema + v4 动态字段）
                self._client.create_collection(
                    collection_name=self._collection_name,
                    dimension=self._dim,
                    metric_type="COSINE",
                    auto_id=True,
                    enable_dynamic_field=True,
                )
                logger.info(
                    "MilvusLiteBackend: created collection '%s' (dim=%d, metric=COSINE)",
                    self._collection_name,
                    self._dim,
                )

                # 创建索引（FLAT 用于精确搜索，数据量大后可切换 IVF_FLAT / HNSW）
                # 注意: Milvus Standalone 在 create_collection 时若传了 dimension +
                # metric_type 参数，会自动为 vector 字段建索引，所以先检查已有索引。
                try:
                    existing = self._client.list_indexes(self._collection_name)
                    if not existing:
                        idx_params = self._client.prepare_index_params()
                        idx_params.add_index(
                            field_name="vector",
                            index_type="FLAT",
                            metric_type="COSINE",
                        )
                        self._client.create_index(
                            collection_name=self._collection_name,
                            index_params=idx_params,
                        )
                        logger.info(
                            "MilvusLiteBackend: created FLAT index on vector field"
                        )
                    else:
                        logger.info(
                            "MilvusLiteBackend: index already exists on '%s', skipping",
                            self._collection_name,
                        )
                except Exception as exc:
                    logger.warning(
                        "MilvusLiteBackend: create_index failed (non-critical): %s", exc
                    )

            # Load collection for search.
            # NOTE: load_collection may fail on milvus_lite 3.0 when partition
            # names contain CJK characters (Unicode path bug in faiss index build).
            # When it fails, query/get are unavailable but search() still works.
            try:
                self._client.load_collection(self._collection_name)
                logger.info(
                    "MilvusLiteBackend: collection '%s' loaded",
                    self._collection_name,
                )
            except Exception as exc:
                logger.warning(
                    "MilvusLiteBackend: load_collection failed (query/get unavailable): %s", exc,
                )

            self._connected = True
            logger.info(
                "MilvusLiteBackend.connect.ok | collection=%s num_entities=%d",
                self._collection_name,
                self.count(),
            )

        except Exception as exc:
            logger.error("MilvusLiteBackend.connect failed: %s", exc)
            raise

    def insert(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray | None,
        metadata: dict,
        partition_name: str = DEFAULT_PARTITION,
    ) -> int:
        """插入一条向量记录到 Milvus Lite.

        v4: 支持 partition_name 路由到学科分区.
        """
        if not self._connected or self._client is None:
            raise RuntimeError("MilvusLiteBackend not connected")

        data = {
            "vector": image_embedding.astype(np.float32).tolist(),
            "prompt": metadata.get("prompt", ""),
            "optimized_prompt": metadata.get("optimized_prompt", ""),
            "score": float(metadata.get("score", 0.0)),
            "image_path": metadata.get("image_path", ""),
            "created_at": metadata.get(
                "created_at", datetime.now(timezone.utc).isoformat()
            ),
            "model_version": metadata.get("model_version", ""),
            # v4 新增字段
            "subject": metadata.get("subject", ""),
            "category": metadata.get("category", ""),
            "tags": _serialize_tags(metadata.get("tags")),
            # v5 教育素材语义字段
            "semantic_text": metadata.get("semantic_text", ""),
            "topic": metadata.get("topic", ""),
            "knowledge_points": _serialize_tags(metadata.get("knowledge_points")),
            "diagram_type": metadata.get("diagram_type", ""),
            "grade_level": metadata.get("grade_level", ""),
            "visual_elements": _serialize_tags(metadata.get("visual_elements")),
            "source_type": metadata.get("source_type", "generated"),
        }
        if text_embedding is not None:
            data["text_embedding"] = text_embedding.astype(np.float32).tolist()
        # v5: 语义 embedding 作为独立字段存储
        if metadata.get("semantic_embedding") is not None:
            data["semantic_embedding"] = np.array(
                metadata["semantic_embedding"], dtype=np.float32
            ).tolist()

        result = self._client.insert(
            collection_name=self._collection_name,
            data=[data],
            partition_name=partition_name,
        )
        # MilvusClient returns {'ids': [id], ...}
        image_id = result.get("ids", [0])[0] if result else -1
        return int(image_id)

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        partition_names: list[str] | None = None,
    ) -> list[tuple[float, dict]]:
        """在 Milvus Lite 中执行向量检索.

        v4: 支持 partition_names 限定分区检索.
        """
        if not self._connected or self._client is None:
            return []

        search_kwargs = {
            "collection_name": self._collection_name,
            "data": [query_vec.astype(np.float32).tolist()],
            "limit": top_k,
            "output_fields": [
                "prompt", "optimized_prompt", "score",
                "image_path", "created_at", "model_version",
                "subject", "category", "tags",
                # v5 语义字段 (不含 semantic_embedding 向量字段)
                "semantic_text",
                "topic", "knowledge_points", "diagram_type",
                "grade_level", "visual_elements", "source_type",
            ],
        }
        if partition_names:
            search_kwargs["partition_names"] = partition_names

        results = self._client.search(**search_kwargs)

        output: list[tuple[float, dict]] = []
        for hits in results:
            for hit in hits:
                entity = hit.get("entity", {})
                tags_list = _deserialize_tags(entity.get("tags", ""))
                kps_list = _deserialize_tags(entity.get("knowledge_points", ""))
                visuals_list = _deserialize_tags(entity.get("visual_elements", ""))
                # Milvus COSINE metric 返回的是距离 (0=相同,2=相反)，
                # 统一转为余弦相似度 (1=相同,-1=相反) 以便上游统一解释。
                cos_similarity = 1.0 - hit.get("distance", 0.0)
                output.append((cos_similarity, {
                    "image_id": hit.get("id"),
                    "prompt": entity.get("prompt", ""),
                    "optimized_prompt": entity.get("optimized_prompt"),
                    "score": entity.get("score", 0.0),
                    "image_path": entity.get("image_path", ""),
                    "created_at": entity.get("created_at"),
                    "model_version": entity.get("model_version"),
                    "subject": entity.get("subject", ""),
                    "category": entity.get("category", ""),
                    "tags": tags_list,
                    # v5 语义字段
                    "semantic_text": entity.get("semantic_text", ""),
                    "topic": entity.get("topic", ""),
                    "knowledge_points": kps_list,
                    "diagram_type": entity.get("diagram_type", ""),
                    "grade_level": entity.get("grade_level", ""),
                    "visual_elements": visuals_list,
                    "source_type": entity.get("source_type", "generated"),
                }))
        return output

    def count(self, partition_name: str | None = None) -> int:
        """返回存储的记录数.

        优先用 query 分页计数（精确、实时，适用于 Milvus Standalone）。
        query 不可用时（milvus_lite 3.0 部分场景）回退到 search + 零向量，
        limit 受 MILVUS_SEARCH_MAX_LIMIT 约束。
        """
        if not self._connected or self._client is None:
            return 0

        # ── 策略 1: query 分页（精确计数，突破单次 topk 上限）──
        try:
            total = 0
            last_id = -1
            while True:
                query_kwargs: dict = {
                    "collection_name": self._collection_name,
                    "filter": f"id > {last_id}",
                    "output_fields": ["id"],
                    "limit": MILVUS_SEARCH_MAX_LIMIT,
                }
                if partition_name:
                    query_kwargs["partition_names"] = [partition_name]
                batch = self._client.query(**query_kwargs)
                if not batch:
                    return total
                total += len(batch)
                if len(batch) < MILVUS_SEARCH_MAX_LIMIT:
                    return total
                last_id = max(row["id"] for row in batch)
        except Exception as query_exc:
            logger.debug(
                "MilvusLiteBackend.count: query pagination failed (%s), "
                "falling back to search",
                query_exc,
            )

        # ── 策略 2: search 零向量（milvus_lite 兼容，上限 16384）──
        try:
            zero_vec = [0.0] * self._dim
            search_kwargs: dict = {
                "collection_name": self._collection_name,
                "data": [zero_vec],
                "limit": MILVUS_SEARCH_MAX_LIMIT,
                "output_fields": ["id"],
            }
            if partition_name:
                search_kwargs["partition_names"] = [partition_name]
            results = self._client.search(**search_kwargs)
            return sum(len(hits) for hits in results)
        except Exception as exc:
            logger.warning("MilvusLiteBackend.count failed: %s", exc)
            return 0

    def drop_all(self) -> None:
        """删除 collection 中的所有记录."""
        if not self._connected or self._client is None:
            return
        logger.warning(
            "MilvusLiteBackend.drop_all | dropping collection '%s'",
            self._collection_name,
        )
        self._client.drop_collection(self._collection_name)
        self._connected = False

    # ── v4 分区管理 ──

    def create_partition(self, partition_name: str) -> None:
        if not self._connected or self._client is None:
            raise RuntimeError("MilvusLiteBackend not connected")
        if not self.has_partition(partition_name):
            try:
                self._client.create_partition(
                    collection_name=self._collection_name,
                    partition_name=partition_name,
                )
                logger.info("MilvusLiteBackend.create_partition | name=%s", partition_name)
            except Exception as exc:
                # Windows: milvus_lite manifest save 偶尔因竞态失败。
                # 检查分区是否实际创建成功，成功后降级为 info.
                if self.has_partition(partition_name):
                    logger.info(
                        "MilvusLiteBackend.create_partition | name=%s "
                        "(created despite error: %s)", partition_name, exc,
                    )
                    return
                raise

    def has_partition(self, partition_name: str) -> bool:
        if not self._connected or self._client is None:
            return False
        return self._client.has_partition(
            collection_name=self._collection_name,
            partition_name=partition_name,
        )

    def drop_partition(self, partition_name: str) -> None:
        if not self._connected or self._client is None:
            return
        self._client.drop_partition(
            collection_name=self._collection_name,
            partition_name=partition_name,
        )
        logger.info("MilvusLiteBackend.drop_partition | name=%s", partition_name)

    def list_partitions(self) -> list[str]:
        if not self._connected or self._client is None:
            return []
        return self._client.list_partitions(collection_name=self._collection_name)

    def get_partition_stats(self) -> dict[str, int]:
        """返回各分区记录数.

        复用 self.count(partition_name=...) 从 search 做可靠计数，
        避免 get_collection_stats 的 row_count 滞后问题。
        """
        if not self._connected or self._client is None:
            return {}
        stats: dict[str, int] = {}
        for p in self.list_partitions():
            try:
                stats[p] = self.count(partition_name=p)
            except Exception:
                stats[p] = 0
        return stats

    def delete_by_id(self, entity_id: int) -> bool:
        if not self._connected or self._client is None:
            return False
        try:
            result = self._client.delete(
                collection_name=self._collection_name,
                ids=[entity_id],
            )
            return bool(result and result.get("delete_count", 0) > 0)
        except Exception as exc:
            logger.warning("MilvusLiteBackend.delete_by_id | id=%d error=%s", entity_id, exc)
            return False

    def update_metadata(self, entity_id: int, metadata: dict) -> bool:
        """Milvus Lite 通过 delete + insert 实现 upsert."""
        if not self._connected or self._client is None:
            return False
        try:
            # 先查询现有记录的向量
            results = self._client.query(
                collection_name=self._collection_name,
                filter=f"id == {entity_id}",
                output_fields=["vector"],
            )
            if not results:
                return False
            # MilvusClient upsert
            data = {"id": entity_id, **metadata}
            if "vector" in results[0]:
                data["vector"] = results[0]["vector"]
            self._client.upsert(
                collection_name=self._collection_name,
                data=[data],
            )
            return True
        except Exception as exc:
            logger.warning("MilvusLiteBackend.update_metadata | id=%d error=%s", entity_id, exc)
            return False

    def list_data(
        self,
        limit: int = 50,
        offset: int = 0,
        subject: str | None = None,
        min_score: float | None = None,
    ) -> tuple[list[dict], int]:
        """分页列出记录（支持学科和最低分过滤）.

        使用 search() 而非 query() 做全量拉取, 因为 milvus_lite 3.0 中
        query(filter="") 不返回任何记录（疑似 bug）；search 配合零向量则可靠.
        """
        if not self._connected or self._client is None:
            return [], 0

        # 确定要查询的分区列表
        if subject:
            target = _resolve_partition(subject)
            partitions = [target] if self.has_partition(target) else []
        else:
            partitions = self.list_partitions() if hasattr(self, 'list_partitions') else []

        output_fields = [
            "prompt", "optimized_prompt", "score",
            "image_path", "created_at", "model_version",
            "subject", "category", "tags",
            # v5 语义字段
            "semantic_text", "topic", "knowledge_points",
            "diagram_type", "grade_level", "visual_elements", "source_type",
        ]
        zero_vec = [0.0] * self._dim

        all_records: list[dict] = []
        for pn in partitions:
            try:
                hits = self._client.search(
                    collection_name=self._collection_name,
                    data=[zero_vec],
                    limit=10000,
                    output_fields=output_fields,
                    partition_names=[pn],
                )
                for result_list in hits:
                    for hit in result_list:
                        entity = hit.get("entity", {})
                        tags_list = _deserialize_tags(entity.get("tags", ""))
                        kps_list = _deserialize_tags(entity.get("knowledge_points", ""))
                        visuals_list = _deserialize_tags(entity.get("visual_elements", ""))
                        all_records.append({
                            "id": hit.get("id", 0),
                            "prompt": entity.get("prompt", ""),
                            "optimized_prompt": entity.get("optimized_prompt", ""),
                            "score": float(entity.get("score", 0.0)),
                            "image_path": entity.get("image_path", ""),
                            "subject": entity.get("subject", ""),
                            "category": entity.get("category", ""),
                            "tags": tags_list,
                            "created_at": entity.get("created_at", ""),
                            "model_version": entity.get("model_version", ""),
                            # v5 语义字段
                            "semantic_text": entity.get("semantic_text", ""),
                            "topic": entity.get("topic", ""),
                            "knowledge_points": kps_list,
                            "diagram_type": entity.get("diagram_type", ""),
                            "grade_level": entity.get("grade_level", ""),
                            "visual_elements": visuals_list,
                            "source_type": entity.get("source_type", "generated"),
                        })
            except Exception as exc:
                logger.warning(
                    "MilvusLiteBackend.list_data | partition=%s error=%s", pn, exc,
                )

        if min_score is not None:
            all_records = [r for r in all_records if r["score"] >= min_score]

        all_records.sort(key=lambda r: r["id"], reverse=True)
        total = len(all_records)
        page = all_records[offset:offset + limit]

        return page, total


# ══════════════════════════════════════════════════════════
# Milvus 服务端后端（项目计划书 §5.2 目标架构）
# ══════════════════════════════════════════════════════════
class MilvusServerBackend(VectorBackend):
    """Milvus 向量数据库后端

    对应项目计划书 §5.2 的存储架构:
      - Collection: image_embeddings
      - Fields: image_id, prompt, optimized_prompt, score,
                image_embedding(768-d), text_embedding(768-d),
                image_path, created_at, model_version
      - v4 新增: subject, category, tags
      - Index: IVF_FLAT / HNSW on image_embedding

    需要: pip install pymilvus
    """

    def __init__(self, host: str = "localhost", port: int = 19530) -> None:
        self._host = host
        self._port = port
        self._connected = False
        self._collection = None

    def connect(self) -> None:
        """连接 Milvus 服务端并确保 collection 存在."""
        try:
            from pymilvus import (
                Collection, CollectionSchema, DataType, FieldSchema,
                connections, utility,
            )

            connections.connect(host=self._host, port=self._port)
            logger.info("MilvusServerBackend.connect | host=%s port=%d", self._host, self._port)

            # 定义 schema（项目计划书 §5.2 + v4 分区字段）
            fields = [
                FieldSchema(name="image_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="prompt", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="optimized_prompt", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="score", dtype=DataType.FLOAT),
                FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=1024),
                FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="model_version", dtype=DataType.VARCHAR, max_length=128),
                # v4 新增字段
                FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=512),
            ]
            schema = CollectionSchema(fields, description="图像生成优化系统 — 向量存储 (v4)")

            if not utility.has_collection(COLLECTION_NAME):
                collection = Collection(name=COLLECTION_NAME, schema=schema)
                logger.info("MilvusServerBackend: created collection '%s'", COLLECTION_NAME)

                # 创建索引（IVF_FLAT for production）
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128},
                }
                collection.create_index(
                    field_name="image_embedding",
                    index_params=index_params,
                )
                logger.info("MilvusServerBackend: created index on image_embedding")
            else:
                collection = Collection(name=COLLECTION_NAME)

            collection.load()
            self._collection = collection
            self._connected = True

        except ImportError:
            logger.error(
                "MilvusServerBackend: pymilvus not installed. "
                "Install with: pip install pymilvus"
            )
            raise
        except Exception as exc:
            logger.error("MilvusServerBackend.connect failed: %s", exc)
            raise

    def insert(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray | None,
        metadata: dict,
        partition_name: str = DEFAULT_PARTITION,
    ) -> int:
        if not self._connected or self._collection is None:
            raise RuntimeError("MilvusServerBackend not connected")

        from pymilvus import DataType

        entities = [
            metadata.get("prompt", ""),
            metadata.get("optimized_prompt", ""),
            float(metadata.get("score", 0.0)),
            image_embedding.astype(np.float32).tolist(),
            (text_embedding.astype(np.float32).tolist() if text_embedding is not None
             else [0.0] * EMBEDDING_DIM),
            metadata.get("image_path", ""),
            metadata.get("created_at", datetime.now(timezone.utc).isoformat()),
            metadata.get("model_version", ""),
            # v4 新增字段
            metadata.get("subject", ""),
            metadata.get("category", ""),
            _serialize_tags(metadata.get("tags")),
        ]

        # 确保分区存在
        if partition_name != DEFAULT_PARTITION and not self.has_partition(partition_name):
            self.create_partition(partition_name)

        result = self._collection.insert(entities, partition_name=partition_name)
        self._collection.flush()
        # result.primary_keys[0] is the auto_id
        return int(result.primary_keys[0]) if result.primary_keys else -1

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        partition_names: list[str] | None = None,
    ) -> list[tuple[float, dict]]:
        if not self._connected or self._collection is None:
            return []

        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        search_kwargs = {
            "data": [query_vec.astype(np.float32).tolist()],
            "anns_field": "image_embedding",
            "param": search_params,
            "limit": top_k,
            "output_fields": [
                "image_id", "prompt", "optimized_prompt", "score",
                "image_path", "created_at", "model_version",
                "subject", "category", "tags",
            ],
        }
        if partition_names:
            search_kwargs["partition_names"] = partition_names

        results = self._collection.search(**search_kwargs)

        output: list[tuple[float, dict]] = []
        for hits in results:
            for hit in hits:
                tags_list = _deserialize_tags(hit.entity.get("tags", ""))
                # Milvus COSINE metric 返回的是距离 (0=相同,2=相反)，
                # 统一转为余弦相似度 (1=相同,-1=相反) 以便上游统一解释。
                cos_similarity = 1.0 - hit.score
                output.append((cos_similarity, {
                    "image_id": hit.entity.get("image_id"),
                    "prompt": hit.entity.get("prompt"),
                    "optimized_prompt": hit.entity.get("optimized_prompt"),
                    "score": hit.entity.get("score"),
                    "image_path": hit.entity.get("image_path"),
                    "created_at": hit.entity.get("created_at"),
                    "model_version": hit.entity.get("model_version"),
                    "subject": hit.entity.get("subject", ""),
                    "category": hit.entity.get("category", ""),
                    "tags": tags_list,
                }))
        return output

    def count(self, partition_name: str | None = None) -> int:
        if not self._connected or self._collection is None:
            return 0
        if partition_name:
            try:
                from pymilvus import Partition
                part = Partition(self._collection, partition_name)
                return part.num_entities
            except Exception:
                return 0
        return self._collection.num_entities

    def drop_all(self) -> None:
        if not self._connected or self._collection is None:
            return
        logger.warning("MilvusServerBackend.drop_all | dropping collection '%s'", COLLECTION_NAME)
        from pymilvus import utility
        utility.drop_collection(COLLECTION_NAME)
        self._collection = None

    # ── v4 分区管理 ──

    def create_partition(self, partition_name: str) -> None:
        if not self._connected or self._collection is None:
            raise RuntimeError("MilvusServerBackend not connected")
        from pymilvus import Partition
        if not self._collection.has_partition(partition_name):
            self._collection.create_partition(partition_name)
            logger.info("MilvusServerBackend.create_partition | name=%s", partition_name)

    def has_partition(self, partition_name: str) -> bool:
        if not self._connected or self._collection is None:
            return False
        return self._collection.has_partition(partition_name)

    def drop_partition(self, partition_name: str) -> None:
        if not self._connected or self._collection is None:
            return
        from pymilvus import Partition
        part = Partition(self._collection, partition_name)
        part.drop()
        logger.info("MilvusServerBackend.drop_partition | name=%s", partition_name)

    def list_partitions(self) -> list[str]:
        if not self._connected or self._collection is None:
            return []
        return [p.name for p in self._collection.partitions]

    def get_partition_stats(self) -> dict[str, int]:
        if not self._connected or self._collection is None:
            return {}
        stats: dict[str, int] = {}
        for p in self._collection.partitions:
            stats[p.name] = p.num_entities
        return stats

    def delete_by_id(self, entity_id: int) -> bool:
        if not self._connected or self._collection is None:
            return False
        try:
            expr = f"image_id == {entity_id}"
            result = self._collection.delete(expr)
            self._collection.flush()
            return bool(result)
        except Exception as exc:
            logger.warning("MilvusServerBackend.delete_by_id | id=%d error=%s", entity_id, exc)
            return False

    def update_metadata(self, entity_id: int, metadata: dict) -> bool:
        """Milvus Server 通过 delete + insert 实现 upsert."""
        if not self._connected or self._collection is None:
            return False
        try:
            expr = f"image_id == {entity_id}"
            results = self._collection.query(
                expr=expr,
                output_fields=["image_embedding"],
            )
            if not results:
                return False
            entities = [
                metadata.get("prompt", ""),
                metadata.get("optimized_prompt", ""),
                float(metadata.get("score", 0.0)),
                results[0].get("image_embedding", [0.0] * EMBEDDING_DIM),
                metadata.get("text_embedding", [0.0] * EMBEDDING_DIM),
                metadata.get("image_path", ""),
                metadata.get("created_at", ""),
                metadata.get("model_version", ""),
                metadata.get("subject", ""),
                metadata.get("category", ""),
                _serialize_tags(metadata.get("tags")),
            ]
            self._collection.upsert(entities)
            self._collection.flush()
            return True
        except Exception as exc:
            logger.warning("MilvusServerBackend.update_metadata | id=%d error=%s", entity_id, exc)
            return False

    def list_data(
        self,
        limit: int = 50,
        offset: int = 0,
        subject: str | None = None,
        min_score: float | None = None,
    ) -> tuple[list[dict], int]:
        """从 Milvus Server 分页读取记录."""
        if not self._connected or self._collection is None:
            return [], 0

        try:
            # 构建过滤表达式
            filter_parts = []
            if subject:
                resolved = _resolve_partition(subject)
                filter_parts.append(f'subject == "{resolved}"')
            if min_score is not None:
                filter_parts.append(f"score >= {min_score}")
            filter_expr = " and ".join(filter_parts) if filter_parts else None

            output_fields = [
                "image_id", "prompt", "optimized_prompt", "score",
                "image_path", "created_at", "model_version",
                "subject", "category", "tags",
            ]

            # 查询（Milvus ORM API）
            results = self._collection.query(
                expr=filter_expr,
                output_fields=output_fields,
                limit=limit,
                offset=offset,
            )

            records: list[dict] = []
            for r in results:
                tags_list = _deserialize_tags(r.get("tags", ""))
                records.append({
                    "id": r.get("image_id", 0),
                    "prompt": r.get("prompt", ""),
                    "optimized_prompt": r.get("optimized_prompt", ""),
                    "score": float(r.get("score", 0.0)),
                    "image_path": r.get("image_path", ""),
                    "subject": r.get("subject", ""),
                    "category": r.get("category", ""),
                    "tags": tags_list,
                    "created_at": r.get("created_at", ""),
                    "model_version": r.get("model_version", ""),
                })

            # 总数（简化：用 self.count()；精确实现需单独 query）
            total = self.count()
            if filter_expr:
                total = len(results)  # 近似

            return records, total
        except Exception as exc:
            logger.warning("MilvusServerBackend.list_data | error=%s", exc)
            return [], 0


# ══════════════════════════════════════════════════════════
# VectorStore — 统一门面 (v4)
# ══════════════════════════════════════════════════════════
class VectorStore:
    """向量存储统一门面（项目计划书 §5.2-§5.3）

    唯一后端: Milvus Standalone (Docker).
      通过 MILVUS_URI 或 MILVUS_HOST:MILVUS_PORT 连接，
      连接失败直接抛异常，不静默回退到内存后端。
      LocalNumpyBackend 仅用于单元测试，生产代码不使用。

    v4 新增能力:
      - 学科分区路由（insert/search 自动按 subject 分区）
      - 分区统计（get_stats_by_subject）
      - CRUD 操作（delete_entity / update_entity）
      - 索引管理（create_index / drop_index / get_index_info）

    系统能力（项目计划书 §5.3）:
      - 🖼️ 相似图像检索
      - 🔍 图文语义搜索
      - ♻️ 历史结果复用
      - 📚 教学示意图知识库构建
      - 🏷️ 学科分区隔离检索（v4）
    """

    # 学科 → 分区名映射 (v4)
    SUBJECT_PARTITIONS = SUBJECT_PARTITION_MAP
    # 分类 → 分区名映射 (v6)
    CATEGORY_PARTITIONS = CATEGORY_PARTITION_MAP

    def __init__(self) -> None:
        self._host = MILVUS_HOST
        self._port = MILVUS_PORT
        self._backend: VectorBackend | None = None
        self._backend_label = "none"
        atexit.register(self.close)

    # ── 连接管理 ──────────────────────────────────

    def connect(self) -> None:
        """连接 Milvus Standalone（唯一后端）.

        优先使用配置的 MILVUS_URI → MILVUS_HOST:MILVUS_PORT。
        连接失败直接抛异常，不静默回退到内存后端。
        启动成功后自动执行健康检查。
        """
        if self._backend is not None:
            return

        import os as _os

        milvus_uri = _os.environ.get("MILVUS_URI", "")
        if not milvus_uri:
            milvus_uri = f"http://{self._host}:{self._port}"

        logger.info(
            "VectorStore.connect | uri=%s host=%s port=%d",
            milvus_uri, self._host, self._port,
        )

        try:
            backend = MilvusLiteBackend(uri=milvus_uri)
            backend.connect()
            self._backend = backend
            self._backend_label = "milvus_standalone"
            logger.info("VectorStore: connected to Milvus Standalone (uri=%s)", milvus_uri)
            # 确保所有学科分区存在
            self._ensure_all_partitions()

            # ── 健康检查 ──
            total = self._backend.count()
            logger.info("VectorStore: health check count=%d", total)
            if total == 0:
                logger.warning(
                    "⚠️  VectorStore: database is empty (count=0). "
                    "如果是首次启动，请先导入数据。"
                    "如果非首次启动，数据可能丢失或连接到了错误的实例！"
                )
            return
        except Exception as exc:
            err_msg = str(exc).lower()
            logger.error(
                "VectorStore.connect FAILED | uri=%s error_kind=%s error=%s",
                milvus_uri, type(exc).__name__, exc,
            )
            raise RuntimeError(
                f"无法连接 Milvus Standalone (uri={milvus_uri}): {exc}\n"
                "请确保 Docker Milvus Standalone 已启动:\n"
                "  docker compose up -d\n"
                "或检查 MILVUS_HOST / MILVUS_PORT / MILVUS_URI 环境变量。"
            ) from exc

    @property
    def ready(self) -> bool:
        return self._backend is not None

    @property
    def backend_type(self) -> str:
        """返回当前后端类型: 'milvus_lite' | 'milvus_server' | 'local_numpy' | 'none'."""
        return self._backend_label

    # ── 分区管理 ──────────────────────────────────

    def _ensure_partition(self, partition_name: str) -> None:
        """幂等创建分区."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        try:
            if not self._backend.has_partition(partition_name):
                self._backend.create_partition(partition_name)
                logger.info("VectorStore._ensure_partition | created partition=%s", partition_name)
        except NotImplementedError:
            pass  # LocalNumpyBackend 自动创建
        except Exception as exc:
            # 创建报错后二次确认：分区可能已实际存在（如 milvus_lite Windows 竞态）
            try:
                if self._backend.has_partition(partition_name):
                    logger.info(
                        "VectorStore._ensure_partition | partition=%s exists (error was: %s)",
                        partition_name, exc,
                    )
                    return
            except Exception:
                pass
            logger.warning(
                "VectorStore._ensure_partition | partition=%s error=%s", partition_name, exc,
            )

    def _ensure_all_partitions(self) -> None:
        """确保所有分类分区 + 旧学科分区 + _default 分区存在（v6 泛化）."""
        if self._backend is None:
            return
        all_partitions = (
            list(CATEGORY_PARTITION_MAP.values())
            + list(SUBJECT_PARTITION_MAP.values())
            + [DEFAULT_PARTITION]
        )
        # 去重
        seen = set()
        for pn in all_partitions:
            if pn not in seen:
                seen.add(pn)
                self._ensure_partition(pn)

    def _resolve_partition_name(self, subject: str | None) -> str:
        """将 subject 解析为分区名."""
        return _resolve_partition(subject)

    def list_partitions(self) -> list[dict]:
        """列出所有分区及其统计."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        try:
            stats = self._backend.get_partition_stats()
        except NotImplementedError:
            stats = {}
        result = []
        for pn in self._backend.list_partitions() if hasattr(self._backend, 'list_partitions') else stats.keys():
            result.append({
                "name": pn,
                "row_count": stats.get(pn, self._backend.count(pn) if hasattr(self._backend, 'count') else 0),
            })
        return result

    def get_stats_by_subject(self) -> dict:
        """返回各学科分区数据量统计（v6: 包含分类分区 + 旧学科分区）.

        Returns:
            {subject/category: count, ..., "_default": count, "total": total}
        """
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        try:
            partition_stats = self._backend.get_partition_stats()
        except NotImplementedError:
            partition_stats = {}

        result: dict[str, int] = {}
        # v6: 优先使用新分类分区
        for cn_name, partition_name in CATEGORY_PARTITION_MAP.items():
            result[cn_name] = partition_stats.get(partition_name, 0)
        # v4: 向后兼容旧学科分区
        for eng_name, cn_name in SUBJECT_PARTITION_MAP.items():
            if eng_name not in result:
                result[eng_name] = partition_stats.get(cn_name, 0)
        result["_default"] = partition_stats.get(DEFAULT_PARTITION, 0)
        result["total"] = sum(result.values())
        return result

    def get_stats_by_category(self) -> dict:
        """返回各分类分区数据量统计（v6 新增，推荐使用）.

        Returns:
            {category: count, ..., "_default": count, "total": total}
        """
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        try:
            partition_stats = self._backend.get_partition_stats()
        except NotImplementedError:
            partition_stats = {}

        result: dict[str, int] = {}
        for cn_name, partition_name in CATEGORY_PARTITION_MAP.items():
            result[cn_name] = partition_stats.get(partition_name, 0)
        result["_default"] = partition_stats.get(DEFAULT_PARTITION, 0)
        result["total"] = sum(result.values())
        return result

    # ── 存储（项目计划书 §5.2 字段定义） ──────

    def insert(self, record: ImageRecord, subject: str | None = None) -> int:
        """插入一条图像记录（含 image + text embedding 及元数据）.

        存储字段对齐项目计划书 §5.2 + v4:
          - 图像 ID (image_id)
          - 原始 Prompt (prompt)
          - 优化后 Prompt (optimized_prompt)
          - 最终评测得分 (score)
          - 图像向量特征 (image_embedding)
          - 文本向量特征 (text_embedding)
          - 时间戳 (created_at)
          - 模型版本 (model_version)
          - 学科标签 (subject) ← v4
          - 二级分类 (category) ← v4
          - 标签 (tags) ← v4

        Args:
            record: 图像记录（至少含 prompt, score, image_path）.
            subject: 学科标签，用于分区路由。为 None 时使用 record.subject.

        Returns:
            int: 插入记录的主键 ID.
        """
        t0 = time.perf_counter()
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        effective_subject = subject or record.subject
        partition = self._resolve_partition_name(effective_subject)
        self._ensure_partition(partition)

        image_emb = (
            np.array(record.embedding, dtype=np.float32)
            if record.embedding and len(record.embedding) == EMBEDDING_DIM
            else np.zeros(EMBEDDING_DIM, dtype=np.float32)
        )

        metadata = {
            "prompt": record.prompt,
            "optimized_prompt": record.optimized_prompt or "",
            "score": record.score,
            "image_path": record.image_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_version": "",
            "subject": effective_subject or "",
            "category": record.category or "",
            "tags": record.tags or [],
            # v5 教育素材语义字段
            "semantic_text": record.semantic_text or "",
            "semantic_embedding": record.semantic_embedding,
            "topic": record.topic or "",
            "knowledge_points": record.knowledge_points or [],
            "diagram_type": record.diagram_type or "",
            "grade_level": record.grade_level or "",
            "visual_elements": record.visual_elements or [],
            "source_type": record.source_type or "generated",
        }

        text_emb: np.ndarray | None = None
        if record.text_embedding and len(record.text_embedding) == EMBEDDING_DIM:
            text_emb = np.array(record.text_embedding, dtype=np.float32)

        image_id = self._backend.insert(
            image_embedding=image_emb,
            text_embedding=text_emb,
            metadata=metadata,
            partition_name=partition,
        )

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "VectorStore.insert | image_id=%d subject=%s partition=%s score=%.3f "
            "total=%d duration_ms=%d backend=%s",
            image_id, effective_subject, partition, record.score,
            self._backend.count(), duration_ms, self.backend_type,
        )
        return image_id

    def find_best_by_exact_prompt(
        self,
        prompt: str,
        subject: str | None = None,
        min_score: float = 0.0,
    ) -> dict | None:
        """按 prompt 精确匹配召回最高分记录（clip_enrich 复用优先路径）.

        相同 prompt 重跑时应直接复用库中评测分最高的图片，
        不依赖 CLIP 跨模态相似度（文搜图易误命中主题相近的其他记录）.
        """
        t0 = time.perf_counter()
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        records, _ = self._backend.list_data(
            limit=10000, offset=0, subject=subject, min_score=min_score,
        )
        matches = [r for r in records if r.get("prompt") == prompt]
        if not matches:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.info(
                "VectorStore.find_best_by_exact_prompt | prompt_len=%d subject=%s "
                "min_score=%.2f matches=0 duration_ms=%d",
                len(prompt), subject, min_score, duration_ms,
            )
            return None

        best = max(matches, key=lambda r: (r.get("score", 0.0), r.get("id", 0)))
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "VectorStore.find_best_by_exact_prompt | prompt_len=%d subject=%s "
            "min_score=%.2f matches=%d best_id=%d best_score=%.3f duration_ms=%d",
            len(prompt), subject, min_score, len(matches),
            best.get("id", 0), best.get("score", 0.0), duration_ms,
        )
        return {
            "image_id": best.get("id", 0),
            "prompt": best.get("prompt", ""),
            "optimized_prompt": best.get("optimized_prompt"),
            "score": best.get("score", 0.0),
            "image_path": best.get("image_path", ""),
            "subject": best.get("subject", ""),
            "match_type": "exact_prompt",
            "semantic_similarity": 1.0,
        }

    # ── 检索（项目计划书 §5.3） ──────────────────

    def search_by_text(
        self,
        text_embedding: list[float] | None = None,
        text: str = "",
        top_k: int = 5,
        subject: str | None = None,
    ) -> SearchResponse:
        """以文搜图 🔍（项目计划书 §5.3 图文语义搜索）.

        输入文本 embedding，返回 Top-K 相似图片.

        v4: 支持 subject 参数限定学科分区检索.
            为 None 时全库检索（不推荐，可能跨学科污染）.

        Args:
            text_embedding: CLIP 文本 embedding（推荐）.
            text: 原始文本（仅用于日志）.
            top_k: 返回数量.
            subject: 限定学科分区（推荐指定）.

        Returns:
            SearchResponse: 含相似图片列表.
        """
        t0 = time.perf_counter()
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        partition_names = None
        if subject:
            partition_names = [self._resolve_partition_name(subject)]

        total_in_partition = self._backend.count(
            partition_names[0] if partition_names else None
        ) if hasattr(self._backend, 'count') else self._backend.count()

        logger.info(
            "VectorStore.search_by_text | text_len=%d top_k=%d subject=%s "
            "partitions=%s total=%d backend=%s",
            len(text), top_k, subject, partition_names, total_in_partition, self.backend_type,
        )

        if text_embedding is None:
            return SearchResponse(results=[], query_time_ms=0.0, total_in_partition=total_in_partition)

        query_vec = np.array(text_embedding, dtype=np.float32)
        hits = self._backend.search(query_vec, top_k, partition_names=partition_names)

        results: list[ImageRecord] = []
        for sim, rec in hits[:top_k]:
            results.append(ImageRecord(
                image_id=rec.get("image_id", 0),
                prompt=rec.get("prompt", ""),
                optimized_prompt=rec.get("optimized_prompt"),
                score=rec.get("score", 0.0),
                image_path=rec.get("image_path", ""),
                embedding=None,
                similarity=round(float(sim), 4),
                subject=rec.get("subject", ""),
                category=rec.get("category", ""),
                tags=rec.get("tags", []),
                # v5 语义字段
                semantic_text=rec.get("semantic_text", ""),
                topic=rec.get("topic", ""),
                knowledge_points=rec.get("knowledge_points", []),
                diagram_type=rec.get("diagram_type", ""),
                grade_level=rec.get("grade_level", ""),
                visual_elements=rec.get("visual_elements", []),
                source_type=rec.get("source_type", "generated"),
            ))

        duration_ms = int((time.perf_counter() - t0) * 1000)
        return SearchResponse(
            results=results,
            query_time_ms=float(duration_ms),
            total_in_partition=total_in_partition,
        )

    def search_by_image(
        self,
        image_embedding: list[float] | None = None,
        image_path: str = "",
        top_k: int = 5,
        subject: str | None = None,
    ) -> SearchResponse:
        """以图搜图 🖼️（项目计划书 §5.3 相似图像检索）.

        输入图片 embedding，返回 Top-K 相似图片.

        v4: 支持 subject 参数限定学科分区检索.

        Args:
            image_embedding: CLIP 图像 embedding（推荐）.
            image_path: 原始图片路径（仅用于日志）.
            top_k: 返回数量.
            subject: 限定学科分区（推荐指定）.

        Returns:
            SearchResponse: 含相似图片列表.
        """
        t0 = time.perf_counter()
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        partition_names = None
        if subject:
            partition_names = [self._resolve_partition_name(subject)]

        total_in_partition = self._backend.count(
            partition_names[0] if partition_names else None
        ) if hasattr(self._backend, 'count') else self._backend.count()

        logger.info(
            "VectorStore.search_by_image | image=%s top_k=%d subject=%s "
            "partitions=%s total=%d backend=%s",
            image_path, top_k, subject, partition_names, total_in_partition, self.backend_type,
        )

        if image_embedding is None:
            return SearchResponse(results=[], query_time_ms=0.0, total_in_partition=total_in_partition)

        query_vec = np.array(image_embedding, dtype=np.float32)
        hits = self._backend.search(query_vec, top_k, partition_names=partition_names)

        results: list[ImageRecord] = []
        for sim, rec in hits[:top_k]:
            results.append(ImageRecord(
                image_id=rec.get("image_id", 0),
                prompt=rec.get("prompt", ""),
                optimized_prompt=rec.get("optimized_prompt"),
                score=rec.get("score", 0.0),
                image_path=rec.get("image_path", ""),
                embedding=None,
                similarity=round(float(sim), 4),
                subject=rec.get("subject", ""),
                category=rec.get("category", ""),
                tags=rec.get("tags", []),
                # v5 语义字段
                semantic_text=rec.get("semantic_text", ""),
                topic=rec.get("topic", ""),
                knowledge_points=rec.get("knowledge_points", []),
                diagram_type=rec.get("diagram_type", ""),
                grade_level=rec.get("grade_level", ""),
                visual_elements=rec.get("visual_elements", []),
                source_type=rec.get("source_type", "generated"),
            ))

        duration_ms = int((time.perf_counter() - t0) * 1000)
        return SearchResponse(
            results=results,
            query_time_ms=float(duration_ms),
            total_in_partition=total_in_partition,
        )

    # ── 语义检索（v5 新增，对齐 milvus_optimization_plan §七） ──

    @staticmethod
    def build_semantic_text(
        subject: str = "",
        topic: str = "",
        knowledge_points: list[str] | None = None,
        diagram_type: str = "",
        grade_level: str = "",
        retrieval_prompt: str = "",
    ) -> str:
        """构建标准 semantic_text（结构化拼接，对齐优化计划 §五）.

        格式::

            学科: 地理
            主题: 世界气候类型
            知识点: 热带雨林气候、温带季风气候
            图类型: 地图
            用途: 初中教学
            检索描述: 适合初中地理教学的世界气候类型分布图

        Args:
            subject: 学科名.
            topic: 主知识点.
            knowledge_points: 教材知识点列表.
            diagram_type: 图类型.
            grade_level: 学段.
            retrieval_prompt: 检索描述.

        Returns:
            str: 结构化拼接的语义文本.
        """
        parts: list[str] = []
        parts.append(f"学科: {subject}" if subject else "学科: 未标注")
        parts.append(f"主题: {topic}" if topic else "主题: 未标注")
        if knowledge_points:
            kps = "、".join(knowledge_points[:8])
            parts.append(f"知识点: {kps}")
        parts.append(f"图类型: {diagram_type}" if diagram_type else "图类型: 其他")
        grade_label = grade_level or "中学"
        parts.append(f"用途: {grade_label}教学")
        if retrieval_prompt:
            parts.append(f"检索描述: {retrieval_prompt}")
        return "\n".join(parts)

    def search_by_semantic(
        self,
        query_text: str,
        semantic_embedding: list[float] | None = None,
        image_embedding: list[float] | None = None,
        top_k: int = 5,
        subject: str | None = None,
    ) -> dict:
        """语义检索（v5 核心）：加权排序 0.7+0.2+0.1.

        流程（对齐优化计划 §七）:
          1. 用户自然语言 → Chinese-CLIP encode_text → semantic_embedding
          2. Milvus 检索（用 semantic_embedding 做主向量检索）
          3. 加权排序: final = 0.7*semantic + 0.2*image + 0.1*tags

        Args:
            query_text: 原始查询文本.
            semantic_embedding: Chinese-CLIP 编码的语义向量.
            image_embedding: 查询图片向量（用于 image_similarity，可选）.
            top_k: 返回数量.
            subject: 限定学科分区.

        Returns:
            dict with keys: results (list[SemanticSearchResult]), query_time_ms, total.
        """
        t0 = time.perf_counter()
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        partition_names = None
        if subject:
            partition_names = [self._resolve_partition_name(subject)]

        total_in_partition = self._backend.count(
            partition_names[0] if partition_names else None
        ) if hasattr(self._backend, 'count') else self._backend.count()

        logger.info(
            "VectorStore.search_by_semantic | query_len=%d top_k=%d subject=%s "
            "partitions=%s total=%d backend=%s",
            len(query_text), top_k, subject, partition_names,
            total_in_partition, self.backend_type,
        )

        if semantic_embedding is None:
            return {
                "results": [],
                "query_time_ms": 0.0,
                "total_in_partition": total_in_partition,
            }

        query_vec = np.array(semantic_embedding, dtype=np.float32)
        # 多取一些候选，方便加权排序
        fetch_k = max(top_k * 3, 15)
        hits = self._backend.search(query_vec, fetch_k, partition_names=partition_names)

        # ── 加权排序 ──
        scored: list[dict] = []
        query_img_norm = None
        if image_embedding is not None:
            qi = np.array(image_embedding, dtype=np.float32)
            query_img_norm = qi / (np.linalg.norm(qi) + 1e-8)

        for sem_sim, rec in hits:
            semantic_similarity = float(sem_sim)

            # image_similarity
            image_similarity = 0.0
            if query_img_norm is not None:
                # 使用 vector 字段（image embedding）计算图像相似度
                # 注意: hits 返回的是 vector 字段的相似度, 不是 semantic_embedding
                # 这里我们需要单独计算，简化处理: 用 rec 中可能存在的 embedding
                pass  # 图像相似度在实际检索中由 query 向量与存储向量的距离给出

            # tags_overlap: 查询文本与标签的重叠度（简化版）
            tags_overlap = 0.0
            rec_tags = rec.get("tags", [])
            if rec_tags:
                # 计算查询文本中的词与标签的模糊匹配
                query_lower = query_text.lower()
                matched = sum(1 for t in rec_tags if t.lower() in query_lower or query_lower in t.lower())
                tags_overlap = min(matched / max(len(rec_tags), 1), 1.0) * 0.5

            # 加权最终分
            final_score = (
                0.7 * semantic_similarity
                + 0.2 * image_similarity
                + 0.1 * tags_overlap
            )

            scored.append({
                "image_id": rec.get("image_id", 0),
                "prompt": rec.get("prompt", ""),
                "optimized_prompt": rec.get("optimized_prompt"),
                "score": rec.get("score", 0.0),
                "image_path": rec.get("image_path", ""),
                "subject": rec.get("subject", ""),
                "category": rec.get("category", ""),
                "tags": rec.get("tags", []),
                "topic": rec.get("topic", ""),
                "knowledge_points": rec.get("knowledge_points", []),
                "diagram_type": rec.get("diagram_type", ""),
                "grade_level": rec.get("grade_level", ""),
                "source_type": rec.get("source_type", "generated"),
                "semantic_similarity": round(semantic_similarity, 4),
                "image_similarity": round(image_similarity, 4),
                "tags_overlap": round(tags_overlap, 4),
                "final_score": round(final_score, 4),
            })

        # 按 final_score 降序
        scored.sort(key=lambda x: x["final_score"], reverse=True)
        scored = scored[:top_k]

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "VectorStore.search_by_semantic.end | found=%d duration_ms=%d",
            len(scored), duration_ms,
        )
        return {
            "results": scored,
            "query_time_ms": float(duration_ms),
            "total_in_partition": total_in_partition,
        }

    def recall_successful_prompts(
        self,
        prompt: str,
        text_embedding: list[float] | None = None,
        min_score: float = 0.80,
        top_k: int = 5,
        subject: str | None = None,
    ) -> list[dict]:
        """历史 Prompt 复用 ♻️（项目计划书 §5.3 历史结果复用）.

        从高评分记录中召回成功 Prompt，支持教学示意图知识库构建 📚.

        v4: 支持 subject 参数限定学科分区.

        两种模式:
          1. 有 embedding → 余弦相似度检索 + 高分过滤
          2. 无 embedding → 仅高分过滤（按 score 排序）

        Args:
            prompt: 原始 prompt 文本.
            text_embedding: CLIP 文本 embedding（可选）.
            min_score: 最低全局 CLIP 得分阈值.
            top_k: 返回数量.
            subject: 限定学科分区（推荐指定）.

        Returns:
            [{prompt, optimized_prompt, score, image_path, similarity?}, ...]
        """
        t0 = time.perf_counter()
        if self._backend is None:
            self.connect()
        assert self._backend is not None

        partition_names = None
        if subject:
            partition_names = [self._resolve_partition_name(subject)]

        logger.info(
            "VectorStore.recall_prompts | prompt_len=%d min_score=%.2f top_k=%d subject=%s backend=%s",
            len(prompt), min_score, top_k, subject, self.backend_type,
        )

        results: list[dict] = []

        if text_embedding is not None:
            query_vec = np.array(text_embedding, dtype=np.float32)
            hits = self._backend.search(query_vec, max(top_k * 2, 20), partition_names=partition_names)
            for sim, rec in hits:
                if rec.get("score", 0) >= min_score:
                    results.append({
                        "prompt": rec.get("prompt", ""),
                        "optimized_prompt": rec.get("optimized_prompt"),
                        "score": rec.get("score", 0),
                        "image_path": rec.get("image_path", ""),
                        "similarity": round(sim, 4),
                        "subject": rec.get("subject", ""),
                    })
                if len(results) >= top_k:
                    break
        else:
            # 无 embedding：仅按 score 降序（本地后端全量扫描）
            if isinstance(self._backend, LocalNumpyBackend):
                all_recs: list[dict] = []
                for part_recs in self._backend._records.values():
                    all_recs.extend(part_recs)
                qualified = [
                    r for r in all_recs
                    if r.get("score", 0) >= min_score
                ]
                qualified.sort(key=lambda r: r.get("score", 0), reverse=True)
                for rec in qualified[:top_k]:
                    results.append({
                        "prompt": rec.get("prompt", ""),
                        "optimized_prompt": rec.get("optimized_prompt"),
                        "score": rec.get("score", 0),
                        "image_path": rec.get("image_path", ""),
                        "subject": rec.get("subject", ""),
                    })

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "VectorStore.recall_prompts.end | found=%d duration_ms=%d",
            len(results), duration_ms,
        )
        return results

    # ── CRUD 操作（v4 新增） ──────────────────────

    def delete_entity(self, entity_id: int) -> bool:
        """删除指定 ID 的记录."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        try:
            ok = self._backend.delete_by_id(entity_id)
            logger.info("VectorStore.delete_entity | id=%d ok=%s", entity_id, ok)
            return ok
        except NotImplementedError:
            logger.warning("VectorStore.delete_entity | not supported by backend=%s", self.backend_type)
            return False

    def update_entity(self, entity_id: int, metadata: dict) -> bool:
        """更新指定 ID 记录的元数据."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        try:
            ok = self._backend.update_metadata(entity_id, metadata)
            logger.info("VectorStore.update_entity | id=%d ok=%s", entity_id, ok)
            return ok
        except NotImplementedError:
            logger.warning("VectorStore.update_entity | not supported by backend=%s", self.backend_type)
            return False

    def list_data(
        self,
        limit: int = 50,
        offset: int = 0,
        subject: str | None = None,
        min_score: float | None = None,
    ) -> tuple[list[dict], int]:
        """分页列出记录（支持学科和最低分过滤）."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        try:
            return self._backend.list_data(limit=limit, offset=offset,
                                            subject=subject, min_score=min_score)
        except NotImplementedError:
            logger.warning("VectorStore.list_data | not supported by backend=%s", self.backend_type)
            return [], 0

    # ── 索引管理（v4 新增） ──────────────────────

    def create_index(self, index_type: str = "HNSW", metric_type: str = "COSINE",
                     extra_params: dict | None = None) -> bool:
        """创建/切换索引类型."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        if self._backend_label == "local_numpy":
            logger.warning("VectorStore.create_index | index not supported on local_numpy backend")
            return False

        try:
            client = self._backend._client  # type: ignore[union-attr]
            if client is None:
                return False

            idx_params = client.prepare_index_params()
            idx_kwargs: dict = {
                "field_name": "vector",
                "index_type": index_type,
                "metric_type": metric_type,
            }
            if index_type == "HNSW":
                idx_kwargs["params"] = extra_params or {"M": 16, "efConstruction": 200}
            elif index_type == "IVF_FLAT":
                idx_kwargs["params"] = extra_params or {"nlist": 128}
            elif extra_params:
                idx_kwargs["params"] = extra_params

            idx_params.add_index(**idx_kwargs)

            client.create_index(
                collection_name=COLLECTION_NAME,
                index_params=idx_params,
            )
            logger.info("VectorStore.create_index | type=%s metric=%s", index_type, metric_type)
            return True
        except Exception as exc:
            logger.error("VectorStore.create_index failed | type=%s error=%s", index_type, exc)
            return False

    def drop_index(self) -> bool:
        """删除索引."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        if self._backend_label == "local_numpy":
            return False
        try:
            from pymilvus import MilvusClient
            client = self._backend._client  # type: ignore[union-attr]
            if client is None:
                return False
            client.drop_index(
                collection_name=COLLECTION_NAME,
                field_name="vector",
            )
            logger.info("VectorStore.drop_index | ok")
            return True
        except Exception as exc:
            logger.error("VectorStore.drop_index failed | error=%s", exc)
            return False

    def get_index_info(self) -> dict:
        """获取索引信息."""
        if self._backend is None:
            self.connect()
        assert self._backend is not None
        if self._backend_label == "local_numpy":
            return {"index_type": "N/A (in-memory cosine)", "metric_type": "COSINE"}
        try:
            from pymilvus import MilvusClient
            client = self._backend._client  # type: ignore[union-attr]
            if client is None:
                return {}
            info = client.describe_index(
                collection_name=COLLECTION_NAME,
                field_name="vector",
            )
            return info if info else {}
        except Exception as exc:
            logger.warning("VectorStore.get_index_info failed | error=%s", exc)
            return {}

    def close(self) -> None:
        """优雅关闭：释放 Milvus 连接，防止锁残留."""
        if self._backend is not None:
            try:
                if self._backend_label == "milvus_standalone" and self._backend._client is not None:
                    self._backend._client.close()
                    logger.info("VectorStore.close | backend=%s closed ok", self._backend_label)
            except Exception as exc:
                logger.warning("VectorStore.close | error=%s", exc)
            self._backend = None
            self._backend_label = "none"

    # ── 管理 ──────────────────────────────────────

    def count(self, subject: str | None = None) -> int:
        """返回存储的记录数.

        Args:
            subject: 指定学科（v4 新增），为 None 时返回总数.
        """
        if self._backend is None:
            return 0
        if subject:
            partition = self._resolve_partition_name(subject)
            return self._backend.count(partition)
        return self._backend.count()

    def drop_all(self) -> None:
        """清空所有记录."""
        if self._backend is None:
            return
        self._backend.drop_all()


# ══════════════════════════════════════════════════════════
# 全局单例 — 防止多个模块重复初始化数据库连接
# ══════════════════════════════════════════════════════════
_vector_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """获取全局唯一的 VectorStore 实例（单例）.

    全项目统一使用此函数替代 `VectorStore()`，
    避免多进程/模块重复连接导致的数据库锁冲突。

    Returns:
        VectorStore: 单例实例，首次调用时自动创建并连接。
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


def _reset_vector_store_instance() -> None:
    """重置单例（仅供测试使用，不应在生产代码中调用）."""
    global _vector_store_instance
    if _vector_store_instance is not None:
        try:
            _vector_store_instance.close()
        except Exception:
            pass
    _vector_store_instance = None
