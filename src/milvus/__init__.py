"""Milvus 检索模块 (v4).

v4 新增:
  - 学科分区隔离检索
  - 分区管理 API
  - CRUD 操作
  - 索引管理
"""
from src.milvus.vector_store import (
    DEFAULT_PARTITION,
    SUBJECT_PARTITION_MAP,
    VectorStore,
    _resolve_partition,
    get_vector_store,
)

__all__ = [
    "VectorStore",
    "SUBJECT_PARTITION_MAP",
    "DEFAULT_PARTITION",
    "_resolve_partition",
    "get_vector_store",
]
