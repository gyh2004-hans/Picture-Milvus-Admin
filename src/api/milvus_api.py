"""Milvus 管理 API —— Collection / Partition / Index CRUD.

提供 Attu 风格管理平台所需的后端接口:
  - Collection: 列表 / 详情 / 创建 / 删除
  - Partition: 列表 / 创建 / 删除 / 统计
  - Index: 列表 / 创建 / 删除 / 切换
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.milvus.vector_store import SUBJECT_PARTITION_MAP, VectorStore, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/milvus", tags=["milvus"])

# ── 全局单例 VectorStore ──
_store: VectorStore | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None or not _store.ready:
        _store = get_vector_store()
        _store.connect()
    return _store


# ── 请求/响应模型 ─────────────────────────────────

class CollectionInfo(BaseModel):
    name: str
    entity_count: int
    status: str = "Loaded"
    created_at: str = ""
    schema_fields: list[str] = Field(default_factory=list)
    index_info: dict = Field(default_factory=dict)
    backend_type: str = "unknown"  # milvus_lite | milvus_server | local_numpy


class CreateCollectionRequest(BaseModel):
    name: str
    dimension: int = Field(default=512, ge=1, le=4096)
    metric_type: str = Field(default="COSINE")


class PartitionInfo(BaseModel):
    name: str
    row_count: int
    created_at: str = ""


class CreatePartitionRequest(BaseModel):
    name: str
    parent_partition: Optional[str] = None  # 预留树形分区


class IndexInfo(BaseModel):
    field_name: str
    index_type: str
    metric_type: str
    status: str = "Ready"


class CreateIndexRequest(BaseModel):
    index_type: str = Field(default="HNSW", description="FLAT / HNSW / IVF_FLAT / IVF_SQ8")
    metric_type: str = Field(default="COSINE")
    extra_params: Optional[dict] = None


# ── Collection 端点 ───────────────────────────────

@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections():
    """列出所有 Collection."""
    store = _get_store()
    # 当前只有一个 collection: image_embeddings
    stats = store.get_stats_by_subject()
    return [
        CollectionInfo(
            name="image_embeddings",
            entity_count=stats.get("total", store.count()),
            status="Loaded" if store.ready else "NotLoaded",
            schema_fields=[
                "vector (FLOAT_VECTOR)", "prompt (VARCHAR)", "optimized_prompt (VARCHAR)",
                "score (FLOAT)", "image_path (VARCHAR)", "created_at (VARCHAR)",
                "model_version (VARCHAR)", "subject (VARCHAR)", "category (VARCHAR)",
                "tags (VARCHAR)",
            ],
            index_info=store.get_index_info(),
            backend_type=store.backend_type,
        )
    ]


@router.get("/collections/{name}", response_model=CollectionInfo)
async def get_collection(name: str):
    """获取 Collection 详情."""
    store = _get_store()
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    stats = store.get_stats_by_subject()
    return CollectionInfo(
        name=name,
        entity_count=stats.get("total", store.count()),
        status="Loaded" if store.ready else "NotLoaded",
        schema_fields=[
            "vector (FLOAT_VECTOR)", "prompt (VARCHAR)", "optimized_prompt (VARCHAR)",
            "score (FLOAT)", "image_path (VARCHAR)", "created_at (VARCHAR)",
            "model_version (VARCHAR)", "subject (VARCHAR)", "category (VARCHAR)",
            "tags (VARCHAR)",
        ],
        index_info=store.get_index_info(),
        backend_type=store.backend_type,
    )


@router.delete("/collections/{name}")
async def delete_collection(name: str):
    """删除整个 Collection（危险操作）."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    store.drop_all()
    return {"status": "deleted", "name": name}


# ── Partition 端点 ────────────────────────────────

@router.get("/collections/{name}/partitions", response_model=list[PartitionInfo])
async def list_partitions(name: str):
    """列出所有分区."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    partitions = store.list_partitions()
    return [
        PartitionInfo(name=p["name"], row_count=p["row_count"])
        for p in partitions
    ]


@router.post("/collections/{name}/partitions", status_code=201)
async def create_partition(name: str, req: CreatePartitionRequest):
    """创建新分区."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    store._ensure_partition(req.name)
    return {"status": "created", "name": req.name}


@router.delete("/collections/{name}/partitions/{partition_name}")
async def delete_partition(name: str, partition_name: str):
    """删除分区（仅空分区可删）."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    try:
        cnt = store.count()
        # 通过 backend 直接操作
        store._backend.drop_partition(partition_name)
        logger.info("milvus_api.delete_partition | name=%s", partition_name)
        return {"status": "deleted", "name": partition_name}
    except Exception as exc:
        raise HTTPException(400, f"Failed to drop partition: {exc}")


@router.get("/collections/{name}/stats")
async def get_collection_stats(name: str):
    """获取各学科分区数据量统计."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    return store.get_stats_by_subject()


# ── Index 端点 ────────────────────────────────────

@router.get("/collections/{name}/indexes", response_model=list[IndexInfo])
async def list_indexes(name: str):
    """列出索引."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    info = store.get_index_info()
    if not info:
        return []
    return [
        IndexInfo(
            field_name=info.get("field_name", "vector"),
            index_type=info.get("index_type", "FLAT"),
            metric_type=info.get("metric_type", "COSINE"),
            status="Ready",
        )
    ]


@router.post("/collections/{name}/indexes", status_code=201)
async def create_index(name: str, req: CreateIndexRequest):
    """创建/切换索引."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    ok = store.create_index(
        index_type=req.index_type,
        metric_type=req.metric_type,
        extra_params=req.extra_params,
    )
    if not ok:
        raise HTTPException(500, "Failed to create index")
    return {"status": "created", "index_type": req.index_type, "metric_type": req.metric_type}


@router.delete("/collections/{name}/indexes")
async def drop_index(name: str):
    """删除索引."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    ok = store.drop_index()
    if not ok:
        raise HTTPException(500, "Failed to drop index")
    return {"status": "deleted"}


# ── 数据 CRUD 端点 ────────────────────────────────

class EntityRecord(BaseModel):
    id: int
    prompt: str
    optimized_prompt: Optional[str] = None
    score: float
    image_path: str
    subject: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    model_version: Optional[str] = None


class UpdateEntityRequest(BaseModel):
    prompt: Optional[str] = None
    optimized_prompt: Optional[str] = None
    score: Optional[float] = None
    subject: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[list[str]] = None


@router.get("/collections/{name}/data", response_model=dict)
async def list_data(
    name: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    subject: Optional[str] = None,
    category: Optional[str] = None,
    min_score: Optional[float] = None,
):
    """分页列出数据（支持按分类/学科和最低分过滤）.

    v6: 优先使用 category 参数（新分类体系），向后兼容 subject 参数（旧学科体系）.
    """
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()

    # v6: category 优先于 subject
    partition_filter = category or subject
    records, total = store.list_data(
        limit=limit, offset=offset,
        subject=partition_filter, min_score=min_score,
    )
    return {
        "data": records,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.delete("/collections/{name}/data/{entity_id}")
async def delete_entity(name: str, entity_id: int):
    """删除单条记录."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    ok = store.delete_entity(entity_id)
    if not ok:
        raise HTTPException(404, f"Entity not found: {entity_id}")
    return {"status": "deleted", "id": entity_id}


@router.get("/status")
async def milvus_status():
    """获取 Milvus 连接状态和统计信息（调试用）."""
    store = _get_store()
    stats = store.get_stats_by_subject()
    return {
        "backend_type": store.backend_type,
        "ready": store.ready,
        "total_entities": stats.get("total", store.count()),
        "by_subject": {k: v for k, v in stats.items() if k != "total"},
        "partitions": store.list_partitions(),
        "index_info": store.get_index_info(),
        "warning": (
            "⚠️ 当前使用 LocalNumpyBackend（内存存储），进程重启后数据丢失！"
            if store.backend_type == "local_numpy" else None
        ),
    }


@router.put("/collections/{name}/data/{entity_id}")
async def update_entity(name: str, entity_id: int, req: UpdateEntityRequest):
    """更新记录元数据（prompt / score / subject 等）."""
    if name != "image_embeddings":
        raise HTTPException(404, f"Collection not found: {name}")
    store = _get_store()
    update_data = {k: v for k, v in req.model_dump().items() if v is not None}
    if not update_data:
        raise HTTPException(400, "No fields to update")
    ok = store.update_entity(entity_id, update_data)
    if not ok:
        raise HTTPException(404, f"Entity not found: {entity_id}")
    return {"status": "updated", "id": entity_id}
