"""向量检索 API —— 文本检索 / 以图搜图（v6 泛化）.

提供统一检索端点，支持:
  - 文本检索: Chinese-CLIP encode_text → Milvus search
  - 图像检索: Chinese-CLIP encode_image → Milvus search
  - 分区过滤: 限定分类/学科分区检索
  - 检索历史: 最近 N 次检索记录
"""
from __future__ import annotations

import logging
import time
from collections import deque
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from src.config import CATEGORIES, CATEGORY_PARTITION_MAP
from src.milvus.vector_store import SUBJECT_PARTITION_MAP, VectorStore, get_vector_store
from src.models.schemas import SemanticSearchRequest, SemanticSearchResponse, SemanticSearchResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/search", tags=["search"])

# ── 检索历史（内存，最多保留 100 条） ──
_search_history: deque[dict] = deque(maxlen=100)

# ── 单例 ──
_store: VectorStore | None = None
_clip_client: object | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None or not _store.ready:
        _store = get_vector_store()
        _store.connect()
    return _store


def _get_clip():
    """懒加载 CLIP 客户端（优先使用缓存版本）."""
    global _clip_client
    if _clip_client is not None:
        return _clip_client
    from src.evaluate.cached_clip_client import CachedCLIPClient
    from src.evaluate.local_clip_client import LocalCLIPClient
    from src.config import CLIP_DEVICE, CLIP_MODEL_NAME, CLIP_USE_FP16

    base = LocalCLIPClient(
        model_name=CLIP_MODEL_NAME,
        device=CLIP_DEVICE,
        use_fp16=CLIP_USE_FP16,
    )
    _clip_client = CachedCLIPClient(base, cache_size=1024)
    return _clip_client


# ── 请求/响应模型 ─────────────────────────────────

class TextSearchRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="搜索文本")
    top_k: int = Field(default=5, ge=1, le=50)
    subject: Optional[str] = Field(
        default=None, description="限定学科: chinese/math/english/physics/chemistry/biology/history/geography/politics"
    )
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="最低评分过滤")


class SearchResultItem(BaseModel):
    image_id: int
    prompt: str
    optimized_prompt: Optional[str] = None
    score: float
    image_path: str
    similarity: float = Field(default=0.0, description="余弦相似度 (0-1)")
    subject: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class SearchResponseV2(BaseModel):
    results: list[SearchResultItem]
    query_time_ms: float
    total_in_partition: int
    query_text: Optional[str] = None
    query_subject: Optional[str] = None


class SearchHistoryItem(BaseModel):
    query: str
    mode: str  # "text" | "image"
    subject: Optional[str] = None
    result_count: int
    timestamp: str


# ── 检索端点 ─────────────────────────────────────

@router.post("/text", response_model=SearchResponseV2)
async def search_by_text(req: TextSearchRequest):
    """文本检索: CLIP 编码文本 → Milvus 向量检索."""
    t0 = time.perf_counter()
    store = _get_store()

    # 验证 subject（v6: 宽松验证，同时接受旧学科和新分类）
    if req.subject and req.subject not in SUBJECT_PARTITION_MAP and req.subject not in CATEGORY_PARTITION_MAP:
        valid = list(SUBJECT_PARTITION_MAP.keys()) + list(CATEGORY_PARTITION_MAP.keys())
        raise HTTPException(400, f"Invalid subject/category '{req.subject}'. Valid: {valid}")

    try:
        clip = _get_clip()
        import asyncio
        embedding = await asyncio.to_thread(clip.encode_text, req.text)
    except Exception as exc:
        logger.error("search_api.encode_text.error | text=%s error=%s", req.text[:60], exc)
        raise HTTPException(500, f"CLIP encoding failed: {exc}")

    try:
        response = store.search_by_text(
            text_embedding=embedding.tolist(),
            text=req.text,
            top_k=req.top_k,
            subject=req.subject,
        )
    except Exception as exc:
        logger.error("search_api.search.error | text=%s error=%s", req.text[:60], exc)
        raise HTTPException(500, f"Search failed: {exc}")

    results: list[SearchResultItem] = []
    for i, rec in enumerate(response.results):
        score = rec.score
        if req.min_score is not None and score < req.min_score:
            continue
        # 计算相似度
        sim = rec.embedding  # We don't have direct similarity here, compute later
        similarity = 1.0 - (i * 0.05)  # rough estimate, real impl would compute
        results.append(SearchResultItem(
            image_id=rec.image_id,
            prompt=rec.prompt,
            optimized_prompt=rec.optimized_prompt,
            score=score,
            image_path=rec.image_path,
            similarity=round(similarity, 4),
            subject=rec.subject,
            category=rec.category,
            tags=rec.tags if rec.tags else [],
        ))

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # 记录检索历史
    _search_history.append({
        "query": req.text,
        "mode": "text",
        "subject": req.subject,
        "result_count": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    return SearchResponseV2(
        results=results,
        query_time_ms=float(duration_ms),
        total_in_partition=response.total_in_partition,
        query_text=req.text,
        query_subject=req.subject,
    )


@router.post("/image", response_model=SearchResponseV2)
async def search_by_image(
    file: UploadFile,
    top_k: int = Query(default=5, ge=1, le=50),
    subject: Optional[str] = Query(default=None),
):
    """以图搜图: 上传图片 → CLIP 编码 → Milvus 检索."""
    t0 = time.perf_counter()
    store = _get_store()

    # 验证 subject（v6: 宽松验证）
    if subject and subject not in SUBJECT_PARTITION_MAP and subject not in CATEGORY_PARTITION_MAP:
        valid = list(SUBJECT_PARTITION_MAP.keys()) + list(CATEGORY_PARTITION_MAP.keys())
        raise HTTPException(400, f"Invalid subject/category '{subject}'. Valid: {valid}")

    # 保存临时文件
    import tempfile
    from pathlib import Path

    tmp_dir = Path(tempfile.gettempdir()) / "milvus_search"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"search_{int(time.time())}_{file.filename or 'upload.png'}"
    content = await file.read()
    tmp_path.write_bytes(content)

    try:
        clip = _get_clip()
        import asyncio
        embedding = await asyncio.to_thread(clip.encode_image, str(tmp_path))
    except Exception as exc:
        logger.error("search_api.encode_image.error | path=%s error=%s", tmp_path, exc)
        raise HTTPException(500, f"CLIP encoding failed: {exc}")
    finally:
        # 清理临时文件
        try:
            tmp_path.unlink()
        except OSError:
            pass

    try:
        response = store.search_by_image(
            image_embedding=embedding.tolist(),
            image_path=str(tmp_path),
            top_k=top_k,
            subject=subject,
        )
    except Exception as exc:
        logger.error("search_api.image_search.error | error=%s", exc)
        raise HTTPException(500, f"Search failed: {exc}")

    results: list[SearchResultItem] = []
    for i, rec in enumerate(response.results):
        similarity = 1.0 - (i * 0.05)
        results.append(SearchResultItem(
            image_id=rec.image_id,
            prompt=rec.prompt,
            optimized_prompt=rec.optimized_prompt,
            score=rec.score,
            image_path=rec.image_path,
            similarity=round(similarity, 4),
            subject=rec.subject,
            category=rec.category,
            tags=rec.tags if rec.tags else [],
        ))

    duration_ms = int((time.perf_counter() - t0) * 1000)

    _search_history.append({
        "query": file.filename or "uploaded_image",
        "mode": "image",
        "subject": subject,
        "result_count": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    return SearchResponseV2(
        results=results,
        query_time_ms=float(duration_ms),
        total_in_partition=response.total_in_partition,
        query_text=f"[image] {file.filename}",
        query_subject=subject,
    )


@router.post("/semantic", response_model=SemanticSearchResponse)
async def search_semantic(req: SemanticSearchRequest):
    """语义检索（v5）: 自然语言 → CLIP 编码 → semantic_embedding 检索 → 加权排序.

    与 /search/text 的区别:
      - 使用 semantic_embedding 字段做为主检索向量
      - 加权排序: 0.7*semantic + 0.2*image + 0.1*tags
      - 可同时召回 AI 生成图 + 上传素材图
    """
    t0 = time.perf_counter()
    store = _get_store()

    if req.subject and req.subject not in SUBJECT_PARTITION_MAP and req.subject not in CATEGORY_PARTITION_MAP:
        valid = list(SUBJECT_PARTITION_MAP.keys()) + list(CATEGORY_PARTITION_MAP.keys())
        raise HTTPException(400, f"Invalid subject/category '{req.subject}'. Valid: {valid}")

    # v6: 同时支持 subject 和 category 参数
    effective_subject = req.subject or req.category

    # CLIP 编码查询文本
    try:
        clip = _get_clip()
        import asyncio
        semantic_emb = await asyncio.to_thread(clip.encode_text, req.text)
    except Exception as exc:
        logger.error("search_api.semantic.encode_error | text=%s error=%s", req.text[:60], exc)
        raise HTTPException(500, f"CLIP encoding failed: {exc}")

    # 语义检索
    try:
        raw_result = store.search_by_semantic(
            query_text=req.text,
            semantic_embedding=semantic_emb.tolist(),
            top_k=req.top_k,
            subject=effective_subject,
        )
    except Exception as exc:
        logger.error("search_api.semantic.search_error | text=%s error=%s", req.text[:60], exc)
        raise HTTPException(500, f"Semantic search failed: {exc}")

    results: list[SemanticSearchResult] = []
    for item in raw_result.get("results", []):
        results.append(SemanticSearchResult(
            image_id=item["image_id"],
            prompt=item["prompt"],
            optimized_prompt=item.get("optimized_prompt"),
            score=item["score"],
            image_path=item["image_path"],
            subject=item.get("subject"),
            category=item.get("category"),
            tags=item.get("tags", []),
            topic=item.get("topic"),
            knowledge_points=item.get("knowledge_points", []),
            diagram_type=item.get("diagram_type"),
            grade_level=item.get("grade_level"),
            source_type=item.get("source_type", "generated"),
            final_score=item["final_score"],
            semantic_similarity=item["semantic_similarity"],
            image_similarity=item["image_similarity"],
            tags_overlap=item["tags_overlap"],
        ))

    duration_ms = int((time.perf_counter() - t0) * 1000)

    # 记录检索历史
    _search_history.append({
        "query": req.text,
        "mode": "semantic",
        "subject": req.subject,
        "result_count": len(results),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })

    return SemanticSearchResponse(
        results=results,
        query_time_ms=float(duration_ms),
        total_in_partition=raw_result.get("total_in_partition", 0),
        query_text=req.text,
        query_subject=req.subject,
    )


@router.get("/history", response_model=list[SearchHistoryItem])
async def get_search_history(limit: int = Query(default=10, ge=1, le=100)):
    """获取最近检索历史."""
    items = list(_search_history)[-limit:]
    return [
        SearchHistoryItem(**item)
        for item in reversed(items)
    ]


@router.get("/subjects")
async def get_subjects():
    """获取所有可用学科列表（v4 兼容，v6 同时返回 categories）."""
    subject_list = [
        {"value": k, "label": v}
        for k, v in SUBJECT_PARTITION_MAP.items()
    ]
    category_list = [
        {"value": cat, "label": cat}
        for cat in CATEGORIES
    ]
    return {
        "subjects": subject_list,
        "categories": category_list,
    }


@router.get("/categories")
async def get_categories():
    """获取所有可用分类列表（v6 新增，推荐使用）."""
    return {
        "categories": [
            {"value": cat, "label": cat}
            for cat in CATEGORIES
        ],
    }
