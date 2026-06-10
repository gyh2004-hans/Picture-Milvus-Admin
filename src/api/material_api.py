"""图片素材上传 API（v6 泛化）.

POST /api/upload_material
  上传图片 → VLM 内容解析 → Chinese-CLIP 双向量编码 → Milvus 入库

流程:
  1. 接收上传图片，保存到 IMAGE_DIR
  2. VLM 图片内容解析 (image_content_parser)
  3. 构建 semantic_text
  4. Chinese-CLIP: image_embedding + semantic_embedding
  5. Milvus insert（含完整语义字段）
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.config import IMAGE_DIR
from src.milvus.image_content_parser import ImageContentParser
from src.milvus.vector_store import VectorStore, get_vector_store
from src.models.schemas import (
    ImageContentParseResult,
    ImageRecord,
    MaterialUploadResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["material"])

# 上传文件大小上限: 10MB
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# 单例
_store: VectorStore | None = None
_parser: ImageContentParser | None = None
_clip_client: object | None = None


def _get_store() -> VectorStore:
    global _store
    if _store is None or not _store.ready:
        _store = get_vector_store()
        _store.connect()
    return _store


def _get_parser() -> ImageContentParser:
    global _parser
    if _parser is None:
        _parser = ImageContentParser()
    return _parser


def _get_clip():
    """懒加载 CLIP 客户端."""
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


# ── API 端点 ──────────────────────────────────────────


@router.post("/upload_material", response_model=MaterialUploadResponse)
async def upload_material(file: UploadFile = File(...)):
    """上传图片素材并入库.

    接收图片（PNG/JPG/WebP），自动:
      1. 保存到服务器 IMAGE_DIR
      2. VLM 图片内容解析 → ImageContentParseResult
      3. 构建 semantic_text
      4. Chinese-CLIP 双向量编码
      5. Milvus 入库

    Returns:
        MaterialUploadResponse: 含 record_id / parse_result / semantic_text.
    """
    t0 = time.perf_counter()
    logger.info(
        "material_api.upload.start | filename=%s content_type=%s",
        file.filename, file.content_type,
    )

    # ── 1. 校验文件 ──
    if not file.filename:
        raise HTTPException(400, "文件名不能为空")

    suffix = Path(file.filename).suffix.lower()
    allowed = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
    if suffix not in allowed:
        raise HTTPException(400, f"不支持的图片格式: {suffix}，支持: {', '.join(allowed)}")

    content = await file.read()
    if len(content) > _MAX_UPLOAD_BYTES:
        raise HTTPException(400, f"文件过大: {len(content)} bytes，上限 {_MAX_UPLOAD_BYTES} bytes")

    # ── 2. 保存图片 ──
    await asyncio.to_thread(IMAGE_DIR.mkdir, parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = f"upload_{timestamp}_{file.filename}"
    dest_path = IMAGE_DIR / safe_name
    await asyncio.to_thread(dest_path.write_bytes, content)
    logger.info("material_api.saved | path=%s size=%d", dest_path, len(content))

    # ── 3. VLM 图片内容解析 ──
    parser = _get_parser()
    try:
        parse_result = await parser.parse(str(dest_path))
    except (RuntimeError, FileNotFoundError) as exc:
        logger.error("material_api.parse_error | path=%s error=%s", dest_path, exc)
        # 解析失败不删图，保留文件供手动检查
        raise HTTPException(502, f"VLM 图片内容解析失败: {exc}") from exc

    # ── 4. 构建 semantic_text ──
    semantic_text = ImageContentParser.build_semantic_text(parse_result)

    # ── 5. Chinese-CLIP 双向量编码 ──
    clip = _get_clip()
    try:
        image_embedding = await asyncio.to_thread(clip.encode_image, str(dest_path))
        semantic_embedding = await asyncio.to_thread(clip.encode_text, semantic_text)
    except Exception as exc:
        logger.error("material_api.clip_error | path=%s error=%s", dest_path, exc)
        raise HTTPException(500, f"CLIP 向量编码失败: {exc}") from exc

    # ── 6. 构建 ImageRecord ──
    category = parse_result.category or "其他"

    record = ImageRecord(
        image_id=0,
        prompt=parse_result.retrieval_prompt,
        optimized_prompt=parse_result.retrieval_prompt,
        score=0.0,  # 上传素材无评测分
        image_path=str(dest_path),
        embedding=image_embedding.tolist(),
        category=category,
        subject=category,  # 向后兼容旧字段
        tags=parse_result.tags[:8],
        # 语义字段
        semantic_text=semantic_text,
        semantic_embedding=semantic_embedding.tolist(),
        topic=parse_result.scene_description[:100] if parse_result.scene_description else "",
        content_type=parse_result.content_type,
        diagram_type=parse_result.content_type,  # 向后兼容旧字段
        visual_elements=parse_result.main_objects,
        main_objects=parse_result.main_objects,
        scene_description=parse_result.scene_description,
        style=parse_result.style,
        color_palette=parse_result.color_palette,
        keywords=parse_result.tags,
        knowledge_points=parse_result.tags[:6],  # 向后兼容
        source_type="uploaded",
    )

    # ── 7. Milvus 入库 ──
    store = _get_store()
    try:
        record_id = await asyncio.to_thread(store.insert, record, subject=category)
    except Exception as exc:
        logger.error("material_api.insert_error | path=%s error=%s", dest_path, exc)
        raise HTTPException(500, f"Milvus 入库失败: {exc}") from exc

    duration_ms = int((time.perf_counter() - t0) * 1000)
    logger.info(
        "material_api.upload.end | record_id=%d category=%s content_type=%s "
        "objects=%d tags=%d duration_ms=%d",
        record_id, parse_result.category, parse_result.content_type,
        len(parse_result.main_objects), len(parse_result.tags), duration_ms,
    )

    return MaterialUploadResponse(
        record_id=record_id,
        image_path=str(dest_path),
        parse_result=parse_result,
        semantic_text=semantic_text,
    )
