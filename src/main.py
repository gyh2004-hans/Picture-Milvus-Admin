"""FastAPI 入口 —— 图像生成优化系统 API 服务.

启动: 在 picture2/ 目录内执行 uvicorn src.main:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import base64
import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.config import HOST, IMAGE_DIR, PICTURE2_PUBLIC_BASE_URL, PORT
from src.draw import DRAWER_REGISTRY, DoubaoDrawer, TongyiDrawer
from src.models.schemas import (
    AsyncPipelineResponse,
    DrawRecord,
    DrawRequest,
    DrawResponse,
    EvalRequest,
    EvalResult,
    FeedbackRequest,
    PipelineRequest,
    PipelineResponse,
    SearchRequest,
    SearchResponse,
)
from src.api.milvus_api import router as milvus_router
from src.api.search_api import router as search_router
from src.api.material_api import router as material_router
from src.api.websocket_mgr import router as ws_router
from src.pipeline import ImagePipeline
from src.storage import RecordStore

# ── 日志 ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── 生命周期 ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时注册 drawer 适配器."""
    logger.info("module.start | module=app lifespan=startup")
    DRAWER_REGISTRY["doubao"] = DoubaoDrawer()
    DRAWER_REGISTRY["tongyi"] = TongyiDrawer()
    logger.info("Drawers registered: %s", list(DRAWER_REGISTRY))
    yield
    logger.info("module.end | module=app lifespan=shutdown")


app = FastAPI(
    title="Picture Milvus Admin — AI 教学插图生成与向量检索平台",
    description=(
        "自然语言输入 → AI 生图 → VLM 评测 → Prompt 优化 → Milvus 向量检索 闭环"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# ── 单例 ─────────────────────────────────────────
pipeline = ImagePipeline()
record_store = RecordStore()

# 后台异步生图任务的强引用集合 —— 防止 asyncio.create_task 的 task 被 GC 回收
_async_draw_tasks: set[asyncio.Task] = set()


# ═══════════════════════════════════════════════════
# API 路由
# ═══════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok", "drawers": list(DRAWER_REGISTRY)}


# ── Draw（阶段 1 核心）──────────────────────────────

@app.post("/draw", response_model=DrawResponse)
async def draw(req: DrawRequest):
    """调用指定模型生图，并持久化 prompt / image 记录."""
    logger.info("draw.request | model=%s prompt_len=%d", req.model, len(req.prompt))

    drawer = DRAWER_REGISTRY.get(req.model)
    if drawer is None:
        raise HTTPException(400, f"Unknown model: {req.model}")

    try:
        image_path = await drawer.generate(req.prompt)
    except (ValueError, RuntimeError) as exc:
        logger.error("draw.error | model=%s error=%s", req.model, exc)
        raise HTTPException(502, str(exc)) from exc

    record = await asyncio.to_thread(
        record_store.create,
        prompt=req.prompt,
        model=req.model,
        image_path=image_path,
    )

    return DrawResponse(
        record_id=record.id,
        image_path=image_path,
        model=req.model,
        prompt=req.prompt,
    )


@app.get("/records", response_model=list[DrawRecord])
async def list_records(limit: int = 50):
    """列出最近的生图记录."""
    return await asyncio.to_thread(record_store.list_records, limit=limit)


@app.get("/records/{record_id}", response_model=DrawRecord)
async def get_record(record_id: str):
    record = await asyncio.to_thread(record_store.get, record_id)
    if record is None:
        raise HTTPException(404, f"Record not found: {record_id}")
    return record


@app.post("/feedback", response_model=DrawRecord)
async def submit_feedback(req: FeedbackRequest):
    """提交人工反馈（阶段 1：Prompt → 生图 → 人工反馈）."""
    try:
        return await asyncio.to_thread(
            record_store.add_feedback, req.record_id, req.feedback,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


# ── Evaluate ─────────────────────────────────────

@app.post("/evaluate", response_model=EvalResult)
async def evaluate(req: EvalRequest):
    """对生成图片执行 VLM 五维度评测（§3.2）."""
    from src.evaluate import VLMEvaluator

    evaluator = VLMEvaluator()
    return await evaluator.evaluate(req.prompt, req.image_path)


# ── Pipeline (闭环) ──────────────────────────────

def _encode_image_b64(image_path: str) -> str | None:
    """读取本地图片并 base64 编码，供响应直接回传给前端.

    读盘失败时记日志并返回 None（不影响 final_image_path 的回传）.
    """
    try:
        data = Path(image_path).read_bytes()
    except OSError as exc:
        logger.error("pipeline.encode.error | path=%s error=%s", image_path, exc)
        return None
    return base64.b64encode(data).decode("ascii")


@app.post("/pipeline", response_model=PipelineResponse)
async def run_pipeline(req: PipelineRequest):
    """执行完整闭环：生图 → 评测 → 修正 → 生图.

    响应同时回传本地路径 final_image_path 与 base64 编码 final_image_base64.
    """
    logger.info("pipeline.request | mode=clip_enrich model=%s prompt_len=%d",
                req.model, len(req.prompt))
    result = await pipeline.run(req)
    result.final_image_base64 = await asyncio.to_thread(
        _encode_image_b64, result.final_image_path,
    )
    logger.info(
        "pipeline.response | path=%s b64_len=%d score=%.3f",
        result.final_image_path,
        len(result.final_image_base64 or ""),
        result.final_score,
    )
    return result


# ── Pipeline (异步生图 + 图床) ───────────────────

async def _run_async_draw(task_id: str, req: PipelineRequest) -> None:
    """后台跑生图：走 clip_enrich 完整流程，产出图落到 IMAGE_DIR/{task_id}.png.

    成对日志: pipeline.async.start / pipeline.async.end / pipeline.async.error.
    异常被吞在本函数内（仅记日志）—— 失败则 {task_id}.png 不存在，对外表现为 URL 404.
    """
    prompt_summary = req.prompt[:80].replace("\n", " ")
    logger.info(
        "pipeline.async.start | task_id=%s model=%s prompt_len=%d prompt_summary=%s",
        task_id, req.model, len(req.prompt), prompt_summary,
    )
    t0 = time.perf_counter()
    try:
        result = await pipeline.run(req)

        dest = IMAGE_DIR / f"{task_id}.png"
        await asyncio.to_thread(IMAGE_DIR.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, result.final_image_path, dest)

        logger.info(
            "pipeline.async.end | task_id=%s src=%s dest=%s duration_ms=%d",
            task_id, result.final_image_path, dest,
            int((time.perf_counter() - t0) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 —— 后台任务须吞异常并归一化记日志
        logger.error(
            "pipeline.async.error | task_id=%s error_kind=%s error=%s duration_ms=%d",
            task_id, type(exc).__name__, exc,
            int((time.perf_counter() - t0) * 1000),
        )


@app.post("/pipeline/async", response_model=AsyncPipelineResponse, status_code=202)
async def run_pipeline_async(req: PipelineRequest) -> AsyncPipelineResponse:
    """提交即返回：建 task → 立刻回 {task_id, image_url}（HTTP 202），图后台生成.

    image_url 为预分配占位（此刻 {task_id}.png 还不存在，URL 暂 404）；
    后台任务以 direct 模式生图并落盘到 IMAGE_DIR/{task_id}.png.
    """
    task_id = uuid.uuid4().hex
    image_url = f"{PICTURE2_PUBLIC_BASE_URL}/images/{task_id}.png"

    logger.info(
        "pipeline.async.submit | task_id=%s model=%s prompt_len=%d image_url=%s",
        task_id, req.model, len(req.prompt), image_url,
    )

    task = asyncio.create_task(_run_async_draw(task_id, req))
    # 强引用 + done 回调 discard，避免 create_task 被 GC 回收
    _async_draw_tasks.add(task)
    task.add_done_callback(_async_draw_tasks.discard)

    return AsyncPipelineResponse(task_id=task_id, image_url=image_url)


@app.get("/images/{filename}")
async def get_image(filename: str) -> FileResponse:
    """图床：返回 IMAGE_DIR 下的图片，带路径穿越防护，生成中（文件缺失）→ 404."""
    image_root = IMAGE_DIR.resolve()
    target = (IMAGE_DIR / filename).resolve()

    # 路径穿越防护：resolve 后必须仍在 IMAGE_DIR 之内
    if target != image_root and image_root not in target.parents:
        logger.warning("images.traversal_blocked | filename=%s", filename)
        raise HTTPException(400, "Invalid filename")

    if not target.is_file():
        raise HTTPException(404, "Image not found (still generating?)")

    return FileResponse(target)


# ── Milvus 检索（v4: 使用 search_api 路由器） ──

app.include_router(milvus_router)
app.include_router(search_router)
app.include_router(material_router)
app.include_router(ws_router)

# 保留旧端点向后兼容
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """相似图检索（向后兼容端点）."""
    from src.milvus import get_vector_store

    store = get_vector_store()
    # pymilvus 无原生异步 API，连接与检索卸载到线程
    await asyncio.to_thread(store.connect)

    if req.prompt:
        return await asyncio.to_thread(
            store.search_by_text,
            text_embedding=None,
            text=req.prompt,
            top_k=req.top_k,
            subject=req.subject,
        )
    elif req.image_path:
        return await asyncio.to_thread(
            store.search_by_image,
            image_embedding=None,
            image_path=req.image_path,
            top_k=req.top_k,
            subject=req.subject,
        )
    else:
        raise HTTPException(400, "Either prompt or image_path is required")


# ── 启动 ─────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.main:app", host=HOST, port=PORT, reload=True)
