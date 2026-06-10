"""FastAPI 路由器包 —— Attu 风格 Milvus 管理平台后端.

使用方式（在 main.py 中）:
    from src.api import milvus_router, search_router, ws_router
    app.include_router(milvus_router)
    app.include_router(search_router)
    app.include_router(ws_router)
"""
from src.api.milvus_api import router as milvus_router
from src.api.search_api import router as search_router
from src.api.websocket_mgr import router as ws_router

__all__ = [
    "milvus_router",
    "search_router",
    "ws_router",
]
