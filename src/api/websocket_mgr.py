"""WebSocket 连接管理器 —— 实时状态推送.

推送 Milvus 数据变更事件给前端，避免手动刷新.

事件类型:
  - entity_inserted: 新向量入库
  - entity_deleted: 删除记录
  - prompt_updated: prompt 编辑
  - partition_changed: 分区增删
  - collection_stats: 定时心跳(5s)

前端消费: 收到事件 → 更新 Zustand/Context store → UI 自动重渲染.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """WebSocket 连接管理器.

    管理所有已连接的客户端，支持广播推送.
    """

    def __init__(self) -> None:
        self._connections: dict[str, WebSocket] = {}
        self._heartbeat_task: asyncio.Task | None = None

    async def connect(self, websocket: WebSocket) -> str:
        """接受新的 WebSocket 连接.

        Returns:
            client_id: 分配的唯一客户端 ID.
        """
        await websocket.accept()
        client_id = f"ws_{int(time.time() * 1000)}_{len(self._connections)}"
        self._connections[client_id] = websocket
        logger.info("websocket.connect | client_id=%s total=%d", client_id, len(self._connections))
        return client_id

    def disconnect(self, client_id: str) -> None:
        """移除断开的连接."""
        self._connections.pop(client_id, None)
        logger.info("websocket.disconnect | client_id=%s remaining=%d", client_id, len(self._connections))

    async def broadcast(self, event: dict) -> None:
        """向所有连接的客户端广播事件."""
        disconnected: list[str] = []
        payload = json.dumps(event, ensure_ascii=False)
        for cid, ws in self._connections.items():
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(cid)
        # 清理断开的连接
        for cid in disconnected:
            self.disconnect(cid)

    async def send_to(self, client_id: str, event: dict) -> None:
        """向特定客户端发送事件."""
        ws = self._connections.get(client_id)
        if ws:
            payload = json.dumps(event, ensure_ascii=False)
            try:
                await ws.send_text(payload)
            except Exception:
                self.disconnect(client_id)

    @property
    def active_count(self) -> int:
        return len(self._connections)

    async def start_heartbeat(self, interval: float = 5.0) -> None:
        """启动心跳推送（后台任务）."""
        from src.milvus.vector_store import VectorStore, get_vector_store

        async def _beat():
            store = get_vector_store()
            store.connect()
            while True:
                await asyncio.sleep(interval)
                try:
                    if self.active_count > 0:
                        stats = store.get_stats_by_subject()
                        await self.broadcast({
                            "type": "collection_stats",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "payload": stats,
                        })
                except Exception as exc:
                    logger.warning("websocket.heartbeat.error | %s", exc)

        self._heartbeat_task = asyncio.create_task(_beat())
        logger.info("websocket.heartbeat.started | interval=%.1fs", interval)


# 全局实例
manager = ConnectionManager()


@router.websocket("/ws/milvus")
async def websocket_milvus(websocket: WebSocket):
    """Milvus WebSocket 端点 —— 实时状态同步.

    连接成功后自动接收心跳推送（每 5s 一次 collection_stats）.
    """
    client_id = await manager.connect(websocket)

    # 确保心跳任务已启动
    if manager._heartbeat_task is None:
        await manager.start_heartbeat(interval=5.0)

    try:
        # 发送欢迎消息
        await websocket.send_json({
            "type": "connected",
            "client_id": client_id,
            "active_connections": manager.active_count,
            "message": "Milvus WebSocket 连接成功",
        })

        # 保持连接，等待客户端消息
        while True:
            try:
                data = await websocket.receive_text()
                # 处理客户端消息（如 ping）
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                logger.debug("websocket.invalid_json | client_id=%s", client_id)
    finally:
        manager.disconnect(client_id)


# ── 便捷函数：供其他模块调用推送事件 ──────────────

async def notify_entity_inserted(subject: str, image_id: int, score: float, count: int) -> None:
    """通知前端: 新向量已入库."""
    await manager.broadcast({
        "type": "entity_inserted",
        "payload": {
            "subject": subject,
            "image_id": image_id,
            "score": score,
            "count": count,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


async def notify_entity_deleted(image_id: int, subject: str) -> None:
    """通知前端: 记录已删除."""
    await manager.broadcast({
        "type": "entity_deleted",
        "payload": {"image_id": image_id, "subject": subject},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


async def notify_prompt_updated(image_id: int, new_prompt: str) -> None:
    """通知前端: prompt 已编辑."""
    await manager.broadcast({
        "type": "prompt_updated",
        "payload": {"image_id": image_id, "new_prompt": new_prompt},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


async def notify_partition_changed(action: str, partition_name: str, row_count: int = 0) -> None:
    """通知前端: 分区增删."""
    await manager.broadcast({
        "type": "partition_changed",
        "payload": {"action": action, "partition_name": partition_name, "row_count": row_count},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
