"""Draw 模块."""
from src.draw.base import BaseDrawer
from src.draw.doubao import DoubaoDrawer
from src.draw.tongyi import TongyiDrawer

# 模型名 → 适配器
DRAWER_REGISTRY: dict[str, BaseDrawer] = {}

__all__ = ["BaseDrawer", "DoubaoDrawer", "TongyiDrawer", "DRAWER_REGISTRY"]
