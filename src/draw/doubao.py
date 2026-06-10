"""Draw 模块 —— 豆包（火山引擎）文生图适配器."""
from __future__ import annotations

import logging
import time

from src.config import (
    DOUBAO_API_KEY,
    DOUBAO_BASE_URL,
    DOUBAO_IMAGE_MODEL,
    DOUBAO_IMAGE_SIZE,
    IMAGE_DIR,
)
from src.draw.base import BaseDrawer
from src.draw.utils import post_image_generation, prompt_digest, save_image_bytes

logger = logging.getLogger(__name__)


class DoubaoDrawer(BaseDrawer):
    """豆包 / 火山引擎 文生图 API.

    API 文档: https://www.volcengine.com/docs/82379/1541523
    """

    model_name = "doubao"

    def __init__(self) -> None:
        self._api_key = DOUBAO_API_KEY
        self._base_url = DOUBAO_BASE_URL
        self._model = DOUBAO_IMAGE_MODEL
        self._size = DOUBAO_IMAGE_SIZE
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    async def generate(self, prompt: str) -> str:
        """调用豆包文生图 API，保存图片到 storage/images/."""
        t0 = time.perf_counter()
        logger.info(
            "draw.doubao.start | prompt_len=%d prompt_digest=%s model=%s",
            len(prompt),
            prompt_digest(prompt),
            self._model,
        )

        image_bytes = await post_image_generation(
            base_url=self._base_url,
            api_key=self._api_key,
            model=self._model,
            prompt=prompt,
            size=self._size,
            provider="doubao",
        )
        filepath = save_image_bytes(image_bytes, IMAGE_DIR, "doubao")

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "draw.doubao.end | duration_ms=%d image_path=%s image_bytes=%d",
            duration_ms,
            filepath,
            len(image_bytes),
        )
        return str(filepath)
