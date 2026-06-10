"""Draw 模块 —— 通义千问文生图适配器."""
from __future__ import annotations

import logging
import time

from src.config import (
    IMAGE_DIR,
    TONGYI_API_KEY,
    TONGYI_IMAGE_MODEL,
    TONGYI_IMAGE_SIZE,
)
from src.draw.base import BaseDrawer
from src.draw.utils import (
    post_dashscope_image_generation,
    prompt_digest,
    save_image_bytes,
)

logger = logging.getLogger(__name__)


class TongyiDrawer(BaseDrawer):
    """通义千问 / 百炼 文生图 API (原生异步接口).

    API 文档: https://help.aliyun.com/zh/model-studio/image-faq
    """

    model_name = "tongyi"

    def __init__(self) -> None:
        self._api_key = TONGYI_API_KEY
        self._model = TONGYI_IMAGE_MODEL
        self._size = TONGYI_IMAGE_SIZE
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    async def generate(self, prompt: str) -> str:
        """调用通义千问文生图 API，保存图片到 storage/images/."""
        t0 = time.perf_counter()
        logger.info(
            "draw.tongyi.start | prompt_len=%d prompt_digest=%s model=%s",
            len(prompt),
            prompt_digest(prompt),
            self._model,
        )

        image_bytes = await post_dashscope_image_generation(
            api_key=self._api_key,
            model=self._model,
            prompt=prompt,
            size=self._size,
        )
        filepath = save_image_bytes(image_bytes, IMAGE_DIR, "tongyi")

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "draw.tongyi.end | duration_ms=%d image_path=%s image_bytes=%d",
            duration_ms,
            filepath,
            len(image_bytes),
        )
        return str(filepath)
