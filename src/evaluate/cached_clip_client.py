"""CachedCLIPClient —— 带 LRU 缓存的 Chinese-CLIP 客户端包装器.

对同一图片/Prompt 的重复编码直接返回缓存结果，避免浪费 GPU/CPU 算力。

策略:
  - 基于 content hash（SHA256）的 LRU 缓存
  - 图片: 读取文件二进制 → SHA256 hash → 查缓存
  - 文本: SHA256(text.encode()) → 查缓存
  - 默认缓存 1024 条 embedding，超出时 LRU 淘汰

使用方式:
    from src.evaluate.local_clip_client import LocalCLIPClient
    from src.evaluate.cached_clip_client import CachedCLIPClient

    base = LocalCLIPClient(model_name="OFA-Sys/chinese-clip-vit-base-patch16")
    client = CachedCLIPClient(base, cache_size=1024)
    emb = client.encode_text("一幅地理教学插图")  # 第一次 → 编码
    emb = client.encode_text("一幅地理教学插图")  # 第二次 → 缓存命中
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CachedCLIPClient:
    """带 LRU 缓存的 Chinese-CLIP 客户端包装器.

    对 encode_text / encode_image / encode_texts / encode_image_patches
    均提供基于 content hash 的缓存。
    """

    def __init__(
        self,
        clip_client: object,
        cache_size: int = 1024,
    ) -> None:
        """初始化缓存客户端.

        Args:
            clip_client: LocalCLIPClient 实例.
            cache_size: 最大缓存条目数（默认 1024）.
        """
        self._client = clip_client
        self._cache_size = cache_size
        self._text_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._patch_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        # 统计
        self._hits = 0
        self._misses = 0

    # ── 公开方法 ──────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """返回底层模型的 embedding 维度."""
        return self._client.embedding_dim  # type: ignore[attr-defined]

    @property
    def cache_stats(self) -> dict:
        """返回缓存统计信息."""
        return {
            "text_entries": len(self._text_cache),
            "image_entries": len(self._image_cache),
            "patch_entries": len(self._patch_cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": (self._hits / max(self._hits + self._misses, 1)),
        }

    def encode_text(self, text: str) -> np.ndarray:
        """将文本编码为归一化 CLIP embedding（带缓存）."""
        h = self._hash_text(text)
        if h in self._text_cache:
            self._hits += 1
            self._text_cache.move_to_end(h)
            logger.debug("CachedCLIP.encode_text.cache_hit | text_hash=%s", h[:16])
            return self._text_cache[h].copy()

        self._misses += 1
        t0 = time.perf_counter()
        result = self._client.encode_text(text)  # type: ignore[attr-defined]
        duration_ms = int((time.perf_counter() - t0) * 1000)

        self._text_cache[h] = result.copy()
        self._evict_if_needed(self._text_cache)

        logger.info(
            "CachedCLIP.encode_text.miss | text_len=%d hash=%s duration_ms=%d cache_size=%d",
            len(text), h[:16], duration_ms, len(self._text_cache),
        )
        return result

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """批量编码文本（逐条查缓存，未命中统一编码）."""
        n = len(texts)
        hashes = [self._hash_text(t) for t in texts]
        result = np.zeros((n, self.embedding_dim), dtype=np.float32)

        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, (t, h) in enumerate(zip(texts, hashes)):
            if h in self._text_cache:
                self._hits += 1
                self._text_cache.move_to_end(h)
                result[i] = self._text_cache[h]
            else:
                self._misses += 1
                uncached_indices.append(i)
                uncached_texts.append(t)

        if uncached_texts:
            t0 = time.perf_counter()
            encoded = self._client.encode_texts(uncached_texts)  # type: ignore[attr-defined]
            duration_ms = int((time.perf_counter() - t0) * 1000)

            for idx, h, emb in zip(uncached_indices,
                                   [hashes[i] for i in uncached_indices],
                                   encoded):
                result[idx] = emb
                self._text_cache[h] = emb.copy()

            self._evict_if_needed(self._text_cache)
            logger.info(
                "CachedCLIP.encode_texts | total=%d cached=%d uncached=%d duration_ms=%d",
                n, n - len(uncached_texts), len(uncached_texts), duration_ms,
            )
        else:
            logger.info("CachedCLIP.encode_texts.cache_all_hit | n=%d", n)

        return result

    def encode_image(self, image_path: str) -> np.ndarray:
        """将图片编码为归一化 CLIP embedding（带缓存）."""
        h = self._hash_file(image_path)
        if h is not None and h in self._image_cache:
            self._hits += 1
            self._image_cache.move_to_end(h)
            logger.debug("CachedCLIP.encode_image.cache_hit | image=%s", image_path)
            return self._image_cache[h].copy()

        self._misses += 1
        t0 = time.perf_counter()
        result = self._client.encode_image(image_path)  # type: ignore[attr-defined]
        duration_ms = int((time.perf_counter() - t0) * 1000)

        if h is not None:
            self._image_cache[h] = result.copy()
            self._evict_if_needed(self._image_cache)

        logger.info(
            "CachedCLIP.encode_image.miss | image=%s duration_ms=%d cache_size=%d",
            image_path, duration_ms, len(self._image_cache),
        )
        return result

    def encode_image_patches(self, image_path: str) -> np.ndarray:
        """提取图像 patch embeddings（带缓存）."""
        h = self._hash_file(image_path, suffix="_patches")
        if h is not None and h in self._patch_cache:
            self._hits += 1
            self._patch_cache.move_to_end(h)
            logger.debug("CachedCLIP.encode_image_patches.cache_hit | image=%s", image_path)
            return self._patch_cache[h].copy()

        self._misses += 1
        t0 = time.perf_counter()
        result = self._client.encode_image_patches(image_path)  # type: ignore[attr-defined]
        duration_ms = int((time.perf_counter() - t0) * 1000)

        if h is not None:
            self._patch_cache[h] = result.copy()
            self._evict_if_needed(self._patch_cache)

        logger.info(
            "CachedCLIP.encode_image_patches.miss | image=%s duration_ms=%d cache_size=%d",
            image_path, duration_ms, len(self._patch_cache),
        )
        return result

    def close(self) -> None:
        """释放资源."""
        self._text_cache.clear()
        self._image_cache.clear()
        self._patch_cache.clear()
        self._client.close()  # type: ignore[attr-defined]
        logger.info(
            "CachedCLIP.close | stats=%s",
            self.cache_stats,
        )

    def clear_cache(self) -> None:
        """清空所有缓存."""
        self._text_cache.clear()
        self._image_cache.clear()
        self._patch_cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("CachedCLIP.clear_cache | cache cleared")

    # ── 内部方法 ──────────────────────────────────────

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _hash_file(file_path: str, suffix: str = "") -> str | None:
        """读取文件二进制并计算 SHA256 hash."""
        try:
            data = Path(file_path).read_bytes()
            if suffix:
                data += suffix.encode("utf-8")
            return hashlib.sha256(data).hexdigest()
        except (FileNotFoundError, OSError) as exc:
            logger.warning("CachedCLIP._hash_file | path=%s error=%s", file_path, exc)
            return None

    def _evict_if_needed(self, cache: OrderedDict) -> None:
        """LRU 淘汰：超过 cache_size 时淘汰最久未使用条目."""
        while len(cache) > self._cache_size:
            evicted_key, _ = cache.popitem(last=False)
            logger.debug("CachedCLIP.evict | key=%s", evicted_key[:16])
