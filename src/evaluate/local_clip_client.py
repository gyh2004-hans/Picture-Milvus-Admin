"""LocalCLIPClient —— 本地 Chinese-CLIP 推理客户端.

使用 HuggingFace transformers 加载 OFA-Sys/chinese-clip-vit-large-patch14 模型，
替代原有的 SigLIP SO400M。Chinese-CLIP 针对中英文双语场景训练，
对中文教学插图的文本-图像对齐效果更好。

接口与 CLIPScorer 的调用约定兼容：
  - encode_image(path) -> np.ndarray   (768-dim, L2 归一化)
  - encode_text(text) -> np.ndarray    (768-dim, L2 归一化)
  - encode_texts(texts) -> np.ndarray  (批量编码, N×768-dim)
  - close()                            (释放显存)
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class LocalCLIPClient:
    """本地 Chinese-CLIP 推理客户端.

    使用 HuggingFace transformers 加载 OFA-Sys/chinese-clip-vit-large-patch14，
    支持 CPU / CUDA 推理，可选 FP16 加速（仅 CUDA）。

    典型用法::

        client = LocalCLIPClient(
            model_name="OFA-Sys/chinese-clip-vit-large-patch14",
            device="cuda",
            use_fp16=True,
        )
        img_emb = client.encode_image("output/geo_001.png")
        txt_emb = client.encode_text("一张地理教学插图")
        txts_emb = client.encode_texts(["蓝色海洋", "白色云朵", "绿色陆地"])
        client.close()
    """

    # 向量维度与 patch 数由 config 根据 CLIP_MODEL_NAME 推导，避免与模型不一致
    from src.config import EMBEDDING_DIM, CLIP_NUM_PATCHES as NUM_PATCHES

    def __init__(
        self,
        model_name: str,
        pretrained: str = "",
        device: str = "cpu",
        use_fp16: bool = False,
    ) -> None:
        """初始化本地 Chinese-CLIP 客户端.

        Args:
            model_name: HuggingFace model ID, e.g. "OFA-Sys/chinese-clip-vit-large-patch14".
            pretrained: 保留参数（向后兼容），Chinese-CLIP 不需要此参数.
            device: 推理设备 — "cpu" / "cuda" / "cuda:0".
            use_fp16: 是否启用 FP16 推理（仅 CUDA 有效，CPU 上自动忽略）.
        """
        self._model_name = model_name
        self._pretrained = pretrained  # no-op, kept for backward compat
        self._device = device
        self._use_fp16 = use_fp16 and device.startswith("cuda")

        # 延迟初始化
        self._model: Optional[object] = None
        self._processor: Optional[object] = None
        self._embedding_dim: Optional[int] = None

    # ── 公开方法 ──────────────────────────────────────

    @property
    def embedding_dim(self) -> int:
        """返回当前模型的 embedding 维度."""
        self._ensure_loaded()
        assert self._embedding_dim is not None
        return self._embedding_dim

    def encode_image(self, image_path: str) -> np.ndarray:
        """将图片编码为归一化 CLIP embedding.

        Args:
            image_path: 本地图片文件路径（PNG / JPG / WebP）.

        Returns:
            np.ndarray: L2 归一化后的 embedding 向量，shape=(768,).
        """
        t0 = time.perf_counter()
        logger.info(
            "local_clip.encode_image.start | model=%s image=%s",
            self._model_name,
            image_path,
        )

        self._ensure_loaded()

        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")

        with torch.no_grad():
            if self._device.startswith("cuda"):
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                # 模型已通过 .half() 转为 FP16 时，不再使用 autocast，
                # 避免 autocast 内部 cast 与模型权重 dtype 不一致导致 mat1/mat2 dtype 错误
                with torch.amp.autocast("cuda", enabled=not self._use_fp16):
                    features = self._model.get_image_features(**inputs)
            else:
                features = self._model.get_image_features(**inputs)

        # transformers >=4.45 返回 BaseModelOutputWithPooling，需取 .pooler_output
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = features.cpu().numpy().astype(np.float32)
        arr = features[0]  # (768,)
        # L2 归一化
        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "local_clip.encode_image.end | dim=%d duration_ms=%d",
            len(arr),
            duration_ms,
        )
        return arr

    def encode_image_patches(self, image_path: str) -> np.ndarray:
        """提取图像 patch embeddings（去 CLS token），用于 Patch-Level 评测.

        与全局 CLS embedding 不同，patch embeddings 保留了空间局部信息，
        可解决教学示意图中局部元素（箭头、标签、云层等）被整图语义稀释的问题。

        ViT-Large Patch14: 224×224 → 16×16 grid → 256 patches.
        每个 patch 经 visual_projection 映射到 768-dim joint embedding space。

        Args:
            image_path: 本地图片文件路径.

        Returns:
            np.ndarray: L2 归一化后的 patch embeddings，shape=(256, 768).
        """
        t0 = time.perf_counter()
        logger.info(
            "local_clip.encode_image_patches.start | model=%s image=%s",
            self._model_name,
            image_path,
        )

        self._ensure_loaded()

        import torch
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=img, return_tensors="pt")

        with torch.no_grad():
            if self._device.startswith("cuda"):
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                # 模型已通过 .half() 转为 FP16 时，不再使用 autocast，
                # 避免 autocast 内部 cast 与模型权重 dtype 不一致导致 mat1/mat2 dtype 错误
                with torch.amp.autocast("cuda", enabled=not self._use_fp16):
                    vision_outputs = self._model.vision_model(**inputs)
            else:
                vision_outputs = self._model.vision_model(**inputs)

            # vision_outputs.last_hidden_state: (1, 257, hidden_dim)
            #   position 0 = CLS token, positions 1..256 = patch tokens
            patch_features = vision_outputs.last_hidden_state[:, 1:, :]  # (1, 256, hidden_dim)

            # Project to joint embedding space (768-dim)
            patch_features = self._model.visual_projection(patch_features)  # (1, 256, 768)

            # Move to CPU, convert to numpy
            patch_features = patch_features.cpu().numpy().astype(np.float32)[0]  # (256, 768)

        # L2 normalize each patch independently
        norms = np.linalg.norm(patch_features, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        patch_features = patch_features / norms

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "local_clip.encode_image_patches.end | shape=%s duration_ms=%d",
            patch_features.shape,
            duration_ms,
        )
        return patch_features

    def encode_text(self, text: str) -> np.ndarray:
        """将文本编码为归一化 CLIP embedding.

        Args:
            text: 输入文本（中文或英文）.

        Returns:
            np.ndarray: L2 归一化后的 embedding 向量，shape=(768,).
        """
        result = self.encode_texts([text])
        return result[0]

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """批量编码文本，返回归一化 CLIP embedding 数组.

        相比逐个调用 encode_text()，批量编码在 CPU 上可提速 4-5×，
        在 GPU 上差距更大。模板集成场景（6 个模板 × N 个属性）
        应优先使用此方法。

        Args:
            texts: 输入文本列表.

        Returns:
            np.ndarray: L2 归一化后的 embedding 数组，shape=(len(texts), 768).
        """
        t0 = time.perf_counter()
        n = len(texts)
        preview = texts[0][:60] if texts else "(empty)"
        logger.info(
            "local_clip.encode_texts.start | model=%s n=%d preview=%s",
            self._model_name,
            n,
            preview,
        )

        self._ensure_loaded()

        import torch

        # ChineseCLIPProcessor handles tokenization
        inputs = self._processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Chinese-CLIP 默认 max_length
        )

        with torch.no_grad():
            if self._device.startswith("cuda"):
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                # 模型已通过 .half() 转为 FP16 时，不再使用 autocast，
                # 避免 autocast 内部 cast 与模型权重 dtype 不一致导致 mat1/mat2 dtype 错误
                with torch.amp.autocast("cuda", enabled=not self._use_fp16):
                    features = self._model.get_text_features(**inputs)
            else:
                features = self._model.get_text_features(**inputs)

        # transformers >=4.45 返回 BaseModelOutputWithPooling，需取 .pooler_output
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        features = features.cpu().numpy().astype(np.float32)  # (N, 768)

        # L2 逐行归一化
        norms = np.linalg.norm(features, axis=1, keepdims=True)  # (N, 1)
        norms[norms == 0] = 1.0
        features = features / norms

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "local_clip.encode_texts.end | n=%d dim=%d duration_ms=%d",
            n,
            features.shape[1],
            duration_ms,
        )
        return features

    def close(self) -> None:
        """释放模型占用的资源（显存 / 内存）."""
        if self._model is not None:
            import torch

            logger.info("local_clip.close | model=%s", self._model_name)
            self._model = None
            self._processor = None
            self._embedding_dim = None

            if self._device.startswith("cuda"):
                torch.cuda.empty_cache()

    # ── 内部方法 ──────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """延迟加载模型（首次调用时初始化，避免 FastAPI 启动阻塞）."""
        if self._model is not None:
            return

        t0 = time.perf_counter()
        logger.info(
            "local_clip.load.start | model=%s device=%s fp16=%s",
            self._model_name,
            self._device,
            self._use_fp16,
        )

        try:
            import torch
            from transformers import ChineseCLIPProcessor, ChineseCLIPModel
        except ImportError as exc:
            raise ImportError(
                "Chinese-CLIP 依赖未安装。请运行:\n"
                "  pip install transformers torch torchvision pillow\n"
                "CUDA 版本额外:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu121"
            ) from exc

        # 检查设备可用性
        _device = self._device
        if _device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning(
                    "local_clip.cuda_unavailable | 请求 device=%s 但 CUDA 不可用，回退到 CPU",
                    _device,
                )
                _device = "cpu"
                self._use_fp16 = False
            else:
                # ── GPU 信息日志 ─────────────────────────────────
                _gpu_name = torch.cuda.get_device_name(_device) or "Unknown"
                _total_mb = (
                    torch.cuda.get_device_properties(_device).total_memory / (1024**2)
                )
                _reserved_mb = torch.cuda.memory_reserved(_device) / (1024**2)
                _free_mb = _total_mb - _reserved_mb
                logger.info(
                    "local_clip.gpu_info | device=%s name=%s "
                    "vram_total_mb=%.0f vram_free_mb=%.0f fp16=%s",
                    _device,
                    _gpu_name,
                    _total_mb,
                    _free_mb,
                    self._use_fp16,
                )
                # Chinese-CLIP ViT-Large: FP32 ~1.4GB + overhead ~0.6GB → ~2GB
                # FP16 ~0.7GB + overhead ~0.5GB → ~1.2GB, 6GB 足够
                if not self._use_fp16 and _free_mb < 2500:
                    logger.warning(
                        "local_clip.vram_tight | free_vram_mb=%.0f "
                        "fp32_may_require_~2000mb | "
                        "建议设置 CLIP_USE_FP16=true 或 CLIP_DEVICE=cpu",
                        _free_mb,
                    )
                # cuDNN benchmark：固定输入尺寸（224×224），开启后持续收益
                torch.backends.cudnn.benchmark = True

        # ── HF 镜像（国内加速） ──────────────────────
        _hf_endpoint = os.environ.get("HF_ENDPOINT", "")
        if _hf_endpoint:
            logger.info("local_clip.hf_mirror | endpoint=%s", _hf_endpoint)

        # ── 加载模型与处理器 ─────────────────────────
        model = ChineseCLIPModel.from_pretrained(
            self._model_name,
            cache_dir=None,  # 使用默认缓存目录
        )
        processor = ChineseCLIPProcessor.from_pretrained(
            self._model_name,
            cache_dir=None,
        )

        # ── Fix: Chinese-CLIP tokenizer defaults to model_max_length=52 ──
        # which severely truncates Chinese text (52 chars ≈ 26 Chinese words).
        # Override to 512 to match the encode_texts() max_length parameter.
        # This ensures text encoding doesn't silently truncate long prompts.
        _orig_max_len = getattr(processor.tokenizer, "model_max_length", "?")
        processor.tokenizer.model_max_length = 512
        logger.info(
            "local_clip.tokenizer_max_length | original=%s set_to=512",
            _orig_max_len,
        )

        # 移到设备
        model = model.to(_device)

        # FP16 (仅 CUDA)
        if self._use_fp16:
            model = model.half()
            logger.info("local_clip.fp16_enabled | model converted to float16")

        model.eval()

        self._model = model
        self._processor = processor
        self._embedding_dim = model.config.projection_dim  # 768 for ViT-Large

        # 预估显存
        if _device.startswith("cuda"):
            mem_mb = torch.cuda.max_memory_allocated(_device) / (1024 * 1024)
            logger.info(
                "local_clip.gpu_mem_allocated | device=%s mem_mb=%.1f",
                _device,
                mem_mb,
            )

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "local_clip.load.end | model=%s dim=%d device=%s duration_ms=%d",
            self._model_name,
            self._embedding_dim,
            _device,
            duration_ms,
        )
