"""图片内容 VLM 解析器（v6: 通用版，替代 education_parser）.

将图片送入视觉 LLM（VLM），解析出标准化的内容结构 JSON，
用于构建 semantic_text 并存入 Milvus。

核心能力:
  - 图片 → 通用内容结构 JSON（分类/主体对象/场景/风格/色调/标签/检索描述）
  - 构建 semantic_text（结构化拼接，供 Chinese-CLIP 编码）
  - 开放域分类，不限定特定领域

用法:
  parser = ImageContentParser()
  result = await parser.parse(image_path)
  semantic_text = parser.build_semantic_text(result)
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path

from src.config import LLM_API_KEY, LLM_BASE_URL, VLM_VISION_MODEL, CATEGORIES
from src.llm_utils import vision_completion
from src.models.schemas import ImageContentParseResult

logger = logging.getLogger(__name__)

# ── VLM 图片内容解析 prompt ──────────────────────────────

IMAGE_CONTENT_PARSE_SYSTEM = """\
你是一个通用图片内容分析专家。你的任务是查看上传的图片（照片/插图/图表/截图等），
从内容角度分析其信息，输出标准化的内容结构 JSON。

你必须严格返回 JSON 格式，不要输出任何其他内容:
{
  "category": "图片分类（从可用分类中选择最匹配的一项，如无法匹配则选'其他'）",
  "content_type": "图片类型: 照片/插图/图表/截图/海报/其他",
  "main_objects": ["主体对象1", "主体对象2"],
  "scene_description": "场景的简要描述",
  "style": "风格描述（如写实/卡通/极简/复古/科技感/手绘等）",
  "color_palette": ["主色调1", "主色调2", "主色调3"],
  "tags": ["关键词1", "关键词2", "关键词3", "关键词4"],
  "retrieval_prompt": "未来用户可能输入的搜索描述（自然语言，用于检索召回）"
}

要求:
1. category 必须从可用分类列表中选择最匹配的一项。如果图片内容不属于任何已有分类，使用"其他"。
2. retrieval_prompt 必须是未来用户可能输入的自然语言搜索描述，
   例如 "城市夜景航拍照片，摩天大楼灯光璀璨" 或 "极简风格美食摄影，日式拉面特写"。
   要包含: 主体内容 + 风格 + 类型等关键信息。
3. main_objects 列出图中最核心的2-6个主体对象。
4. color_palette 列出3-5个主要色调。
5. tags 列出4-8个核心关键词，用于检索匹配。
6. 如果无法确定某个字段，用空字符串 "" 或空数组 []。

示例输出:
{
  "category": "美食",
  "content_type": "照片",
  "main_objects": ["日式拉面", "筷子", "餐桌"],
  "scene_description": "一碗热气腾腾的日式拉面特写，桌上有筷子和调料",
  "style": "美食摄影",
  "color_palette": ["暖黄色", "白色", "深棕色"],
  "tags": ["拉面", "日料", "美食摄影", "特写", "暖色"],
  "retrieval_prompt": "日式拉面美食摄影特写，热气腾腾的暖色调风格"
}
"""

# 图片压缩参数
_MAX_IMAGE_DIM = 1024
_MAX_IMAGE_BYTES = 300 * 1024


def _image_to_base64(image_path: str) -> str:
    """读取图片文件并转为 base64 data URI，自动压缩大图."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".webp": "image/webp", ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "image/png")

    data = path.read_bytes()
    original_size = len(data)

    if original_size > _MAX_IMAGE_BYTES:
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(data))
            w, h = img.size
            long_side = max(w, h)
            if long_side > _MAX_IMAGE_DIM:
                ratio = _MAX_IMAGE_DIM / long_side
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info("image_content_parser.resize | %s %dx%d → %dx%d", path.name, w, h, new_w, new_h)
            buf = io.BytesIO()
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=85, optimize=True)
            data = buf.getvalue()
            mime = "image/jpeg"
            logger.info("image_content_parser.compress | %s %d → %d bytes", path.name, original_size, len(data))
        except ImportError:
            logger.warning("image_content_parser: PIL not available, sending original image")
        except Exception as exc:
            logger.warning("image_content_parser.compress_error | %s: %s", path.name, exc)

    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _parse_content_json(answer: str) -> dict | None:
    """从 VLM 输出中提取内容解析 JSON."""
    text = answer.strip()

    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end > start:
            text = text[start + 3:end].strip()
            if text.startswith("json"):
                text = text[4:].strip()

    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return None


# ══════════════════════════════════════════════════════
# ImageContentParser
# ══════════════════════════════════════════════════════

class ImageContentParser:
    """通用图片内容 VLM 解析器.

    将图片送入视觉 LLM，解析为 ImageContentParseResult。

    典型用法::

        parser = ImageContentParser()
        result = await parser.parse("uploads/photo.png")
        semantic_text = parser.build_semantic_text(result)
    """

    def __init__(self) -> None:
        self._api_key = LLM_API_KEY
        self._base_url = LLM_BASE_URL.rstrip("/")
        self._model = VLM_VISION_MODEL
        self._categories = list(CATEGORIES)

    async def parse(self, image_path: str) -> ImageContentParseResult:
        """解析图片，返回通用内容结构.

        Args:
            image_path: 本地图片文件路径.

        Returns:
            ImageContentParseResult: 含分类/主体对象/场景/风格/标签等.

        Raises:
            RuntimeError: VLM API 调用失败或解析失败.
            FileNotFoundError: 图片文件不存在.
        """
        t0 = time.perf_counter()
        logger.info("image_content_parser.parse.start | image=%s model=%s", image_path, self._model)

        # 编码图片
        image_b64 = _image_to_base64(image_path)

        # 构建带可用分类列表的 system prompt
        categories_hint = "、".join(self._categories)
        system_prompt = IMAGE_CONTENT_PARSE_SYSTEM + f"\n\n当前可用分类列表: {categories_hint}"

        # 构建用户消息
        user_text = "请对这张图片进行内容分析，输出标准 JSON。"

        # 调用 VLM
        try:
            answer = await vision_completion(
                system=system_prompt,
                user_text=user_text,
                image_b64=image_b64,
                model=self._model,
                temperature=0.1,
                max_tokens=1024,
                max_retries=2,
            )
        except (RuntimeError, ValueError) as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error("image_content_parser.parse.error | duration_ms=%d error=%s", duration_ms, exc)
            raise RuntimeError(f"VLM 图片内容解析调用失败: {exc}") from exc

        # 解析 JSON
        parsed = _parse_content_json(answer)
        if parsed is None:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "image_content_parser.parse.json_error | duration_ms=%d answer_preview=%s",
                duration_ms, answer[:300],
            )
            raise RuntimeError(f"VLM 图片内容解析返回无法解析为 JSON，原始输出前300字: {answer[:300]}")

        # 构建结果
        result = ImageContentParseResult(
            category=self._normalize_category(parsed.get("category", "")),
            content_type=str(parsed.get("content_type", "")).strip(),
            main_objects=[str(o) for o in parsed.get("main_objects", [])],
            scene_description=str(parsed.get("scene_description", "")).strip(),
            style=str(parsed.get("style", "")).strip(),
            color_palette=[str(c) for c in parsed.get("color_palette", [])],
            tags=[str(t) for t in parsed.get("tags", [])],
            retrieval_prompt=str(parsed.get("retrieval_prompt", "")).strip(),
        )

        # 兜底: retrieval_prompt 缺失时自动构造
        if not result.retrieval_prompt:
            result.retrieval_prompt = self._build_retrieval_prompt(result)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "image_content_parser.parse.end | category=%s content_type=%s objects=%d "
            "tags=%d duration_ms=%d",
            result.category, result.content_type, len(result.main_objects),
            len(result.tags), duration_ms,
        )
        return result

    # ── semantic_text 构建 ──

    @staticmethod
    def build_semantic_text(parse_result: ImageContentParseResult) -> str:
        """将图片内容解析结果构建为标准 semantic_text.

        格式::

            分类: 美食
            类型: 照片
            主体: 日式拉面、筷子、餐桌
            场景: 一碗热气腾腾的日式拉面特写
            风格: 美食摄影
            色调: 暖黄色、白色、深棕色
            标签: 拉面、日料、美食摄影、特写
            检索描述: 日式拉面美食摄影特写，热气腾腾的暖色调风格

        Args:
            parse_result: VLM 解析返回的图片内容结构.

        Returns:
            str: 结构化拼接的语义文本，供 Chinese-CLIP 编码.
        """
        parts: list[str] = []

        parts.append(f"分类: {parse_result.category}" if parse_result.category else "分类: 其他")
        parts.append(f"类型: {parse_result.content_type}" if parse_result.content_type else "类型: 其他")

        if parse_result.main_objects:
            objects = "、".join(parse_result.main_objects[:6])
            parts.append(f"主体: {objects}")

        if parse_result.scene_description:
            parts.append(f"场景: {parse_result.scene_description}")

        parts.append(f"风格: {parse_result.style}" if parse_result.style else "风格: 未标注")

        if parse_result.color_palette:
            colors = "、".join(parse_result.color_palette[:5])
            parts.append(f"色调: {colors}")

        if parse_result.tags:
            tags = "、".join(parse_result.tags[:8])
            parts.append(f"标签: {tags}")

        if parse_result.retrieval_prompt:
            parts.append(f"检索描述: {parse_result.retrieval_prompt}")

        return "\n".join(parts)

    # ── 内部方法 ──

    def _normalize_category(self, raw: str) -> str:
        """归一化分类名为可用分类之一."""
        s = raw.strip()
        if not s:
            return "其他"
        # 精确匹配
        if s in self._categories:
            return s
        # 模糊匹配
        for cat in self._categories:
            if cat in s or s in cat:
                return cat
        logger.warning("image_content_parser.unknown_category | raw=%r, using '其他'", raw)
        return "其他"

    @staticmethod
    def _build_retrieval_prompt(result: ImageContentParseResult) -> str:
        """检索描述缺失时，从其他字段自动构造."""
        parts = []
        if result.category and result.category != "其他":
            parts.append(result.category)
        if result.main_objects:
            parts.append("、".join(result.main_objects[:3]))
        if result.style:
            parts.append(result.style + "风格")
        if result.content_type:
            parts.append(result.content_type)
        return " ".join(parts) if parts else "通用图片"


# ══════════════════════════════════════════════════════
# 向后兼容：保留旧的 EducationParser 作为别名
# ══════════════════════════════════════════════════════

# 从旧模块导入的代码可以通过此别名继续工作
# 但 build_semantic_text 行为已变更为通用格式
