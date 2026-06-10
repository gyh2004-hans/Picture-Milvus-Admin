"""VLM 五维度主评测器 —— 对齐项目计划书 §3.2.

将图片（base64）+ 原始 prompt 送入视觉 LLM，按五个维度逐一评分（0-1）:
  1. 主体对象一致性 — 图像是否包含描述中的关键主体
  2. 属性一致性 — 颜色、大小、形状、数量等属性是否正确
  3. 空间关系一致性 — 对象位置关系是否满足描述
  4. 场景完整性 — 背景与环境是否符合要求
  5. 整体语义匹配度 — 图像与文本之间的综合相关程度

VLM 返回结构化 JSON：各维度得分 + 综合分 + 问题列表 + 缺失元素 + 优化建议.

用法：
  evaluator = VLMEvaluator()
  result = evaluator.evaluate(prompt="...", image_path="...")
  # result: EvalResult with overall_score, dimension_scores, issues, etc.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

from src.config import LLM_API_KEY, LLM_BASE_URL, VLM_EVAL_MODEL
from src.llm_utils import vision_completion
from src.models.schemas import DimensionScore, EvalResult

logger = logging.getLogger(__name__)

# ── VLM 五维度评测 prompt ──────────────────────────────

EVAL_SYSTEM_ZH = """\
你是一个专业的文生图质量评测专家。你的任务是查看生成的图片，并根据原始 prompt 描述，
从以下五个维度逐一评分（每个维度 0.0~1.0 分）：

1. **主体对象一致性** — 图像是否包含 prompt 中描述的关键主体对象（人物、物体、图标等）。
2. **属性一致性** — 颜色、大小、形状、数量、材质等具体属性是否与 prompt 描述一致。
3. **空间关系一致性** — 对象之间的位置关系（上下/左右/前后/包含等）是否正确。
4. **场景完整性** — 背景、环境、氛围、标注文字等整体场景要素是否齐全。
5. **整体语义匹配度** — 图像从整体上看与 prompt 的语义吻合程度。

你必须严格返回 JSON 格式，不要输出任何其他内容：
{
  "overall_score": 0.0,
  "dimension_scores": [
    {"dimension": "主体对象一致性", "score": 0.0, "comment": "评语不超过20字"},
    {"dimension": "属性一致性", "score": 0.0, "comment": "评语不超过20字"},
    {"dimension": "空间关系一致性", "score": 0.0, "comment": "评语不超过20字"},
    {"dimension": "场景完整性", "score": 0.0, "comment": "评语不超过20字"},
    {"dimension": "整体语义匹配度", "score": 0.0, "comment": "评语不超过20字"}
  ],
  "issues": ["问题1", "问题2"],
  "missing_elements": ["缺失的元素1"],
  "suggestions": ["优化建议1", "优化建议2"]
}

评分标准：
- 1.0：完全符合 prompt 描述，无任何偏差
- 0.8-0.9：基本符合，有轻微偏差但无伤大雅
- 0.6-0.7：部分符合，存在明显偏差
- 0.4-0.5：勉强沾边，大部分描述未体现
- 0.0-0.3：基本不相关或完全错误
"""

EVAL_SYSTEM_EN = """\
You are a professional text-to-image quality evaluator. Your task is to view the generated image
and score it against the original prompt across five dimensions (each 0.0~1.0):

1. **Subject Consistency** — Does the image contain the key subjects described in the prompt?
2. **Attribute Consistency** — Are colors, sizes, shapes, quantities, and materials correct?
3. **Spatial Consistency** — Are positional relationships between objects correct?
4. **Scene Completeness** — Are background, environment, atmosphere, annotations all present?
5. **Overall Semantic Match** — How well does the image semantically match the prompt overall?

You MUST return strict JSON format, no other text:
{
  "overall_score": 0.0,
  "dimension_scores": [
    {"dimension": "Subject Consistency", "score": 0.0, "comment": "brief reason"},
    {"dimension": "Attribute Consistency", "score": 0.0, "comment": "brief reason"},
    {"dimension": "Spatial Consistency", "score": 0.0, "comment": "brief reason"},
    {"dimension": "Scene Completeness", "score": 0.0, "comment": "brief reason"},
    {"dimension": "Overall Semantic Match", "score": 0.0, "comment": "brief reason"}
  ],
  "issues": ["issue1", "issue2"],
  "missing_elements": ["missing element1"],
  "suggestions": ["suggestion1", "suggestion2"]
}
"""


# ── 图片编码 ──────────────────────────────────────────

# VLM 图片压缩上限（长边像素），避免大图导致 API 超时/拒绝
_MAX_IMAGE_DIM = 768
_MAX_IMAGE_BYTES = 200 * 1024  # 200KB，超过则压缩到 JPEG


def _image_to_base64(image_path: str) -> str:
    """读取图片文件并转为 base64 data URI.

    自动压缩大图：长边 > _MAX_IMAGE_DIM 时等比缩放，
    体积 > _MAX_IMAGE_BYTES 时转 JPEG 压缩。
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")

    suffix = path.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    mime = mime_map.get(suffix, "image/png")

    data = path.read_bytes()
    original_size = len(data)

    # 大图始终需要打开检查尺寸；体积超标则额外压缩
    needs_compress = original_size > _MAX_IMAGE_BYTES

    if needs_compress:
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            w, h = img.size
            long_side = max(w, h)

            # 等比缩放
            if long_side > _MAX_IMAGE_DIM:
                ratio = _MAX_IMAGE_DIM / long_side
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.info(
                    "vlm_evaluator.image_resize | %s %dx%d → %dx%d",
                    path.name, w, h, new_w, new_h,
                )

            # 转 JPEG 压缩体积
            buf = io.BytesIO()
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(buf, format="JPEG", quality=85, optimize=True)
            data = buf.getvalue()
            mime = "image/jpeg"
            logger.info(
                "vlm_evaluator.image_compress | %s %d → %d bytes (%.0f%%)",
                path.name, original_size, len(data),
                len(data) / original_size * 100,
            )
        except ImportError:
            logger.warning("vlm_evaluator: PIL not available, sending original image")
        except Exception as exc:
            logger.warning("vlm_evaluator.image_compress_error | %s: %s", path.name, exc)

    b64 = base64.b64encode(data).decode("ascii")
    logger.info(
        "vlm_evaluator.image_b64 | %s original=%d final=%d b64_chars=%d",
        path.name, original_size, len(data), len(b64),
    )
    return f"data:{mime};base64,{b64}"


# ── JSON 解析工具 ──────────────────────────────────────

def _parse_eval_json(answer: str) -> dict | None:
    """从 VLM 输出中提取评测 JSON 对象."""
    text = answer.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end > start:
            text = text[start + 3:end].strip()
            if text.startswith("json"):
                text = text[4:].strip()

    # 找到 JSON 对象边界
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
# VLMEvaluator — VLM 五维度主评测器
# ══════════════════════════════════════════════════════

class VLMEvaluator:
    """VLM 五维度主评测器（项目计划书 §3.2）.

    将生成的图片与原始 prompt 送入视觉 LLM，从五个维度评分，
    返回结构化的 EvalResult（含维度分 + 问题列表 + 缺失元素 + 建议）.
    """

    def __init__(self) -> None:
        self._api_key = LLM_API_KEY
        self._base_url = LLM_BASE_URL.rstrip("/")
        self._model = VLM_EVAL_MODEL

    async def evaluate(
        self,
        prompt: str,
        image_path: str,
        lang: str = "zh",
    ) -> EvalResult:
        """对生成图片执行五维度 VLM 评测.

        Args:
            prompt: 原始用户 prompt（评测的参照标准）.
            image_path: 生成图片的本地路径.
            lang: 评测语言 — "zh" 中文 / "en" 英文.

        Returns:
            EvalResult: 含 overall_score + dimension_scores + issues 等.

        Raises:
            RuntimeError: VLM API 调用失败.
            FileNotFoundError: 图片文件不存在.
        """
        t0 = time.perf_counter()
        logger.info(
            "vlm_evaluator.evaluate.start | prompt_len=%d image=%s lang=%s",
            len(prompt),
            image_path,
            lang,
        )

        # 编码图片
        try:
            image_b64 = _image_to_base64(image_path)
        except FileNotFoundError:
            raise

        # 选择 prompt 模板
        is_zh = lang == "zh"
        system = EVAL_SYSTEM_ZH if is_zh else EVAL_SYSTEM_EN

        # 构建用户消息
        user_text = (
            f"原始 prompt：{prompt}\n\n请对以上 prompt 对应的生成图片进行五维度评测。"
            if is_zh
            else f"Original prompt: {prompt}\n\nPlease evaluate the generated image against this prompt across five dimensions."
        )

        # 调用 VLM
        try:
            answer = await vision_completion(
                system=system,
                user_text=user_text,
                image_b64=image_b64,
                model=self._model,
                temperature=0.1,
                max_tokens=1024,
                max_retries=2,
            )
        except (RuntimeError, ValueError) as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "vlm_evaluator.evaluate.error | duration_ms=%d error=%s",
                duration_ms,
                exc,
            )
            raise RuntimeError(f"VLM 评测调用失败: {exc}") from exc

        # 解析 VLM 返回的 JSON
        parsed = _parse_eval_json(answer)
        if parsed is None:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "vlm_evaluator.evaluate.parse_error | duration_ms=%d answer_preview=%s",
                duration_ms,
                answer[:300],
            )
            raise RuntimeError(
                f"VLM 评测返回无法解析为 JSON，原始输出前300字: {answer[:300]}"
            )

        # 构建 EvalResult
        overall_score = float(parsed.get("overall_score", 0.0))
        dimension_scores = [
            DimensionScore(
                dimension=d.get("dimension", f"dim_{i}"),
                score=float(d.get("score", 0.0)),
                comment=str(d.get("comment", "")),
            )
            for i, d in enumerate(parsed.get("dimension_scores", []))
        ]
        issues = [str(i) for i in parsed.get("issues", [])]
        missing_elements = [str(m) for m in parsed.get("missing_elements", [])]
        suggestions = [str(s) for s in parsed.get("suggestions", [])]

        # 如果 VLM 未返回 dimension_scores，用 overall_score 兜底
        if not dimension_scores and overall_score > 0:
            dim_names = (
                ["主体对象一致性", "属性一致性", "空间关系一致性", "场景完整性", "整体语义匹配度"]
                if is_zh
                else ["Subject Consistency", "Attribute Consistency", "Spatial Consistency", "Scene Completeness", "Overall Semantic Match"]
            )
            dimension_scores = [
                DimensionScore(dimension=name, score=overall_score, comment="(from overall_score)")
                for name in dim_names
            ]

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "vlm_evaluator.evaluate.end | overall=%.3f dims=%d issues=%d missing=%d duration_ms=%d",
            overall_score,
            len(dimension_scores),
            len(issues),
            len(missing_elements),
            duration_ms,
        )

        return EvalResult(
            overall_score=round(overall_score, 4),
            dimension_scores=dimension_scores,
            issues=issues,
            missing_elements=missing_elements,
            suggestions=suggestions,
        )
