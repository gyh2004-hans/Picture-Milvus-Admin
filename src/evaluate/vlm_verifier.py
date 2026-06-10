"""VLM 复核模块 —— 用豆包视觉模型对 CLIP missing 属性做二次确认.

当 CLIP 将某属性判为 missing（< ATTRIBUTE_WEAK）时，可能只是因为
CLIP 对复杂教学插图、中文标注等场景的 embedding 分辨率不足。
VLM（视觉语言模型）则能直接「看」图回答问题，对组合场景的理解
远超 CLIP，恰好弥补短板。

v2 升级:
  - VLM 输出强制 JSON 格式（避免 parse fail）
  - 统一复用 llm_utils.vision_completion()（统一日志 + 重试 + 错误处理）
  - _expand_attr() 从字符串精确匹配改为 semantic retrieval
  - VLM promoted 属性使用有意义的占位分（VLM_PROMOTED_SCORE）替代 score=-1
  - 新增 verify_style_attributes() 对抽象风格做 soft evaluation
  - 新增 verify_content_attributes() 作为 content attribute 最终裁判

用法：
  - 通过环境变量 VLM_VERIFY_ENABLED=true 开启
  - 在 CLIPScorer.evaluate() 之后调用 VLMVerifier.verify_missing()
  - VLM 判定 YES 的属性从 missing 提升到 weak（score=VLM_PROMOTED_SCORE 标记为 VLM 覆盖）.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from src.config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    VLM_PROMOTED_SCORE,
    VLM_VERIFY_MAX_ATTRS,
    VLM_VERIFY_LANG,
    VLM_VISION_MODEL,
)
from src.llm_utils import vision_completion
from src.models.schemas import AttributeScore

logger = logging.getLogger(__name__)

# ── VLM prompt 模板（v2: 强制 JSON 输出） ─────────────────

VERIFY_SYSTEM_ZH = """\
你是一个图像审核员。你的任务是查看图片，判断某个视觉属性是否在图片中可见。

你必须严格返回 JSON 格式，不要输出任何其他内容：
{"verdict": "YES", "reason": "简短理由不超过20字"}
或
{"verdict": "NO", "reason": "简短理由不超过20字"}

判断标准：
- YES：该属性在图片中明确可见（即使不够突出、局部、或尺寸较小）。
- NO：该属性在图片中完全看不到或无法辨认。"""

VERIFY_SYSTEM_EN = """\
You are an image reviewer. Your task is to look at an image and judge \
whether a visual attribute is visible in the image.

You MUST return strict JSON format, no other text:
{"verdict": "YES", "reason": "brief reason, one short sentence"}
or
{"verdict": "NO", "reason": "brief reason, one short sentence"}

Criteria:
- YES: the attribute is clearly visible in the image (even if not prominent, partial, or small).
- NO: the attribute is not visible or not recognizable at all."""

# ── Style VLM 评估 prompt（soft evaluation，3 级判断） ──

STYLE_VERIFY_SYSTEM_ZH = """\
你是一个教材插图风格审核员。你的任务是查看图片，判断某个抽象风格属性在图片中的体现程度。

你必须严格返回 JSON 格式，不要输出任何其他内容：
{"verdict": "PRESENT", "reason": "风格已体现，简述理由不超过20字"}
或
{"verdict": "PARTIAL", "reason": "风格部分体现，简述理由不超过20字"}
或
{"verdict": "ABSENT", "reason": "风格未体现，简述理由不超过20字"}

判断标准：
- PRESENT：该风格属性在图片中明确可感知，整体风格一致。
- PARTIAL：该风格属性在图片中有痕迹但不明显或不一致。
- ABSENT：该风格属性在图片中完全看不到与描述相符的风格特征。"""

STYLE_VERIFY_SYSTEM_EN = """\
You are a textbook illustration style reviewer. Your task is to evaluate \
how well an abstract style attribute is reflected in the image.

You MUST return strict JSON format, no other text:
{"verdict": "PRESENT", "reason": "brief reason"}
or
{"verdict": "PARTIAL", "reason": "brief reason"}
or
{"verdict": "ABSENT", "reason": "brief reason"}

Criteria:
- PRESENT: the style attribute is clearly perceivable, overall style consistent.
- PARTIAL: traces of the style exist but not prominent or inconsistent.
- ABSENT: no visible style features matching the description."""

# ── 属性自然语言展开（v2: semantic retrieval） ──


def _expand_attr_semantic(attr: str, original_prompt: str, lang: str) -> str:
    """用语义上下文展开属性描述（替代旧版的字符串精确匹配）.

    策略：
    1. 将属性嵌入到原始 prompt 的语义上下文中描述
    2. 不再依赖 find() 精确匹配（LLM 拆解后的属性可能与原始表达不同）
    3. VLM 能看到属性的完整语义背景，更容易与图片建立对应关系

    这解决了旧版的核心问题：LLM 拆解出的属性（如"箭头标注"）可能
    不在原始 prompt 中字面出现（原始可能是"蓝色箭头=碰撞挤压"），
    导致 _expand_attr() 回退到裸属性关键字模式。
    """
    # 截断过长的原始 prompt（VLM 的上下文窗口有限）
    prompt_preview = original_prompt[:300] + ("..." if len(original_prompt) > 300 else "")

    if lang == "zh":
        return (
            f"根据原始图片描述「{prompt_preview}」，"
            f"图中是否可以看到与「{attr}」相关的视觉元素或特征？"
        )
    else:
        return (
            f"Based on the original image description '{prompt_preview}', "
            f"can you see visual elements or features related to '{attr}' in the image?"
        )


def _expand_attr(attr: str, original_prompt: str, lang: str) -> str:
    """向后兼容包装：调用 semantic retrieval 版本.

    旧版签名保留，但内部实现已升级为 semantic retrieval。
    """
    return _expand_attr_semantic(attr, original_prompt, lang)


# ── 图片编码 ──────────────────────────────────────────

def _image_to_base64(image_path: str) -> str:
    """读取图片文件并转为 base64 data URI."""
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
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ── JSON 解析工具 ──────────────────────────────────────


def _parse_vlm_json(answer: str) -> dict | list | None:
    """从 VLM 输出中提取 JSON 对象或数组.

    兼容模型输出带 markdown 代码块或前后说明文字的情况。
    v2: 同时支持 JSON 对象 ({...}) 和 JSON 数组 ([{...}]),
        优先尝试数组（batch 模式），再回退到对象（单属性模式）。
    返回 None 表示无法提取有效 JSON。
    """
    text = answer.strip()

    # 尝试提取 ```json ... ``` 代码块
    if "```" in text:
        start = text.find("```")
        end = text.find("```", start + 3)
        if end > start:
            text = text[start + 3:end].strip()
            if text.startswith("json"):
                text = text[4:].strip()

    # v2: 优先尝试 JSON 数组（batch 模式最常见），再尝试对象
    bracket_start = text.find("[")
    bracket_end = text.rfind("]")
    brace_start = text.find("{")
    brace_end = text.rfind("}")

    # 数组优先（如果数组边界在对象边界之前或同时存在）
    if bracket_start != -1 and bracket_end > bracket_start:
        if brace_start == -1 or bracket_start < brace_start:
            try:
                parsed = json.loads(text[bracket_start:bracket_end + 1])
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass

    # 回退到 JSON 对象
    if brace_start != -1 and brace_end > brace_start:
        try:
            parsed = json.loads(text[brace_start:brace_end + 1])
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass

    # 最后尝试直接解析全文
    try:
        parsed = json.loads(text)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass

    return None


def _extract_verdict_json(parsed: dict) -> Optional[bool]:
    """从已解析的 JSON 中提取 YES/NO verdict."""
    verdict = str(parsed.get("verdict", "")).upper().strip()
    if verdict in ("YES", "PRESENT"):
        return True
    if verdict in ("NO", "ABSENT"):
        return False
    if verdict == "PARTIAL":
        return None  # 三级判断的中间状态
    return None


# ── 旧版兼容: 自由文本解析（fallback） ────────────────


def _extract_yes_no(text: str) -> Optional[bool]:
    """从自由文本中提取 YES/NO 判定（v1 fallback）.

    注意：否定词先于肯定词检查，避免 "不存在" 被 "存在" 误判为肯定。
    """
    t = text.upper()

    # ── 否定词（先检查） ──
    _negative_en = [
        "NO", "ABSENT", "MISSING", "NOT VISIBLE", "NOT FOUND",
        "NOT PRESENT", "CANNOT SEE", "CAN'T SEE", "NONE",
    ]
    _negative_zh = [
        "否", "不", "无", "没有", "看不到", "不存在", "不可以",
        "无法", "未", "缺少", "缺失",
    ]

    for kw in _negative_en:
        if kw in t:
            return False
    for kw in _negative_zh:
        if kw in text:
            return False

    # ── 肯定词 ──
    _positive_en = [
        "YES", "PRESENT", "VISIBLE", "FOUND", "CONTAINS",
    ]
    _positive_zh = ["是", "可见", "存在", "包含", "有"]

    for kw in _positive_en:
        if kw in t:
            return True
    for kw in _positive_zh:
        if kw in text:
            return True
    if "可以" in text:
        return True

    return None


class VLMVerifier:
    """豆包视觉模型复核器 —— v2 升级版.

    - 对 CLIP missing 属性做二次判断（content attribute 最终裁判）
    - 对抽象风格属性做 soft evaluation（style VLM）
    - 统一使用 llm_utils.vision_completion() 进行 API 调用
    """

    def __init__(self) -> None:
        self._api_key = LLM_API_KEY
        self._base_url = LLM_BASE_URL.rstrip("/")
        self._model = VLM_VISION_MODEL

    # ── 公开 API ──────────────────────────────────────

    async def verify_missing(
        self,
        image_path: str,
        missing: list[AttributeScore],
        original_prompt: str = "",
        lang: str = "",
    ) -> tuple[list[AttributeScore], list[AttributeScore]]:
        """对 CLIP 判为 missing 的属性逐项 VLM 复核（v2: JSON 输出 + 统一 LLM）.

        Args:
            image_path: 图片路径.
            missing: CLIP 判为 missing 的属性列表.
            original_prompt: 原始图片生成 prompt，作为 VLM 判断的语义上下文.
            lang: 复核语言 — "zh" 中文 / "en" 英文（默认取 VLM_VERIFY_LANG）.

        Returns:
            (still_missing, promoted_to_weak):
              - still_missing: VLM 也认为缺失的（维持 missing）.
              - promoted_to_weak: VLM 认为可见的（提升为 weak，score=VLM_PROMOTED_SCORE
                而非旧版 score=-1，避免干扰 convergence / composite_score）.
        """
        if not missing:
            logger.info("vlm_verifier.verify_missing | no missing attrs, skip")
            return [], []

        lang = lang or VLM_VERIFY_LANG
        t0 = time.perf_counter()
        logger.info(
            "vlm_verifier.verify_missing.start | n_attrs=%d image=%s lang=%s prompt_len=%d",
            len(missing),
            image_path,
            lang,
            len(original_prompt),
        )

        try:
            image_b64 = _image_to_base64(image_path)
        except FileNotFoundError as exc:
            logger.error("vlm_verifier.verify_missing.error | %s", exc)
            return list(missing), []

        # 分批校验（每批最多 VLM_VERIFY_MAX_ATTRS 个属性）
        still_missing: list[AttributeScore] = []
        promoted: list[AttributeScore] = []

        for i in range(0, len(missing), VLM_VERIFY_MAX_ATTRS):
            batch = missing[i:i + VLM_VERIFY_MAX_ATTRS]
            batch_still, batch_promoted = await self._verify_batch(
                image_b64, batch, original_prompt, lang,
            )
            still_missing.extend(batch_still)
            promoted.extend(batch_promoted)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "vlm_verifier.verify_missing.end | input=%d still_missing=%d promoted=%d duration_ms=%d",
            len(missing),
            len(still_missing),
            len(promoted),
            duration_ms,
        )
        return still_missing, promoted

    async def verify_style_attributes(
        self,
        image_path: str,
        style_attrs: list[str],
        original_prompt: str = "",
        lang: str = "",
    ) -> dict[str, str]:
        """对抽象风格属性做 VLM soft evaluation（三级判断: PRESENT / PARTIAL / ABSENT）.

        风格属性（教材风格、浅蓝色调等）不适用 CLIP 评测，改为 VLM 定性评估。
        与 verify_missing() 不同，此方法输出三级判断而非二分类。

        Args:
            image_path: 图片路径.
            style_attrs: 风格属性字符串列表.
            original_prompt: 原始 prompt 作为上下文.
            lang: 语言.

        Returns:
            {attr: "PRESENT" | "PARTIAL" | "ABSENT"} 字典.
        """
        if not style_attrs:
            return {}

        lang = lang or VLM_VERIFY_LANG
        t0 = time.perf_counter()
        logger.info(
            "vlm_verifier.verify_style.start | n_attrs=%d image=%s",
            len(style_attrs),
            image_path,
        )

        try:
            image_b64 = _image_to_base64(image_path)
        except FileNotFoundError as exc:
            logger.error("vlm_verifier.verify_style.error | %s", exc)
            return {a: "ABSENT" for a in style_attrs}

        is_zh = lang == "zh"
        system = STYLE_VERIFY_SYSTEM_ZH if is_zh else STYLE_VERIFY_SYSTEM_EN

        results: dict[str, str] = {}
        for attr in style_attrs:
            question = _expand_attr_semantic(attr, original_prompt, lang)

            try:
                answer = await vision_completion(
                    system=system,
                    user_text=question,
                    image_b64=image_b64,
                    model=self._model,
                    temperature=0.1,
                    max_tokens=128,
                    max_retries=1,
                )
            except (RuntimeError, ValueError) as exc:
                logger.error(
                    "vlm_verifier.verify_style.error | attr=%s error=%s",
                    attr,
                    exc,
                )
                results[attr] = "ABSENT"
                continue

            parsed = _parse_vlm_json(answer)
            if isinstance(parsed, dict):
                verdict = str(parsed.get("verdict", "")).upper().strip()
                if verdict in ("PRESENT", "PARTIAL", "ABSENT"):
                    results[attr] = verdict
                    logger.info(
                        "vlm_verifier.verify_style | attr=%s verdict=%s reason=%s",
                        attr,
                        verdict,
                        parsed.get("reason", ""),
                    )
                    continue

            # fallback: 自由文本解析
            logger.warning(
                "vlm_verifier.verify_style.parse_fallback | attr=%s raw=%s",
                attr,
                answer[:200],
            )
            yes_no = _extract_yes_no(answer)
            if yes_no is True:
                results[attr] = "PRESENT"
            elif yes_no is False:
                results[attr] = "ABSENT"
            else:
                results[attr] = "PARTIAL"

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "vlm_verifier.verify_style.end | n=%d duration_ms=%d results=%s",
            len(style_attrs),
            duration_ms,
            {a: v for a, v in results.items()},
        )
        return results

    async def verify_content_attributes(
        self,
        image_path: str,
        content_attrs: list[str],
        original_prompt: str = "",
        lang: str = "",
    ) -> dict[str, bool]:
        """对 content attribute 做 VLM 最终裁判（v2: 比 verify_missing 更积极的判定）.

        与 verify_missing() 不同：
        - verify_missing 是"救火"——把 CLIP 漏掉的补回来
        - verify_content_attributes 是"裁判"——对所有 content attribute 独立判断
        - 返回 {attr: is_visible} 字典

        当 Patch CLIP 分数处于灰色区间时（weak 范围内），用此方法做最终判定。
        """
        if not content_attrs:
            return {}

        lang = lang or VLM_VERIFY_LANG
        t0 = time.perf_counter()
        logger.info(
            "vlm_verifier.verify_content.start | n_attrs=%d image=%s",
            len(content_attrs),
            image_path,
        )

        try:
            image_b64 = _image_to_base64(image_path)
        except FileNotFoundError as exc:
            logger.error("vlm_verifier.verify_content.error | %s", exc)
            return {a: False for a in content_attrs}

        is_zh = lang == "zh"
        system = VERIFY_SYSTEM_ZH if is_zh else VERIFY_SYSTEM_EN

        results: dict[str, bool] = {}
        for attr in content_attrs:
            question = _expand_attr_semantic(attr, original_prompt, lang)

            try:
                answer = await vision_completion(
                    system=system,
                    user_text=question,
                    image_b64=image_b64,
                    model=self._model,
                    temperature=0.1,
                    max_tokens=128,
                    max_retries=1,
                )
            except (RuntimeError, ValueError) as exc:
                logger.error(
                    "vlm_verifier.verify_content.error | attr=%s error=%s",
                    attr,
                    exc,
                )
                results[attr] = False
                continue

            # 优先 JSON 解析
            parsed = _parse_vlm_json(answer)
            if isinstance(parsed, dict):
                verdict = _extract_verdict_json(parsed)
                if verdict is not None:
                    results[attr] = verdict
                    logger.info(
                        "vlm_verifier.verify_content | attr=%s verdict=%s reason=%s",
                        attr,
                        "YES" if verdict else "NO",
                        parsed.get("reason", ""),
                    )
                    continue

            # fallback
            logger.warning(
                "vlm_verifier.verify_content.parse_fallback | attr=%s raw=%s",
                attr,
                answer[:200],
            )
            yes_no = _extract_yes_no(answer)
            results[attr] = yes_no if yes_no is not None else False

        duration_ms = int((time.perf_counter() - t0) * 1000)
        n_yes = sum(1 for v in results.values() if v)
        logger.info(
            "vlm_verifier.verify_content.end | n=%d yes=%d duration_ms=%d",
            len(content_attrs),
            n_yes,
            duration_ms,
        )
        return results

    # ── 内部逻辑 ──────────────────────────────────────

    async def _verify_batch(
        self,
        image_b64: str,
        attrs: list[AttributeScore],
        original_prompt: str,
        lang: str,
    ) -> tuple[list[AttributeScore], list[AttributeScore]]:
        """单批 VLM 校验（v2: JSON 输出优先）.

        对单个属性：一问一答。对多个属性：合并为一条消息批量问，
        要求模型返回 JSON 数组。
        """
        is_zh = lang == "zh"
        system = VERIFY_SYSTEM_ZH if is_zh else VERIFY_SYSTEM_EN

        if len(attrs) == 1:
            question = _expand_attr_semantic(attrs[0].attribute, original_prompt, lang)
        else:
            # 批量：编号列出所有属性，要求 JSON 数组输出
            if is_zh:
                lines = [
                    f"原始图片描述：{original_prompt[:200]}",
                    "",
                    "请逐一判断以下视觉元素是否在图中可见：",
                ]
                for j, a in enumerate(attrs, 1):
                    lines.append(f"{j}. {a.attribute}")
                lines.append("")
                lines.append(
                    '返回 JSON 数组，每项格式：{"index":编号,"verdict":"YES"或"NO","reason":"理由"}'
                )
                question = "\n".join(lines)
            else:
                lines = [
                    f"Original image description: {original_prompt[:200]}",
                    "",
                    "Judge each visual element below for visibility in the image:",
                ]
                for j, a in enumerate(attrs, 1):
                    lines.append(f"{j}. {a.attribute}")
                lines.append("")
                lines.append(
                    'Return JSON array: [{"index":N,"verdict":"YES"|"NO","reason":"reason"}, ...]'
                )
                question = "\n".join(lines)

        # 调用 VLM
        t0 = time.perf_counter()
        logger.info(
            "vlm_verifier.batch.start | n=%d model=%s prompt_len=%d",
            len(attrs),
            self._model,
            len(original_prompt),
        )

        try:
            answer = await vision_completion(
                system=system,
                user_text=question,
                image_b64=image_b64,
                model=self._model,
                temperature=0.1,
                max_tokens=512,
                max_retries=1,
            )
        except (RuntimeError, ValueError) as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "vlm_verifier.batch.error | n=%d duration_ms=%d error=%s",
                len(attrs),
                duration_ms,
                exc,
            )
            return list(attrs), []

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "vlm_verifier.batch.end | n=%d duration_ms=%d answer_len=%d",
            len(attrs),
            duration_ms,
            len(answer),
        )

        # 解析回复（优先 JSON）
        return self._parse_batch_answer_v2(answer, attrs, is_zh)

    def _parse_batch_answer_v2(
        self,
        answer: str,
        attrs: list[AttributeScore],
        is_zh: bool,
    ) -> tuple[list[AttributeScore], list[AttributeScore]]:
        """解析 VLM 回复（v2: JSON 数组优先 → 对象 → 按行 fallback）.

        返回 (still_missing, promoted).
        """
        still_missing: list[AttributeScore] = []
        promoted: list[AttributeScore] = []

        # ── 策略 1: 尝试 JSON 解析（v2 支持数组 + 对象） ──
        json_data = _parse_vlm_json(answer)

        # 单对象: {"verdict": "YES", ...}
        if isinstance(json_data, dict):
            verdict = _extract_verdict_json(json_data)
            if verdict is True:
                promoted.append(AttributeScore(
                    attribute=attrs[0].attribute, score=VLM_PROMOTED_SCORE,
                ))
            elif verdict is False:
                still_missing.append(attrs[0])
            else:
                logger.warning(
                    "vlm_verifier.parse_v2.warning | attr=%s cannot_extract_verdict "
                    "json=%s",
                    attrs[0].attribute,
                    json_data,
                )
                still_missing.append(attrs[0])
            return still_missing, promoted

        # JSON 数组: [{"index":1,"verdict":"YES",...}, ...]
        if isinstance(json_data, list):
            verdict_map: dict[int, bool] = {}
            for item in json_data:
                if isinstance(item, dict):
                    idx = item.get("index", item.get("id", -1))
                    v = _extract_verdict_json(item)
                    if isinstance(idx, int) and 1 <= idx <= len(attrs) and v is not None:
                        verdict_map[idx] = v

            for j, attr in enumerate(attrs, 1):
                v = verdict_map.get(j)
                if v is True:
                    promoted.append(AttributeScore(
                        attribute=attr.attribute, score=VLM_PROMOTED_SCORE,
                    ))
                elif v is False:
                    still_missing.append(attr)
                else:
                    logger.warning(
                        "vlm_verifier.parse_v2.warning | attr=%s idx=%d "
                        "missing_in_json_array",
                        attr.attribute,
                        j,
                    )
                    still_missing.append(attr)
            return still_missing, promoted

        # ── 策略 2: fallback 按行解析（兼容旧版自由文本输出） ──
        return self._parse_batch_answer_fallback(answer, attrs, is_zh)

    def _parse_batch_answer_fallback(
        self,
        answer: str,
        attrs: list[AttributeScore],
        is_zh: bool,
    ) -> tuple[list[AttributeScore], list[AttributeScore]]:
        """旧版按行解析 fallback（当 JSON 解析失败时使用）."""
        still_missing: list[AttributeScore] = []
        promoted: list[AttributeScore] = []

        if len(attrs) == 1:
            verdict = _extract_yes_no(answer)
            if verdict is True:
                promoted.append(AttributeScore(
                    attribute=attrs[0].attribute, score=VLM_PROMOTED_SCORE,
                ))
            elif verdict is False:
                still_missing.append(attrs[0])
            else:
                logger.warning(
                    "vlm_verifier.parse_fallback.warning | attr=%s cannot_extract "
                    "raw_answer=%s",
                    attrs[0].attribute,
                    answer[:200],
                )
                still_missing.append(attrs[0])
            return still_missing, promoted

        # 多属性：按行解析
        lines = answer.strip().split("\n")
        verdicts: dict[int, Optional[bool]] = {}

        for line in lines:
            line = line.strip()
            for j in range(1, len(attrs) + 1):
                for prefix in [f"{j}.", f"{j})", f"{j}、", f"{j} ", f"{j}"]:
                    if line.startswith(prefix):
                        rest = line[len(prefix):].strip()
                        verdicts[j] = _extract_yes_no(rest)
                        break
                if j in verdicts:
                    break

        for j, attr in enumerate(attrs, 1):
            v = verdicts.get(j)
            if v is True:
                promoted.append(AttributeScore(
                    attribute=attr.attribute, score=VLM_PROMOTED_SCORE,
                ))
            elif v is False:
                still_missing.append(attr)
            else:
                logger.warning(
                    "vlm_verifier.parse_fallback.warning | attr=%s (idx=%d) "
                    "cannot_extract raw_answer_lines=%s",
                    attr.attribute,
                    j,
                    answer[:300],
                )
                still_missing.append(attr)

        return still_missing, promoted

    def _parse_batch_answer(
        self,
        answer: str,
        attrs: list[AttributeScore],
        is_zh: bool,
    ) -> tuple[list[AttributeScore], list[AttributeScore]]:
        """向后兼容包装：使用 v2 解析器."""
        return self._parse_batch_answer_v2(answer, attrs, is_zh)

    @staticmethod
    def _extract_yes_no(text: str) -> Optional[bool]:
        """向后兼容包装."""
        return _extract_yes_no(text)
