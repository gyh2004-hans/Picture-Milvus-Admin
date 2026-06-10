"""属性级成功模式库 —— 从迭代历史中提取"修复属性 X 的最有效策略".

NCLB 风格: 跨迭代积累成功模式，当新 prompt 缺失某属性时直接注入已验证的修复策略.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class PatternLibrary:
    """属性级 prompt 优化模式库.

    从多轮迭代中收集 (属性, 旧prompt, 新prompt, score_delta) 四元组，
    按 Δscore 排序，取 top-K 作为该属性的已验证修复策略。
    """

    def __init__(self, max_patterns_per_attr: int = 10) -> None:
        # {attribute_name: [(old_snippet, new_snippet, delta), ...]}
        self._patterns: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        self._max_per_attr = max_patterns_per_attr

    def record(
        self,
        attribute: str,
        old_prompt: str,
        new_prompt: str,
        score_delta: float,
    ) -> None:
        """记录一次属性改善.

        Args:
            attribute: 属性名.
            old_prompt: 修复前的 prompt.
            new_prompt: 修复后的 prompt.
            score_delta: 属性分变化（正值=改善）.
        """
        if score_delta <= 0:
            return  # 只记录正向改善

        patterns = self._patterns[attribute]
        patterns.append((
            _extract_snippet(old_prompt, attribute),
            _extract_snippet(new_prompt, attribute),
            score_delta,
        ))
        # 按 delta 降序，保留 top-K
        patterns.sort(key=lambda x: x[2], reverse=True)
        if len(patterns) > self._max_per_attr:
            self._patterns[attribute] = patterns[: self._max_per_attr]

        logger.info(
            "PatternLibrary.record | attr=%s delta=+%.3f total=%d",
            attribute,
            score_delta,
            len(self._patterns[attribute]),
        )

    def get_top_patterns(
        self, attribute: str, top_k: int = 3
    ) -> list[dict]:
        """获取某属性的 top-K 成功修复模式.

        Returns:
            [{attribute, old_snippet, new_snippet, delta}, ...]
        """
        patterns = self._patterns.get(attribute, [])
        return [
            {
                "attribute": attribute,
                "old_snippet": old,
                "new_snippet": new,
                "delta": round(delta, 4),
            }
            for old, new, delta in patterns[:top_k]
        ]

    def get_patterns_for_attributes(
        self, attributes: list[str], top_k: int = 2
    ) -> list[dict]:
        """批量获取多个属性的成功模式.

        Returns:
            去重合并后的模式列表.
        """
        results: list[dict] = []
        seen: set[str] = set()
        for attr in attributes:
            for p in self.get_top_patterns(attr, top_k=top_k):
                key = p["new_snippet"]
                if key not in seen:
                    seen.add(key)
                    results.append(p)
        return results

    def format_as_hints(self, attributes: list[str], top_k: int = 2) -> str:
        """将成功模式格式化为 LLM 可用的提示文本.

        Returns:
            格式化的模式提示字符串，可直接注入 refiner user message.
        """
        patterns = self.get_patterns_for_attributes(attributes, top_k=top_k)
        if not patterns:
            return ""

        lines = ["", "Proven fix strategies (from past successful iterations):"]
        for i, p in enumerate(patterns, 1):
            lines.append(
                f"  {i}. To strengthen '{p['attribute']}' (Δ=+{p['delta']:.2f}): "
                f"use phrasing like \"{p['new_snippet'][:100]}\""
            )
        return "\n".join(lines)

    @property
    def attribute_count(self) -> int:
        return len(self._patterns)

    @property
    def total_patterns(self) -> int:
        return sum(len(v) for v in self._patterns.values())

    def clear(self) -> None:
        """清空模式库."""
        self._patterns.clear()


def _extract_snippet(prompt: str, attribute: str, window: int = 15) -> str:
    """从 prompt 中提取与属性相关的上下文片段.

    简单策略: 找到属性词在 prompt 中的位置，截取前后 window 个字符.
    """
    idx = prompt.lower().find(attribute.lower())
    if idx >= 0:
        start = max(0, idx - window)
        end = min(len(prompt), idx + len(attribute) + window)
        snippet = prompt[start:end].strip()
        if start > 0:
            snippet = "…" + snippet
        if end < len(prompt):
            snippet += "…"
        return snippet
    # 属性词不在 prompt 中，返回 prompt 的前后段
    return prompt[: window * 2].strip()
