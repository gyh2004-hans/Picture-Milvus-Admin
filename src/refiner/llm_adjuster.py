"""LLM 调整模块 —— 对齐项目计划书 §3.4.

从旧 prompt_refiner.py 中吸收 LLM 调用逻辑，独立为 LLMAdjuster 类。
职责：接收策略分析结果（StrategyAnalysis），执行**最小编辑**式 prompt 修正。

与 PromptRefiner (§3.3 策略分析) 的关系：
  - PromptRefiner.analyze() → 输出 StrategyAnalysis（问题分类 → 优化策略映射）
  - LLMAdjuster.adjust() → 接收策略，调用 LLM 对当前 prompt 做局部修补
"""
from __future__ import annotations

import logging
import time

from src.config import EVAL_THRESHOLD
from src.llm_utils import chat_completion
from src.models.schemas import RefinerOutput, StrategyAnalysis

logger = logging.getLogger(__name__)

# ── LLM Adjuster system prompt（最小编辑模式） ──

ADJUSTER_SYSTEM = """\
You are a prompt engineer for text-to-image models. \
Your task is to apply MINIMAL LOCAL EDITS to an image generation prompt \
based on a structured evaluation and a strategy analysis.

The evaluation tells you:
- which visual elements are MISSING from the image
- which attributes have errors (color, shape, position, etc.)
- which spatial relationships are incorrect
- the overall scene completeness and semantic match

The strategy analysis tells you WHAT to fix and HOW:
- Missing elements → strengthen subject emphasis
- Attribute errors → reinforce color/shape constraints
- Compositional issues → add positional/layout descriptions
- Style inconsistencies → add style constraint keywords

CRITICAL — MINIMAL EDIT MODE (must follow):
1. **Do NOT rewrite** the prompt from scratch. **Do NOT** restructure, reorder, \
or paraphrase existing sentences.
2. **Preserve ≥90% of the original text verbatim.** Only insert short phrases or \
clauses (≤15 words each) to fix the listed issues.
3. **Address ONLY the issues/strategies provided** (at most 3). Ignore everything else.
4. **Anchor to the current prompt** — Every concept, object, and relationship in the \
original MUST remain. Only ADD visual detail; never REMOVE or REPLACE existing attributes.
5. Fix attribute errors by reinforcing specific constraints (color, size, shape, material).
6. Correct spatial/compositional issues by adding clear positional descriptions inline.
7. The refined prompt MUST be in the SAME LANGUAGE as the original prompt.
8. Output a single paragraph of natural prose.
9. Do NOT add entirely new concepts that weren't in the original prompt.
10. Return ONLY the refined prompt text, no commentary.
"""


class LLMAdjuster:
    """LLM 驱动 Prompt 调整器（项目计划书 §3.4）.

    接收 PromptRefiner 的策略分析结果，对当前 prompt 做最小编辑式局部修正。
    """

    def __init__(self) -> None:
        pass

    async def adjust(
        self,
        origin_prompt: str,
        strategy: StrategyAnalysis,
        issues: list[str] | None = None,
        missing_elements: list[str] | None = None,
        overall_score: float | None = None,
        eval_threshold: float | None = None,
    ) -> RefinerOutput:
        """根据策略分析结果对当前 prompt 做最小编辑.

        Args:
            origin_prompt: 当前轮 prompt（局部修补的基底，不可整段重写）.
            strategy: PromptRefiner 输出的策略分析结果（已 top-k 截断）.
            issues: 评测发现的问题列表（可选，已 top-k 截断）.
            missing_elements: 评测发现的缺失元素列表（可选，已 top-k 截断）.
            overall_score: 综合得分（可选，用于全局反馈）.
            eval_threshold: 达标阈值（可选，默认读 config）.

        Returns:
            RefinerOutput: 优化后的 prompt + 修改说明.

        Raises:
            RuntimeError: LLM 调用失败时抛出.
        """
        t0 = time.perf_counter()
        threshold = eval_threshold if eval_threshold is not None else EVAL_THRESHOLD
        logger.info(
            "llm_adjuster.adjust.start | prompt_len=%d strategies=%d score=%.3f "
            "issues=%d missing=%d minimal_edit=True",
            len(origin_prompt),
            len(strategy.strategies),
            overall_score if overall_score is not None else -1,
            len(issues or []),
            len(missing_elements or []),
        )

        user_message = self._build_user_message(
            origin_prompt=origin_prompt,
            strategy=strategy,
            issues=issues,
            missing_elements=missing_elements,
            overall_score=overall_score,
            eval_threshold=threshold,
        )

        try:
            optimized = await chat_completion(
                system=ADJUSTER_SYSTEM,
                user=user_message,
                temperature=0.3,
                max_tokens=1024,
            )
        except (RuntimeError, ValueError) as exc:
            duration_ms = int((time.perf_counter() - t0) * 1000)
            logger.error(
                "llm_adjuster.adjust.error | prompt_len=%d duration_ms=%d error=%s",
                len(origin_prompt),
                duration_ms,
                exc,
            )
            raise

        optimized = optimized.strip()
        changes_summary = self._build_changes_summary(
            origin_prompt, optimized, strategy,
        )

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "llm_adjuster.adjust.end | prompt_len=%d→%d duration_ms=%d",
            len(origin_prompt),
            len(optimized),
            duration_ms,
        )

        return RefinerOutput(
            optimized_prompt=optimized,
            changes_summary=changes_summary,
        )

    def _build_user_message(
        self,
        origin_prompt: str,
        strategy: StrategyAnalysis,
        issues: list[str] | None = None,
        missing_elements: list[str] | None = None,
        overall_score: float | None = None,
        eval_threshold: float = EVAL_THRESHOLD,
    ) -> str:
        """构建发送给 LLM 的 user message."""
        lines = [
            "MODE: MINIMAL EDIT — patch the prompt locally; do NOT rewrite.\n",
            f"Current prompt (edit this in place):\n{origin_prompt}\n",
        ]

        if overall_score is not None:
            lines.append(
                f"Overall score: {overall_score:.3f} (target: >= {eval_threshold:.2f})"
            )
            lines.append("")

        if issues:
            lines.append(f"Issues to fix (top {len(issues)}, address ONLY these):")
            for issue in issues:
                lines.append(f"  ❌ {issue}")
            lines.append("")

        if missing_elements:
            lines.append(f"Missing elements (top {len(missing_elements)}):")
            for elem in missing_elements:
                lines.append(f"  🔍 {elem}")
            lines.append("")

        if strategy.strategies:
            lines.append(f"Optimization strategies (top {len(strategy.strategies)}):")
            for s in strategy.strategies:
                lines.append(
                    f"  [{s.category}] {s.target} → {s.action}"
                )
            lines.append("")

        if strategy.summary:
            lines.append(f"Strategy summary: {strategy.summary}")
            lines.append("")

        lines.append(
            "Apply MINIMAL LOCAL EDITS only: insert short phrases/clauses to fix "
            "the issues above. Preserve ≥90% of the current prompt verbatim. "
            "Do NOT rewrite or restructure. Output ONLY the refined prompt text."
        )
        return "\n".join(lines)

    def _build_changes_summary(
        self,
        original: str,
        optimized: str,
        strategy: StrategyAnalysis,
    ) -> str:
        """生成修改摘要."""
        targets = [s.target for s in strategy.strategies]
        fixed = ", ".join(targets) if targets else "none"
        return (
            f"Minimal edit: {len(original)} → {len(optimized)} chars. "
            f"Targeted ({len(targets)}): {fixed}"
        )
