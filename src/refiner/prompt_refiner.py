"""Prompt Refiner 策略分析模块 —— 对齐项目计划书 §3.3.

纯策略分析，不包含 LLM 调用（LLM 调用已拆分到 llm_adjuster.py §3.4）。

职责：
  接收 VLM 评测结果（EvalResult），分析问题类型，映射为优化策略。

问题类型与优化策略映射（§3.3 表格）:
  - 缺失对象/元素 → 增强主体强调
  - 属性错误（颜色/大小/形状/数量） → 强化约束描述
  - 构图/空间偏差 → 增加位置关系描述
  - 风格不一致 → 补充风格限制词
"""
from __future__ import annotations

import logging
import re

from src.config import MAX_ISSUES_PER_ROUND
from src.models.schemas import EvalResult, StrategyAnalysis, StrategyItem

logger = logging.getLogger(__name__)

# ── 问题分类关键词 ──────────────────────────────────────

# 缺失类关键词（元素/对象在图中不存在）
_MISSING_KEYWORDS_ZH = [
    "缺失", "缺少", "没有", "不存在", "看不到", "未出现",
    "遗漏", "丢失", "不见", "消失", "完全没有",
]

# 属性错误类关键词（颜色/大小/形状/数量不对）
_ATTRIBUTE_ERROR_KEYWORDS_ZH = [
    "颜色", "大小", "形状", "数量", "尺寸", "比例",
    "材质", "纹理", "粗细", "深浅", "明暗",
    "不对", "错误", "不正确", "不一致", "不符",
    "偏差", "不准确", "不匹配",
]

# 构图/空间偏差类关键词
_COMPOSITION_KEYWORDS_ZH = [
    "位置", "方向", "上下", "左右", "前后", "排列",
    "布局", "构图", "对齐", "偏移", "错位",
    "空间关系", "距离", "相对位置",
]

# 风格不一致类关键词
_STYLE_KEYWORDS_ZH = [
    "风格", "画风", "色调", "氛围", "质感",
    "不一致", "不符合", "不统一",
]

# ── 类别 → 优化动作映射 ─────────────────────────────────

CATEGORY_ACTION_MAP: dict[str, str] = {
    "missing": "增强主体强调：在 prompt 中添加更具体的主体描述词，使用强调语气",
    "attribute_error": "强化属性约束：明确指定颜色/大小/形状/数量等精确属性值",
    "composition": "增加位置关系描述：使用方位词和布局说明明确对象空间关系",
    "style": "补充风格限制词：添加画风/色调/质感等风格关键词以约束生成方向",
}

# 严重度排序（数值越小越严重，优先修复）
CATEGORY_SEVERITY: dict[str, int] = {
    "missing": 0,
    "composition": 1,
    "attribute_error": 2,
    "style": 3,
}


class PromptRefiner:
    """策略分析器（项目计划书 §3.3）.

    仅做问题分类和策略映射，不调用 LLM。
    LLM 调用由 LLMAdjuster (§3.4) 负责。
    """

    def __init__(self) -> None:
        pass

    def analyze(
        self,
        eval_result: EvalResult,
        origin_prompt: str = "",
    ) -> StrategyAnalysis:
        """分析评测结果，输出优化策略列表.

        按计划书 §3.3 表格规则：
        - 遍历 issues + missing_elements + suggestions
        - 对每个问题分类：missing / attribute_error / composition / style
        - 映射为对应的优化动作

        Args:
            eval_result: VLM 五维度评测结果.
            origin_prompt: 原始 prompt（用于上下文分析，可选）.

        Returns:
            StrategyAnalysis: 含策略列表 + 总结.
        """
        strategies: list[StrategyItem] = []

        # ── 从 issues 分类 ──
        for issue in eval_result.issues:
            item = self._classify_issue(issue)
            if item:
                strategies.append(item)

        # ── 从 missing_elements 生成缺失策略 ──
        for elem in eval_result.missing_elements:
            # 检查是否已有同名策略
            if not any(s.target == elem and s.category == "missing" for s in strategies):
                strategies.append(StrategyItem(
                    category="missing",
                    target=elem,
                    action=CATEGORY_ACTION_MAP["missing"],
                ))

        # ── 从 suggestions 补充 ──
        for suggestion in eval_result.suggestions:
            item = self._classify_issue(suggestion)
            if item:
                # 避免重复（相同 target + category）
                if not any(
                    s.target == item.target and s.category == item.category
                    for s in strategies
                ):
                    strategies.append(item)

        # ── 从维度分中分析薄弱维度 ──
        for dim in eval_result.dimension_scores:
            if dim.score < 0.6:
                strategies.append(StrategyItem(
                    category=self._map_dimension_to_category(dim.dimension),
                    target=dim.dimension,
                    action=CATEGORY_ACTION_MAP.get(
                        self._map_dimension_to_category(dim.dimension),
                        "增强描述细节",
                    ),
                ))

        # ── 生成总结 ──
        strategies = self._topk_strategies(strategies, MAX_ISSUES_PER_ROUND)
        summary = self._build_summary(strategies, eval_result.overall_score)

        logger.info(
            "prompt_refiner.analyze | overall=%.3f strategies=%d categories=%s",
            eval_result.overall_score,
            len(strategies),
            list(set(s.category for s in strategies)),
        )

        return StrategyAnalysis(
            strategies=strategies,
            summary=summary,
        )

    @staticmethod
    def select_top_feedback(
        eval_result: EvalResult,
        max_items: int = MAX_ISSUES_PER_ROUND,
    ) -> tuple[list[str], list[str], list[str]]:
        """从评测反馈中选取 top-k 最严重的问题（issues / missing / suggestions 共享配额）."""
        ranked: list[tuple[int, int, str, str]] = []  # severity, order, text, kind

        for i, issue in enumerate(eval_result.issues):
            item = PromptRefiner._classify_issue(issue)
            sev = CATEGORY_SEVERITY.get(item.category if item else "attribute_error", 2)
            ranked.append((sev, i, issue, "issue"))

        for i, elem in enumerate(eval_result.missing_elements):
            ranked.append((CATEGORY_SEVERITY["missing"], i, elem, "missing"))

        for i, sug in enumerate(eval_result.suggestions):
            item = PromptRefiner._classify_issue(sug)
            sev = CATEGORY_SEVERITY.get(item.category if item else "attribute_error", 2)
            ranked.append((sev + 1, i, sug, "suggestion"))

        ranked.sort(key=lambda x: (x[0], x[1]))
        top = ranked[:max_items]

        issues = [text for _, _, text, kind in top if kind == "issue"]
        missing = [text for _, _, text, kind in top if kind == "missing"]
        suggestions = [text for _, _, text, kind in top if kind == "suggestion"]
        return issues, missing, suggestions

    @staticmethod
    def _topk_strategies(
        strategies: list[StrategyItem],
        max_items: int,
    ) -> list[StrategyItem]:
        """按严重度保留 top-k 策略."""
        if len(strategies) <= max_items:
            return strategies
        ranked = sorted(
            enumerate(strategies),
            key=lambda pair: (
                CATEGORY_SEVERITY.get(pair[1].category, 2),
                pair[0],
            ),
        )
        return [strategies[i] for i, _ in ranked[:max_items]]

    @staticmethod
    def _classify_issue(text: str) -> StrategyItem | None:
        """将单个问题文本分类为策略项.

        检查关键词，映射到四类问题之一。
        优先级: 缺失 > 构图 > 风格 > 属性错误（构图优先于属性，避免"位置不对"被误判为属性错误）.
        """
        if not text or not text.strip():
            return None

        # ── 缺失类（最高优先级） ──
        if any(kw in text for kw in _MISSING_KEYWORDS_ZH):
            return StrategyItem(
                category="missing",
                target=PromptRefiner._extract_target(text),
                action=CATEGORY_ACTION_MAP["missing"],
            )

        # ── 构图/空间偏差（优先于属性错误，避免"位置不对"误判） ──
        if any(kw in text for kw in _COMPOSITION_KEYWORDS_ZH):
            return StrategyItem(
                category="composition",
                target=PromptRefiner._extract_target(text),
                action=CATEGORY_ACTION_MAP["composition"],
            )

        # ── 风格不一致 ──
        if any(kw in text for kw in _STYLE_KEYWORDS_ZH):
            return StrategyItem(
                category="style",
                target=PromptRefiner._extract_target(text),
                action=CATEGORY_ACTION_MAP["style"],
            )

        # ── 属性错误（检查颜色/大小/形状/数量等） ──
        if any(kw in text for kw in _ATTRIBUTE_ERROR_KEYWORDS_ZH):
            return StrategyItem(
                category="attribute_error",
                target=PromptRefiner._extract_target(text),
                action=CATEGORY_ACTION_MAP["attribute_error"],
            )

        # ── 兜底：无法分类的归为 attribute_error ──
        return StrategyItem(
            category="attribute_error",
            target=PromptRefiner._extract_target(text),
            action="增强描述细节",
        )

    @staticmethod
    def _extract_target(text: str) -> str:
        """从问题文本中提取目标元素名（截取前30字作为描述）."""
        # 去除常见前缀词
        cleaned = re.sub(
            r"^(缺少|缺失|没有|不存在|看不到|问题[:：]|需要|建议|应该|可以)",
            "",
            text.strip(),
        )
        # 截取前 30 字作为 target
        if len(cleaned) > 30:
            cleaned = cleaned[:30] + "..."
        return cleaned or text.strip()[:30]

    @staticmethod
    def _map_dimension_to_category(dimension: str) -> str:
        """将维度名称映射到问题分类."""
        mapping = {
            "主体对象一致性": "missing",
            "Subject Consistency": "missing",
            "属性一致性": "attribute_error",
            "Attribute Consistency": "attribute_error",
            "空间关系一致性": "composition",
            "Spatial Consistency": "composition",
            "场景完整性": "style",
            "Scene Completeness": "style",
            "整体语义匹配度": "attribute_error",
            "Overall Semantic Match": "attribute_error",
        }
        return mapping.get(dimension, "attribute_error")

    def _build_summary(
        self,
        strategies: list[StrategyItem],
        overall_score: float,
    ) -> str:
        """生成策略分析总结."""
        if not strategies:
            return f"All dimensions pass (overall={overall_score:.2f}), no optimization needed."

        categories = set(s.category for s in strategies)
        cat_names = {
            "missing": "缺失元素",
            "attribute_error": "属性错误",
            "composition": "构图偏差",
            "style": "风格不一致",
        }
        cat_str = "、".join(cat_names.get(c, c) for c in categories)

        lines = [
            f"综合得分 {overall_score:.2f}，识别到 {len(strategies)} 个优化项",
            f"涉及问题类型: {cat_str}",
        ]

        # 列出关键策略
        for s in strategies[:5]:
            lines.append(f"  - [{cat_names.get(s.category, s.category)}] {s.target}")
        if len(strategies) > 5:
            lines.append(f"  ... 及其他 {len(strategies) - 5} 项")

        return "；".join(lines)
