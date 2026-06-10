"""Phase 2 单测 —— VLMEvaluator + PromptRefiner + LLMAdjuster + Pipeline.

注意：VLM API 调用和 LLM API 调用需要真实环境，此处重点测试纯逻辑.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════
# VLMEvaluator 纯逻辑测试
# ═══════════════════════════════════════════════════════════


class TestVLMEvaluator:
    """测试 VLM 评测结果的 JSON 解析逻辑."""

    def test_parse_valid_eval_json(self):
        from src.evaluate.vlm_evaluator import _parse_eval_json

        answer = '''```json
{
  "overall_score": 0.82,
  "dimension_scores": [
    {"dimension": "Subject Consistency", "score": 0.85, "comment": "good"},
    {"dimension": "Attribute Consistency", "score": 0.80, "comment": "ok"},
    {"dimension": "Spatial Consistency", "score": 0.75, "comment": "minor issues"},
    {"dimension": "Scene Completeness", "score": 0.88, "comment": "complete"},
    {"dimension": "Overall Semantic Match", "score": 0.82, "comment": "matches well"}
  ],
  "issues": ["color mismatch"],
  "missing_elements": ["arrow label"],
  "suggestions": ["add more contrast"]
}
```'''
        result = _parse_eval_json(answer)
        assert result is not None
        assert result["overall_score"] == 0.82
        assert len(result["dimension_scores"]) == 5
        assert result["dimension_scores"][0]["dimension"] == "Subject Consistency"
        assert result["issues"] == ["color mismatch"]
        assert result["missing_elements"] == ["arrow label"]
        assert result["suggestions"] == ["add more contrast"]

    def test_parse_json_without_fence(self):
        from src.evaluate.vlm_evaluator import _parse_eval_json

        answer = '{"overall_score": 0.90, "dimension_scores": [], "issues": [], "missing_elements": [], "suggestions": []}'
        result = _parse_eval_json(answer)
        assert result is not None
        assert result["overall_score"] == 0.90

    def test_parse_invalid_json_returns_none(self):
        from src.evaluate.vlm_evaluator import _parse_eval_json

        result = _parse_eval_json("this is not json at all")
        assert result is None

    def test_image_to_base64(self):
        from src.evaluate.vlm_evaluator import _image_to_base64
        import tempfile
        from pathlib import Path

        # Create a minimal PNG file
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82')
        tmp.close()

        try:
            result = _image_to_base64(tmp.name)
            assert result.startswith("data:image/png;base64,")
        finally:
            Path(tmp.name).unlink()

    def test_image_not_found_raises(self):
        from src.evaluate.vlm_evaluator import _image_to_base64

        with pytest.raises(FileNotFoundError, match="图片不存在"):
            _image_to_base64("/nonexistent/path/image.png")


# ═══════════════════════════════════════════════════════════
# PromptRefiner 策略分析测试（§3.3）
# ═══════════════════════════════════════════════════════════


class TestPromptRefinerAnalyze:
    """测试策略分析逻辑."""

    def test_analyze_with_issues(self):
        from src.models.schemas import EvalResult, DimensionScore
        from src.refiner.prompt_refiner import PromptRefiner

        refiner = PromptRefiner()
        eval_result = EvalResult(
            overall_score=0.65,
            dimension_scores=[
                DimensionScore(dimension="主体对象一致性", score=0.80, comment="基本符合"),
                DimensionScore(dimension="属性一致性", score=0.50, comment="颜色不对"),
            ],
            issues=["海洋颜色不对", "缺少蒸发箭头标注"],
            missing_elements=["蒸发箭头"],
            suggestions=["增强颜色对比度"],
        )

        strategy = refiner.analyze(eval_result)

        assert strategy.strategies is not None
        assert len(strategy.strategies) > 0
        assert strategy.summary != ""
        # Check categories
        categories = {s.category for s in strategy.strategies}
        assert "missing" in categories or "attribute_error" in categories
        # Check that missing_elements generate missing strategies
        missing_targets = [s.target for s in strategy.strategies if s.category == "missing"]
        assert any("蒸发箭头" in t for t in missing_targets)

    def test_analyze_empty_result(self):
        from src.models.schemas import EvalResult
        from src.refiner.prompt_refiner import PromptRefiner

        refiner = PromptRefiner()
        eval_result = EvalResult(
            overall_score=0.95,
            dimension_scores=[],
            issues=[],
            missing_elements=[],
            suggestions=[],
        )

        strategy = refiner.analyze(eval_result)
        assert len(strategy.strategies) == 0
        assert "no optimization needed" in strategy.summary.lower()

    def test_classify_missing_issue(self):
        from src.refiner.prompt_refiner import PromptRefiner

        refiner = PromptRefiner()
        item = refiner._classify_issue("缺少红色标注文字")
        assert item is not None
        assert item.category == "missing"

    def test_classify_attribute_error(self):
        from src.refiner.prompt_refiner import PromptRefiner

        refiner = PromptRefiner()
        item = refiner._classify_issue("颜色偏差，蓝色太浅")
        assert item is not None
        assert item.category == "attribute_error"

    def test_classify_composition_issue(self):
        from src.refiner.prompt_refiner import PromptRefiner

        refiner = PromptRefiner()
        item = refiner._classify_issue("布局偏移，左右位置不对")
        assert item is not None
        assert item.category == "composition"

    def test_classify_style_issue(self):
        from src.refiner.prompt_refiner import PromptRefiner

        refiner = PromptRefiner()
        item = refiner._classify_issue("画风不统一，风格不一致")
        assert item is not None
        assert item.category == "style"

    def test_extract_target(self):
        from src.refiner.prompt_refiner import PromptRefiner

        target = PromptRefiner._extract_target("缺少红色标注文字和蓝色箭头")
        assert "红色标注文字" in target or len(target) > 0

    def test_map_dimension_to_category(self):
        from src.refiner.prompt_refiner import PromptRefiner

        assert PromptRefiner._map_dimension_to_category("主体对象一致性") == "missing"
        assert PromptRefiner._map_dimension_to_category("属性一致性") == "attribute_error"
        assert PromptRefiner._map_dimension_to_category("空间关系一致性") == "composition"
        assert PromptRefiner._map_dimension_to_category("场景完整性") == "style"
        assert PromptRefiner._map_dimension_to_category("整体语义匹配度") == "attribute_error"

    def test_select_top_feedback_limits_to_three(self):
        from src.models.schemas import EvalResult
        from src.refiner.prompt_refiner import PromptRefiner

        eval_result = EvalResult(
            overall_score=0.60,
            issues=["风格不统一", "颜色偏差", "布局偏移", "缺少标注"],
            missing_elements=["蒸发箭头", "降雨虚线"],
            suggestions=["加强对比度"],
        )
        issues, missing, suggestions = PromptRefiner.select_top_feedback(
            eval_result, max_items=3,
        )
        assert len(issues) + len(missing) + len(suggestions) == 3
        # missing 严重度最高，应优先入选
        assert "蒸发箭头" in missing or "降雨虚线" in missing

    def test_topk_strategies(self):
        from src.models.schemas import StrategyItem
        from src.refiner.prompt_refiner import PromptRefiner

        strategies = [
            StrategyItem(category="style", target="a", action="x"),
            StrategyItem(category="missing", target="b", action="y"),
            StrategyItem(category="attribute_error", target="c", action="z"),
            StrategyItem(category="composition", target="d", action="w"),
        ]
        top = PromptRefiner._topk_strategies(strategies, 3)
        assert len(top) == 3
        assert top[0].category == "missing"


# ═══════════════════════════════════════════════════════════
# LLMAdjuster 纯逻辑测试
# ═══════════════════════════════════════════════════════════


class TestLLMAdjuster:
    """测试 LLM Adjuster 的纯逻辑（不调用 API）."""

    def test_build_user_message(self):
        from src.models.schemas import StrategyAnalysis, StrategyItem
        from src.refiner.llm_adjuster import LLMAdjuster

        adjuster = LLMAdjuster()
        strategy = StrategyAnalysis(
            strategies=[
                StrategyItem(category="missing", target="箭头标注", action="增强主体强调"),
                StrategyItem(category="attribute_error", target="颜色偏差", action="强化约束描述"),
            ],
            summary="test summary",
        )

        msg = adjuster._build_user_message(
            origin_prompt="test prompt",
            strategy=strategy,
            issues=["颜色不对"],
            missing_elements=["箭头"],
            overall_score=0.70,
            eval_threshold=0.82,
        )

        assert "test prompt" in msg
        assert "0.70" in msg
        assert "颜色不对" in msg
        assert "箭头" in msg
        assert "missing" in msg
        assert "attribute_error" in msg
        assert "target: >= 0.82" in msg
        assert "MINIMAL EDIT" in msg

    def test_build_changes_summary(self):
        from src.models.schemas import StrategyAnalysis, StrategyItem
        from src.refiner.llm_adjuster import LLMAdjuster

        adjuster = LLMAdjuster()
        strategy = StrategyAnalysis(
            strategies=[
                StrategyItem(category="missing", target="x", action="增强"),
                StrategyItem(category="attribute_error", target="y", action="强化"),
            ],
            summary="test",
        )

        summary = adjuster._build_changes_summary("abc", "abcdefg", strategy)
        assert "3 → 7" in summary  # length check
        assert "Minimal edit" in summary
        assert "x" in summary
        assert "y" in summary

    def test_build_changes_summary_empty(self):
        from src.models.schemas import StrategyAnalysis
        from src.refiner.llm_adjuster import LLMAdjuster

        adjuster = LLMAdjuster()
        strategy = StrategyAnalysis(strategies=[], summary="none")

        summary = adjuster._build_changes_summary("a", "ab", strategy)
        assert "none" in summary


# ═══════════════════════════════════════════════════════════
# EvalResult 模型测试
# ═══════════════════════════════════════════════════════════


class TestEvalResult:
    def test_default_values(self):
        from src.models.schemas import EvalResult

        result = EvalResult(overall_score=0.5)
        assert result.overall_score == 0.5
        assert result.dimension_scores == []
        assert result.issues == []
        assert result.missing_elements == []
        assert result.suggestions == []

    def test_full_result(self):
        from src.models.schemas import EvalResult, DimensionScore

        result = EvalResult(
            overall_score=0.85,
            dimension_scores=[
                DimensionScore(dimension="主体对象一致性", score=0.90, comment="ok"),
            ],
            issues=["some issue"],
            missing_elements=["some element"],
            suggestions=["some suggestion"],
        )
        assert len(result.dimension_scores) == 1
        assert result.dimension_scores[0].dimension == "主体对象一致性"
        assert result.issues == ["some issue"]


class TestDimensionScore:
    def test_score_clamping(self):
        from src.models.schemas import DimensionScore

        ds = DimensionScore(dimension="test", score=0.75, comment="ok")
        assert ds.score == 0.75
        assert ds.dimension == "test"


class TestStrategyAnalysis:
    def test_empty_strategies(self):
        from src.models.schemas import StrategyAnalysis, StrategyItem

        sa = StrategyAnalysis(strategies=[], summary="nothing to do")
        assert len(sa.strategies) == 0
        assert sa.summary == "nothing to do"

    def test_with_strategies(self):
        from src.models.schemas import StrategyAnalysis, StrategyItem

        sa = StrategyAnalysis(
            strategies=[
                StrategyItem(category="missing", target="arrow", action="增强主体强调"),
            ],
            summary="1 strategy",
        )
        assert len(sa.strategies) == 1
        assert sa.strategies[0].category == "missing"
        assert sa.strategies[0].action == "增强主体强调"


# ═══════════════════════════════════════════════════════════
# Pipeline 逻辑测试（mock 所有外部依赖）
# ═══════════════════════════════════════════════════════════


class TestPipeline:
    """测试闭环控制逻辑：停止条件、迭代计数."""

    @pytest.fixture(autouse=True)
    def _register_drawers(self):
        from src.draw import DRAWER_REGISTRY

        mock_drawer = MagicMock()
        # generate 现在是 async，需返回 awaitable
        mock_drawer.generate = AsyncMock(return_value="/fake/test.png")
        DRAWER_REGISTRY["doubao"] = mock_drawer
        yield
        DRAWER_REGISTRY.clear()

    @staticmethod
    def _setup_pipeline():
        """创建已 mock CLIP/VectorStore 的 ImagePipeline，使 _run_clip_enrich 可降级到 evaluate_loop."""
        import numpy as np
        from src.pipeline import ImagePipeline

        pipeline = ImagePipeline()

        # Mock CLIP client —— encode_text 返回随机 embedding
        mock_clip = MagicMock()
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        mock_clip.encode_text.return_value = emb
        mock_clip.encode_image.return_value = emb
        pipeline._clip_client = mock_clip

        # Mock VectorStore —— 无精确匹配、无语义命中，确保降级到 evaluate_loop
        mock_store = MagicMock()
        mock_store.connect.return_value = None
        mock_store.backend_type = "local_numpy"
        mock_store.count.return_value = 0
        mock_store.find_best_by_exact_prompt.return_value = None
        mock_store.search_by_semantic.return_value = {
            "results": [],
            "query_time_ms": 1.0,
            "total_in_partition": 0,
        }
        mock_store.insert.return_value = 1
        pipeline._vector_store = mock_store

        return pipeline

    def _make_eval(self, score: float):
        from src.models.schemas import EvalResult

        return EvalResult(
            overall_score=score,
            dimension_scores=[],
            issues=[],
            missing_elements=[],
            suggestions=[],
        )

    def test_max_iterations_stop(self):
        """达到 max_iterations 时应停止."""
        from src.models.schemas import PipelineRequest

        pipeline = self._setup_pipeline()

        call_count = [0]

        def _eval_side_effect(prompt: str, image_path: str, lang: str = "zh"):
            call_count[0] += 1
            # 每轮 +0.05，但阈值 0.99，永远不达标
            return self._make_eval(0.50 + call_count[0] * 0.05)

        with patch.object(pipeline._evaluator, "evaluate", new_callable=AsyncMock, side_effect=_eval_side_effect):
            with patch.object(pipeline._adjuster, "adjust", new_callable=AsyncMock) as mock_adjust:
                mock_adjust.return_value.optimized_prompt = "refined prompt"

                req = PipelineRequest(
                    prompt="test",
                    model="doubao",
                    max_iterations=3,
                    eval_threshold=0.99,
                )
                resp = asyncio.run(pipeline.run(req))

                assert resp.total_iterations == 3
                assert resp.stopped_reason == "max_iterations"
                assert len(resp.history) == 3
                assert resp.history[0].overall_score == 0.55
                assert resp.history[1].overall_score == 0.60
                assert resp.history[2].overall_score == 0.65

    def test_threshold_met_stops_early(self):
        """综合分达标时应立即停止."""
        from src.models.schemas import EvalResult, PipelineRequest

        pipeline = self._setup_pipeline()

        with patch.object(pipeline._evaluator, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = EvalResult(
                overall_score=0.90,
                dimension_scores=[],
                issues=[],
                missing_elements=[],
                suggestions=[],
            )

            req = PipelineRequest(
                prompt="test",
                model="doubao",
                max_iterations=5,
                eval_threshold=0.82,
            )
            resp = asyncio.run(pipeline.run(req))

            assert resp.total_iterations == 1
            assert resp.stopped_reason == "threshold_met"
            assert resp.final_score >= 0.82

    def test_score_regression_rollback(self):
        """分数较历史最佳下降超过 SCORE_ROLLBACK_DELTA 时应回滚并停止."""
        from src.models.schemas import EvalResult, PipelineRequest

        pipeline = self._setup_pipeline()

        mock_scores = [
            EvalResult(
                overall_score=0.75,
                dimension_scores=[],
                issues=["issue1"],
                missing_elements=[],
                suggestions=[],
            ),
            EvalResult(
                overall_score=0.65,  # 下降 0.10 > 0.05
                dimension_scores=[],
                issues=["issue2"],
                missing_elements=[],
                suggestions=[],
            ),
        ]

        with patch.object(pipeline._evaluator, "evaluate", new_callable=AsyncMock, side_effect=mock_scores):
            with patch.object(pipeline._adjuster, "adjust", new_callable=AsyncMock) as mock_adjust:
                mock_adjust.return_value.optimized_prompt = "refined worse"

                req = PipelineRequest(
                    prompt="original prompt",
                    model="doubao",
                    max_iterations=5,
                    eval_threshold=0.82,
                )
                resp = asyncio.run(pipeline.run(req))

                assert resp.total_iterations == 2
                assert resp.stopped_reason == "score_regression"
                assert resp.final_score == 0.75
                assert resp.final_prompt == "original prompt"

    def test_convergence_stop(self):
        """分数不再提升（delta < 0.01）时应因收敛停止."""
        from src.models.schemas import EvalResult, PipelineRequest

        pipeline = self._setup_pipeline()

        mock_scores = [
            EvalResult(overall_score=0.70, dimension_scores=[], issues=[], missing_elements=[], suggestions=[]),
            EvalResult(overall_score=0.705, dimension_scores=[], issues=[], missing_elements=[], suggestions=[]),  # delta=0.005 < 0.01
        ]

        with patch.object(pipeline._evaluator, "evaluate", new_callable=AsyncMock, side_effect=mock_scores):
            with patch.object(pipeline._adjuster, "adjust", new_callable=AsyncMock) as mock_adjust:
                mock_adjust.return_value.optimized_prompt = "refined"

                req = PipelineRequest(
                    prompt="test",
                    model="doubao",
                    max_iterations=5,
                    eval_threshold=0.82,
                )
                resp = asyncio.run(pipeline.run(req))

                assert resp.total_iterations == 2
                assert resp.stopped_reason == "converged"

    def test_unknown_model_raises(self):
        from src.models.schemas import PipelineRequest

        pipeline = self._setup_pipeline()
        req = PipelineRequest(prompt="test", model="nonexistent", max_iterations=3)
        with pytest.raises(ValueError, match="Unknown model"):
            asyncio.run(pipeline.run(req))

    def test_history_is_cumulative(self):
        from src.models.schemas import EvalResult, PipelineRequest

        pipeline = self._setup_pipeline()

        with patch.object(pipeline._evaluator, "evaluate", new_callable=AsyncMock) as mock_eval:
            mock_eval.return_value = EvalResult(
                overall_score=0.90,
                dimension_scores=[],
                issues=[],
                missing_elements=[],
                suggestions=[],
            )

            req = PipelineRequest(
                prompt="test", model="doubao",
            )
            resp = asyncio.run(pipeline.run(req))

            assert len(resp.history) == 1
            assert resp.history[0].iteration == 1
            assert resp.history[0].image_path == "/fake/test.png"
            assert resp.history[0].overall_score == 0.90


# ═══════════════════════════════════════════════════════════
# llm_utils 测试（保留）
# ═══════════════════════════════════════════════════════════


class TestParseJsonFromLLM:
    def test_clean_json_array(self):
        from src.llm_utils import parse_json_from_llm

        result = parse_json_from_llm('["a", "b", "c"]')
        assert result == ["a", "b", "c"]

    def test_json_with_markdown_fence(self):
        from src.llm_utils import parse_json_from_llm

        raw = 'Here is the result:\n```json\n["red", "blue"]\n```\nDone.'
        result = parse_json_from_llm(raw)
        assert result == ["red", "blue"]

    def test_json_with_extra_text(self):
        from src.llm_utils import parse_json_from_llm

        raw = 'Sure! the attributes are ["sunlight", "shadow"] for this image.'
        result = parse_json_from_llm(raw)
        assert result == ["sunlight", "shadow"]

    def test_invalid_json_raises(self):
        from src.llm_utils import parse_json_from_llm

        with pytest.raises(ValueError, match="LLM 输出"):
            parse_json_from_llm("this is not json at all")
