"""全链路集成测试 —— Draw → Evaluate → Milvus 端到端验证.

覆盖:
  - clip_enrich 完整链路: prompt → CLIP encode → VectorStore search
    → Draw → Evaluate loop → CLIP encode image → VectorStore insert
    → search 验证数据已入库
  - 分区路由验证: pipeline 中 subject 正确路由到分区
  - 跨学科分区隔离: 地理 prompt 不应召回数学图片

运行:
  pytest tests/test_pipeline_integration.py -v -m integration
  pytest tests/test_pipeline_integration.py -v -m "not slow"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

DIM = 512
pytestmark = [pytest.mark.integration]


# ══════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════

@pytest.fixture
def mock_drawer():
    """Mock Drawer — 返回假图片路径."""
    drawer = AsyncMock()
    drawer.generate.return_value = "/tmp/test_output.png"
    return drawer


@pytest.fixture
def mock_evaluator():
    """Mock VLMEvaluator — 返回高分评价."""
    from src.models.schemas import DimensionScore, EvalResult
    evaluator = AsyncMock()
    evaluator.evaluate.return_value = EvalResult(
        overall_score=0.92,
        dimension_scores=[
            DimensionScore(dimension="主体对象一致性", score=0.95, comment="good"),
            DimensionScore(dimension="属性一致性", score=0.90, comment="good"),
            DimensionScore(dimension="空间关系一致性", score=0.92, comment="good"),
            DimensionScore(dimension="场景完整性", score=0.91, comment="good"),
            DimensionScore(dimension="整体语义匹配度", score=0.93, comment="good"),
        ],
        issues=[],
        missing_elements=[],
        suggestions=[],
    )
    return evaluator


@pytest.fixture
def mock_clip_client():
    """Mock LocalCLIPClient — 返回随机 embedding."""
    client = MagicMock()
    emb = np.random.randn(DIM).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    client.encode_text.return_value = emb
    client.encode_image.return_value = emb
    client.embedding_dim = DIM
    return client


@pytest.fixture
def vector_store_setup(monkeypatch):
    """设置 VectorStore 使用 LocalNumpyBackend（避免依赖 Milvus/Docker）."""
    from src.milvus import vector_store as vs_module

    # 重置单例（测试隔离）
    vs_module._reset_vector_store_instance()

    # Patch connect() 直接设置 LocalNumpyBackend
    def _fake_connect(self_: vs_module.VectorStore) -> None:
        backend = vs_module.LocalNumpyBackend()
        backend.connect()
        self_._backend = backend
        self_._backend_label = "local_numpy"

    monkeypatch.setattr(vs_module.VectorStore, "connect", _fake_connect)

    from src.milvus.vector_store import get_vector_store
    store = get_vector_store()
    store.connect()
    yield store
    store.drop_all()
    vs_module._reset_vector_store_instance()


# ══════════════════════════════════════════════════════════
# 集成测试
# ══════════════════════════════════════════════════════════

class TestPartitionRoutingInPipeline:
    """验证 pipeline 中 subject 正确路由到分区."""

    def test_insert_with_subject_routes_to_correct_partition(
        self, vector_store_setup, mock_clip_client,
    ):
        """insert 传入地理 subject → 记录在地理分区."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        emb = np.random.randn(DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        record = ImageRecord(
            image_id=0,
            prompt="地理测试 prompt",
            optimized_prompt="地理优化 prompt",
            score=0.92,
            image_path="/test/geo.png",
            embedding=emb.tolist(),
            subject="geography",
            tags=["test"],
        )
        store.insert(record, subject="geography")

        assert store.count(subject="geography") == 1
        assert store.count(subject="math") == 0

    def test_subject_routing_default(self, vector_store_setup):
        """无 subject → _default 分区."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        emb = np.random.randn(DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        record = ImageRecord(
            image_id=0,
            prompt="无学科 prompt",
            optimized_prompt="opt",
            score=0.8,
            image_path="/test/no_subj.png",
            embedding=emb.tolist(),
            subject=None,
        )
        store.insert(record)
        stats = store.get_stats_by_subject()
        assert stats["_default"] == 1

    def test_multiple_subjects_routing(self, vector_store_setup):
        """多个学科分别路由到对应分区."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        subjects = ["geography", "math", "physics", "geography", "math"]

        for i, subj in enumerate(subjects):
            emb = np.random.randn(DIM).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            record = ImageRecord(
                image_id=0,
                prompt=f"{subj} prompt {i}",
                optimized_prompt="opt",
                score=0.9,
                image_path=f"/test/{subj}_{i}.png",
                embedding=emb.tolist(),
                subject=subj,
            )
            store.insert(record, subject=subj)

        stats = store.get_stats_by_subject()
        assert stats["geography"] == 2
        assert stats["math"] == 2
        assert stats["physics"] == 1
        assert stats["total"] == 5


class TestCrossSubjectIsolation:
    """验证跨学科分区隔离."""

    def test_geo_prompt_does_not_recall_math(self, vector_store_setup):
        """地理 prompt 不应召回数学图片."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup

        # 插入地理向量（x-axis 方向）
        geo_emb = np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32)
        store.insert(ImageRecord(
            image_id=0, prompt="地理图片", optimized_prompt="geo",
            score=0.9, image_path="/geo.png", embedding=geo_emb.tolist(),
            subject="geography",
        ), subject="geography")

        # 插入数学向量（y-axis 方向）
        math_emb = np.array([0.0, 1.0] + [0.0] * (DIM - 2), dtype=np.float32)
        store.insert(ImageRecord(
            image_id=0, prompt="数学图片", optimized_prompt="math",
            score=0.9, image_path="/math.png", embedding=math_emb.tolist(),
            subject="math",
        ), subject="math")

        # 地理方向的查询，限定地理分区
        query = np.array([0.99, 0.01] + [0.0] * (DIM - 2), dtype=np.float32)
        query = query / np.linalg.norm(query)

        response = store.search_by_text(
            text_embedding=query.tolist(),
            text="地理 query",
            top_k=5,
            subject="geography",
        )
        subjects_found = [r.subject for r in response.results]
        assert "math" not in subjects_found, "学科分区隔离失败！数学图片出现在地理分区检索结果中"

    def test_search_without_subject_searches_all(self, vector_store_setup):
        """不指定 subject → 全库检索."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        emb = np.random.randn(DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        for subj in ["geography", "math"]:
            record = ImageRecord(
                image_id=0, prompt=f"{subj} prompt", optimized_prompt="opt",
                score=0.9, image_path=f"/{subj}.png", embedding=emb.tolist(),
                subject=subj,
            )
            store.insert(record, subject=subj)

        response = store.search_by_text(
            text_embedding=emb.tolist(),
            text="test query",
            top_k=5,
            subject=None,  # 不限定
        )
        subjects_found = {r.subject for r in response.results}
        assert "geography" in subjects_found or "math" in subjects_found
        assert len(response.results) == 2


class TestFullPipelineIntegration:
    """模拟完整 clip_enrich 链路."""

    def test_insert_then_search_roundtrip(self, vector_store_setup):
        """写入一条 → 搜索能召回."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        emb = np.random.randn(DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        record = ImageRecord(
            image_id=0,
            prompt="一幅地理教材风格的火山地貌示意图",
            optimized_prompt="一幅中学地理教材风格的火山地貌示意图...",
            score=0.92,
            image_path="/storage/volcano.png",
            embedding=emb.tolist(),
            subject="geography",
            tags=["volcano", "textbook_style"],
        )
        _id = store.insert(record, subject="geography")

        # 用相同语义的 query 搜索
        response = store.search_by_text(
            text_embedding=emb.tolist(),
            text="火山地貌",
            top_k=5,
            subject="geography",
        )
        assert len(response.results) >= 1
        assert response.results[0].prompt == "一幅地理教材风格的火山地貌示意图"
        assert response.results[0].subject == "geography"
        assert response.total_in_partition == 1

    def test_stats_after_insert(self, vector_store_setup):
        """插入后 get_stats_by_subject 更新."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        emb = np.random.randn(DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)

        # 初始状态
        initial = store.get_stats_by_subject()
        assert initial["geography"] == 0

        # 插入地理
        store.insert(ImageRecord(
            image_id=0, prompt="geo", optimized_prompt="opt",
            score=0.9, image_path="/geo.png", embedding=emb.tolist(),
            subject="geography",
        ), subject="geography")

        updated = store.get_stats_by_subject()
        assert updated["geography"] == 1
        assert updated["total"] == 1

    def test_recall_successful_prompts_with_subject(self, vector_store_setup):
        """recall_successful_prompts 支持 subject 过滤."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        geo_emb = np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32)
        math_emb = np.array([0.0, 1.0] + [0.0] * (DIM - 2), dtype=np.float32)

        store.insert(ImageRecord(
            image_id=0, prompt="地理高分", optimized_prompt="地理优化",
            score=0.95, image_path="/geo_high.png", embedding=geo_emb.tolist(),
            subject="geography",
        ), subject="geography")
        store.insert(ImageRecord(
            image_id=0, prompt="数学高分", optimized_prompt="数学优化",
            score=0.95, image_path="/math_high.png", embedding=math_emb.tolist(),
            subject="math",
        ), subject="math")

        # 限定地理召回
        query = np.array([0.99, 0.01] + [0.0] * (DIM - 2), dtype=np.float32)
        query = query / np.linalg.norm(query)

        results = store.recall_successful_prompts(
            prompt="地理",
            text_embedding=query.tolist(),
            min_score=0.80,
            top_k=5,
            subject="geography",
        )
        assert len(results) >= 1
        subjects = [r.get("subject", "") for r in results]
        assert "math" not in subjects

    def test_find_best_by_exact_prompt_picks_highest_score(self, vector_store_setup):
        """相同 prompt 多条记录时，精确匹配应返回最高分."""
        from src.models.schemas import ImageRecord

        store = vector_store_setup
        shared_prompt = "议论文五种论证方法对比图解"
        emb = np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32)

        store.insert(ImageRecord(
            image_id=0, prompt=shared_prompt, optimized_prompt="低分",
            score=0.82, image_path="/low.png", embedding=emb.tolist(),
            subject="chinese",
        ), subject="chinese")
        store.insert(ImageRecord(
            image_id=0, prompt=shared_prompt, optimized_prompt="高分",
            score=0.90, image_path="/high.png", embedding=emb.tolist(),
            subject="chinese",
        ), subject="chinese")
        store.insert(ImageRecord(
            image_id=0, prompt="其他 prompt", optimized_prompt="其他",
            score=0.99, image_path="/other.png", embedding=emb.tolist(),
            subject="chinese",
        ), subject="chinese")

        hit = store.find_best_by_exact_prompt(
            shared_prompt, subject="chinese", min_score=0.75,
        )
        assert hit is not None
        assert hit["image_path"] == "/high.png"
        assert hit["score"] == 0.90
        assert hit["match_type"] == "exact_prompt"


class TestPipelineClipEnrichMocked:
    """clip_enrich pipeline 的 mock 测试（不依赖真实 CLIP/Draw 模型）."""

    @pytest.mark.asyncio
    async def test_pipeline_clip_enrich_with_subject(self, monkeypatch):
        """验证 clip_enrich 传递 subject 参数."""
        from src.models.schemas import PipelineRequest
        from src.pipeline import ImagePipeline

        pipeline = ImagePipeline()

        # Mock CLIP — 直接设置 _clip_client，让 _lazy_init_clip() 的 None 检查通过
        mock_clip = MagicMock()
        emb = np.random.randn(DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        mock_clip.encode_text.return_value = emb
        mock_clip.encode_image.return_value = emb
        pipeline._clip_client = mock_clip

        # Mock VectorStore
        from src.milvus.vector_store import VectorStore
        mock_store = MagicMock(spec=VectorStore)
        mock_store.connect.return_value = None
        mock_store.backend_type = "local_numpy"
        mock_store.count.return_value = 5
        mock_store.search_by_semantic.return_value = {
            "results": [
                {
                    "image_id": 1,
                    "prompt": "历史高分",
                    "optimized_prompt": "优化",
                    "score": 0.95,
                    "semantic_similarity": 0.5,
                    "image_path": "/old.png",
                    "subject": "geography",
                },
            ],
            "query_time_ms": 12.0,
            "total_in_partition": 100,
        }
        mock_store.find_best_by_exact_prompt.return_value = None
        mock_store.insert.return_value = 42
        pipeline._vector_store = mock_store

        # Mock Drawer
        mock_drawer = AsyncMock()
        mock_drawer.generate.return_value = "/tmp/out.png"
        monkeypatch.setitem(
            __import__('src.draw', fromlist=['DRAWER_REGISTRY']).DRAWER_REGISTRY,
            'tongyi', mock_drawer,
        )

        # Mock Evaluator
        from src.models.schemas import DimensionScore, EvalResult
        mock_eval = AsyncMock()
        mock_eval.evaluate.return_value = EvalResult(
            overall_score=0.95,
            dimension_scores=[
                DimensionScore(dimension="整体语义匹配度", score=0.95, comment="ok"),
            ],
            issues=[], missing_elements=[], suggestions=[],
        )
        monkeypatch.setattr(pipeline, '_evaluator', mock_eval)

        # 运行 clip_enrich
        request = PipelineRequest(
            prompt="一幅地理教学插图",
            model="tongyi",
                        max_iterations=1,
            eval_threshold=0.82,
            subject="geography",
        )

        response = await pipeline._run_clip_enrich(request)

        # 验证 subject 被传递到 search
        mock_store.search_by_semantic.assert_called_once()
        search_call_kwargs = mock_store.search_by_semantic.call_args
        assert search_call_kwargs[1].get('subject') == 'geography'

        # 验证 subject 被传递到 insert
        mock_store.insert.assert_called_once()
        insert_call_args = mock_store.insert.call_args
        assert insert_call_args[1].get('subject') == 'geography'

        # 未命中复用时应使用原始 prompt 生图
        mock_drawer.generate.assert_called_once_with("一幅地理教学插图")

        # 验证响应
        assert response.db_record_id is not None

    @pytest.mark.asyncio
    async def test_pipeline_clip_enrich_reuses_exact_prompt_match(self, monkeypatch):
        """相同 prompt 应优先精确匹配复用，不走 CLIP 跨模态检索."""
        from src.models.schemas import PipelineRequest
        from src.pipeline import ImagePipeline
        from pathlib import Path

        pipeline = ImagePipeline()

        mock_clip = MagicMock()
        emb = np.random.randn(DIM).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        mock_clip.encode_text.return_value = emb
        pipeline._clip_client = mock_clip

        from src.milvus.vector_store import VectorStore
        mock_store = MagicMock(spec=VectorStore)
        mock_store.connect.return_value = None
        mock_store.backend_type = "local_numpy"
        mock_store.count.return_value = 5
        mock_store.find_best_by_exact_prompt.return_value = {
            "image_id": 99,
            "prompt": "议论文五种论证方法对比图解",
            "optimized_prompt": "优化版 prompt",
            "score": 0.90,
            "image_path": "/exact_match.png",
            "subject": "chinese",
            "match_type": "exact_prompt",
            "semantic_similarity": 1.0,
        }
        pipeline._vector_store = mock_store

        mock_drawer = AsyncMock()
        monkeypatch.setitem(
            __import__('src.draw', fromlist=['DRAWER_REGISTRY']).DRAWER_REGISTRY,
            'tongyi', mock_drawer,
        )
        monkeypatch.setattr(Path, "is_file", lambda self: True)

        request = PipelineRequest(
            prompt="议论文五种论证方法对比图解",
            model="tongyi",
                        subject="chinese",
        )

        response = await pipeline._run_clip_enrich(request)

        assert response.stopped_reason == "reused"
        assert response.final_image_path == "/exact_match.png"
        assert response.final_score == 0.90
        assert response.reused_from_record_id == 99
        mock_store.search_by_semantic.assert_not_called()
        mock_drawer.generate.assert_not_called()
        mock_clip.encode_image.assert_not_called()
        mock_store.insert.assert_not_called()
