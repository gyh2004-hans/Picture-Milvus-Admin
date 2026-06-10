"""Milvus 模块单元测试 (v4).

覆盖:
  - VectorBackend 三种后端的连接、增删查、分区
  - VectorStore 门面的分区路由、统计、CRUD
  - Schema 扩展（Subject enum / ImageRecord / SearchRequest）
  - 分区隔离验证
  - CLIP 缓存（CachedCLIPClient）

运行:
  pytest tests/test_milvus.py -v
  pytest tests/test_milvus.py -v -k "test_partition"  # 仅分区相关
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 确保 picture2 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))

DIM = 512  # Chinese-CLIP base-patch16
DEFAULT_SUBJECT = "geography"


# ══════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════

@pytest.fixture
def sample_embedding() -> np.ndarray:
    """生成一个随机的 L2 归一化 embedding."""
    vec = np.random.randn(DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def sample_metadata() -> dict:
    return {
        "prompt": "一幅地理教材风格的火山地貌示意图",
        "optimized_prompt": "一幅中学地理教材风格的火山地貌示意图，包含火山锥、岩浆通道、火山口等标注",
        "score": 0.92,
        "image_path": "/storage/images/volcano_001.png",
        "subject": "geography",
        "category": "landform",
        "tags": ["volcano", "textbook_style", "diagram"],
    }


@pytest.fixture
def numpy_backend():
    """LocalNumpyBackend fixture（自动 cleanup）."""
    from src.milvus.vector_store import LocalNumpyBackend
    backend = LocalNumpyBackend()
    backend.connect()
    yield backend
    backend.drop_all()


# ══════════════════════════════════════════════════════════
# LocalNumpyBackend 测试
# ══════════════════════════════════════════════════════════

class TestLocalNumpyBackend:
    """LocalNumpyBackend 基础功能测试."""

    def test_connect(self, numpy_backend):
        """连接后 _connected=True."""
        assert numpy_backend._connected is True

    def test_insert_and_count(self, numpy_backend, sample_embedding, sample_metadata):
        """插入一条 → count == 1."""
        _id = numpy_backend.insert(sample_embedding, None, sample_metadata)
        assert _id == 1
        assert numpy_backend.count() == 1

    def test_insert_multiple(self, numpy_backend, sample_embedding, sample_metadata):
        """插入 N 条 → count == N."""
        n = 10
        for i in range(n):
            meta = {**sample_metadata, "image_path": f"/img/{i}.png"}
            numpy_backend.insert(sample_embedding, None, meta)
        assert numpy_backend.count() == n

    def test_search_returns_top_k(self, numpy_backend, sample_embedding, sample_metadata):
        """检索返回 Top-K 结果."""
        for i in range(5):
            meta = {**sample_metadata, "image_path": f"/img/{i}.png"}
            numpy_backend.insert(sample_embedding, None, meta)

        results = numpy_backend.search(sample_embedding, top_k=3)
        assert len(results) == 3
        assert all(isinstance(sim, float) for sim, _ in results)
        assert all(isinstance(rec, dict) for _, rec in results)

    def test_search_similarity_order(self, numpy_backend):
        """验证检索结果按相似度降序排列."""
        # 插入几个不同方向的向量
        vecs = [
            np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32),  # x-axis
            np.array([0.0, 1.0] + [0.0] * (DIM - 2), dtype=np.float32),  # y-axis
            np.array([0.0, 0.0, 1.0] + [0.0] * (DIM - 3), dtype=np.float32),  # z-axis
        ]
        for i, v in enumerate(vecs):
            numpy_backend.insert(v, None, {"prompt": f"vec{i}", "score": 0.9, "image_path": f"/img/{i}.png"})

        # 查询接近 x-axis
        query = np.array([0.9, 0.1] + [0.0] * (DIM - 2), dtype=np.float32)
        query = query / np.linalg.norm(query)
        results = numpy_backend.search(query, top_k=3)
        sims = [s for s, _ in results]
        assert sims == sorted(sims, reverse=True), f"Expected descending similarities, got {sims}"
        # x-axis 应该排第一
        assert results[0][1]["prompt"] == "vec0"

    def test_empty_search(self, numpy_backend, sample_embedding):
        """空库检索返回空列表."""
        results = numpy_backend.search(sample_embedding, top_k=5)
        assert results == []

    def test_drop_all(self, numpy_backend, sample_embedding, sample_metadata):
        """drop_all 后 count == 0."""
        numpy_backend.insert(sample_embedding, None, sample_metadata)
        assert numpy_backend.count() == 1
        numpy_backend.drop_all()
        assert numpy_backend.count() == 0

    def test_partition_insert_and_count(self, numpy_backend, sample_embedding, sample_metadata):
        """分区插入 → 分区 count 正确."""
        numpy_backend.insert(sample_embedding, None, sample_metadata, partition_name="地理")
        numpy_backend.insert(sample_embedding, None, sample_metadata, partition_name="数学")
        assert numpy_backend.count() == 2
        assert numpy_backend.count("地理") == 1
        assert numpy_backend.count("数学") == 1
        assert numpy_backend.count("_default") == 0

    def test_partition_search_isolation(self, numpy_backend):
        """分区检索 → 不跨分区召回."""
        geo_vec = np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32)
        math_vec = np.array([0.0, 1.0] + [0.0] * (DIM - 2), dtype=np.float32)
        query_vec = np.array([0.99, 0.01] + [0.0] * (DIM - 2), dtype=np.float32)
        query_vec = query_vec / np.linalg.norm(query_vec)

        numpy_backend.insert(geo_vec, None, {"prompt": "geo", "score": 0.9, "image_path": "/geo.png"}, partition_name="地理")
        numpy_backend.insert(math_vec, None, {"prompt": "math", "score": 0.9, "image_path": "/math.png"}, partition_name="数学")

        # 仅在地理分区搜
        results = numpy_backend.search(query_vec, top_k=5, partition_names=["地理"])
        assert len(results) == 1
        assert results[0][1]["prompt"] == "geo"

        # 全库搜索
        results_all = numpy_backend.search(query_vec, top_k=5)
        assert len(results_all) == 2

    def test_partition_crud(self, numpy_backend):
        """分区创建/列表/删除."""
        numpy_backend.create_partition("测试分区")
        assert numpy_backend.has_partition("测试分区") is True
        assert "测试分区" in numpy_backend.list_partitions()
        numpy_backend.drop_partition("测试分区")
        assert numpy_backend.has_partition("测试分区") is False

    def test_get_partition_stats(self, numpy_backend, sample_embedding, sample_metadata):
        """get_partition_stats 返回各分区数据量."""
        numpy_backend.insert(sample_embedding, None, sample_metadata, partition_name="地理")
        numpy_backend.insert(sample_embedding, None, sample_metadata, partition_name="地理")
        numpy_backend.insert(sample_embedding, None, sample_metadata, partition_name="数学")
        stats = numpy_backend.get_partition_stats()
        assert stats["地理"] == 2
        assert stats["数学"] == 1

    def test_delete_by_id(self, numpy_backend, sample_embedding, sample_metadata):
        """delete_by_id 正确删除."""
        _id = numpy_backend.insert(sample_embedding, None, sample_metadata)
        assert numpy_backend.count() == 1
        ok = numpy_backend.delete_by_id(_id)
        assert ok is True
        assert numpy_backend.count() == 0

    def test_delete_nonexistent(self, numpy_backend):
        """删除不存在的 id 返回 False."""
        ok = numpy_backend.delete_by_id(99999)
        assert ok is False

    def test_update_metadata(self, numpy_backend, sample_embedding, sample_metadata):
        """update_metadata 正确更新."""
        _id = numpy_backend.insert(sample_embedding, None, sample_metadata)
        ok = numpy_backend.update_metadata(_id, {"score": 0.99, "subject": "math"})
        assert ok is True
        # 验证更新
        results = numpy_backend.search(sample_embedding, top_k=1)
        assert results[0][1]["score"] == 0.99
        assert results[0][1]["subject"] == "math"


# ══════════════════════════════════════════════════════════
# VectorStore 门面测试
# ══════════════════════════════════════════════════════════

class TestVectorStoreFacade:
    """VectorStore 门面测试（使用 LocalNumpyBackend 作为底层）."""

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        """强制使用 LocalNumpyBackend（避免依赖 Milvus/Docker）."""
        from src.milvus import vector_store as vs_module

        # 重置单例（测试隔离）
        vs_module._reset_vector_store_instance()

        # Patch connect() 直接设置 LocalNumpyBackend（绕过 Milvus Standalone 连接）
        def _fake_connect(self_: vs_module.VectorStore) -> None:
            backend = vs_module.LocalNumpyBackend()
            backend.connect()
            self_._backend = backend
            self_._backend_label = "local_numpy"

        monkeypatch.setattr(vs_module.VectorStore, "connect", _fake_connect)
        yield
        vs_module._reset_vector_store_instance()

    def test_vector_store_connect_fails_without_milvus(self, monkeypatch):
        """连接 Milvus Standalone 失败时直接抛 RuntimeError（不静默回退）."""
        import pytest as pytest_mod
        from unittest.mock import patch as mock_patch
        from src.milvus.vector_store import VectorStore
        from src.milvus import vector_store as vs_module

        # 撤销 fixture 的 connect patch，恢复真实 connect 实现
        monkeypatch.undo()
        vs_module._reset_vector_store_instance()

        # Mock MilvusLiteBackend 模拟连接失败，不依赖本机 Milvus 是否运行
        with mock_patch.object(vs_module, 'MilvusLiteBackend', autospec=True) as mock_cls:
            mock_cls.return_value.connect.side_effect = ConnectionError("Connection refused")

            store = VectorStore()
            with pytest_mod.raises(RuntimeError, match="无法连接 Milvus Standalone"):
                store.connect()

    def test_insert_with_subject(self, sample_embedding, sample_metadata):
        """insert 传入 subject → 正确路由到分区."""
        from src.milvus.vector_store import get_vector_store
        from src.models.schemas import ImageRecord

        store = get_vector_store()
        store.connect()

        record = ImageRecord(
            image_id=0,
            prompt=sample_metadata["prompt"],
            optimized_prompt=sample_metadata["optimized_prompt"],
            score=sample_metadata["score"],
            image_path=sample_metadata["image_path"],
            embedding=sample_embedding.tolist(),
            subject="geography",
            category="landform",
            tags=["volcano"],
        )
        _id = store.insert(record, subject="geography")
        assert _id >= 1
        assert store.count() == 1
        assert store.count(subject="geography") == 1
        store.drop_all()

    def test_search_by_text_with_subject(self, sample_embedding, sample_metadata):
        """search_by_text 限定 subject → 仅在该分区检索."""
        from src.milvus.vector_store import get_vector_store
        from src.models.schemas import ImageRecord

        store = get_vector_store()
        store.connect()

        # 插入地理
        record_geo = ImageRecord(
            image_id=0, prompt="地理图片", optimized_prompt="地理优化",
            score=0.9, image_path="/geo.png", embedding=sample_embedding.tolist(),
            subject="geography",
        )
        store.insert(record_geo, subject="geography")

        # 插入数学（不同的 embedding 方向）
        math_emb = np.array([0.0] * DIM, dtype=np.float32)
        math_emb[1] = 1.0
        math_emb = math_emb / np.linalg.norm(math_emb)
        record_math = ImageRecord(
            image_id=0, prompt="数学图片", optimized_prompt="数学优化",
            score=0.9, image_path="/math.png", embedding=math_emb.tolist(),
            subject="math",
        )
        store.insert(record_math, subject="math")

        # 用地理念度查询，限定地理分区 → 应只返回地理
        response = store.search_by_text(
            text_embedding=sample_embedding.tolist(),
            text="地理",
            top_k=5,
            subject="geography",
        )
        assert len(response.results) >= 1
        # 不应该包含数学
        subjects_found = [r.subject for r in response.results]
        assert "math" not in subjects_found

        store.drop_all()

    def test_partition_isolation(self, sample_embedding):
        """验证跨学科分区隔离：语文分区数据在数学检索时不可见."""
        from src.milvus.vector_store import get_vector_store
        from src.models.schemas import ImageRecord

        store = get_vector_store()
        store.connect()

        chinese_emb = np.array([1.0] + [0.0] * (DIM - 1), dtype=np.float32)
        math_emb = np.array([0.0, 1.0] + [0.0] * (DIM - 2), dtype=np.float32)

        store.insert(ImageRecord(
            image_id=0, prompt="语文课文插图", optimized_prompt="语文",
            score=0.9, image_path="/chinese.png", embedding=chinese_emb.tolist(),
            subject="chinese",
        ), subject="chinese")
        store.insert(ImageRecord(
            image_id=0, prompt="数学几何题图", optimized_prompt="数学",
            score=0.9, image_path="/math.png", embedding=math_emb.tolist(),
            subject="math",
        ), subject="math")

        # 用语文 query 搜数学分区 → 不应返回语文
        query = np.array([0.99, 0.01] + [0.0] * (DIM - 2), dtype=np.float32)
        query = query / np.linalg.norm(query)

        response = store.search_by_text(
            text_embedding=query.tolist(),
            text="语文",
            top_k=5,
            subject="math",
        )
        # 数学分区只有一个数学记录
        for r in response.results:
            assert r.subject != "chinese", "跨分区污染！"

        store.drop_all()

    def test_get_stats_by_subject(self, sample_embedding):
        """get_stats_by_subject 返回正确统计."""
        from src.milvus.vector_store import get_vector_store
        from src.models.schemas import ImageRecord

        store = get_vector_store()
        store.connect()

        for subj in ["geography", "geography", "math", "physics"]:
            emb = np.random.randn(DIM).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            store.insert(ImageRecord(
                image_id=0, prompt=f"{subj} prompt", optimized_prompt="opt",
                score=0.9, image_path=f"/{subj}.png", embedding=emb.tolist(),
                subject=subj,
            ), subject=subj)

        stats = store.get_stats_by_subject()
        assert stats["geography"] == 2
        assert stats["math"] == 1
        assert stats["physics"] == 1
        assert stats["total"] == 4

        store.drop_all()

    def test_count_by_subject(self, sample_embedding):
        """count(subject='xxx') 返回正确计数."""
        from src.milvus.vector_store import get_vector_store
        from src.models.schemas import ImageRecord

        store = get_vector_store()
        store.connect()

        store.insert(ImageRecord(
            image_id=0, prompt="geo", optimized_prompt="opt",
            score=0.9, image_path="/geo.png", embedding=sample_embedding.tolist(),
            subject="geography",
        ), subject="geography")

        assert store.count() == 1
        assert store.count(subject="geography") == 1
        assert store.count(subject="math") == 0

        store.drop_all()

    def test_list_partitions(self, sample_embedding):
        """list_partitions 返回已创建的分区."""
        from src.milvus.vector_store import get_vector_store
        from src.models.schemas import ImageRecord

        store = get_vector_store()
        store.connect()

        store._ensure_partition("地理")
        store._ensure_partition("数学")
        partitions = store.list_partitions()
        names = [p["name"] for p in partitions]
        assert "地理" in names
        assert "数学" in names

        store.drop_all()

    def test_insert_without_subject_goes_to_default(self, sample_embedding):
        """无 subject 插入 → _default 分区."""
        from src.milvus.vector_store import get_vector_store
        from src.models.schemas import ImageRecord

        store = get_vector_store()
        store.connect()

        store.insert(ImageRecord(
            image_id=0, prompt="no subject", optimized_prompt="opt",
            score=0.8, image_path="/no_subj.png", embedding=sample_embedding.tolist(),
            subject=None,
        ))
        stats = store.get_stats_by_subject()
        assert stats["_default"] == 1
        store.drop_all()


# ══════════════════════════════════════════════════════════
# MilvusLiteBackend.count 测试
# ══════════════════════════════════════════════════════════

class TestMilvusLiteBackendCount:
    """MilvusLiteBackend.count 的 topk 上限与分页逻辑."""

    @staticmethod
    def _make_backend(mock_client: MagicMock):
        from src.milvus.vector_store import MilvusLiteBackend
        backend = MilvusLiteBackend(uri="http://localhost:19530")
        backend._connected = True
        backend._dim = DIM
        backend._collection_name = "image_embeddings"
        backend._client = mock_client
        return backend

    def test_count_search_fallback_respects_topk_limit(self):
        """query 失败时 search limit 不超过 Milvus 上限."""
        from src.milvus.vector_store import MILVUS_SEARCH_MAX_LIMIT

        mock_client = MagicMock()
        mock_client.query.side_effect = RuntimeError("query unavailable")
        mock_client.search.return_value = [[{"id": 1}, {"id": 2}]]

        backend = self._make_backend(mock_client)
        assert backend.count() == 2

        search_kwargs = mock_client.search.call_args.kwargs
        assert search_kwargs["limit"] == MILVUS_SEARCH_MAX_LIMIT
        assert search_kwargs["limit"] <= 16384

    def test_count_query_pagination(self):
        """query 分页可统计超过单次 topk 上限的记录数."""
        from src.milvus.vector_store import MILVUS_SEARCH_MAX_LIMIT

        mock_client = MagicMock()
        mock_client.query.side_effect = [
            [{"id": i} for i in range(MILVUS_SEARCH_MAX_LIMIT)],
            [{"id": i} for i in range(MILVUS_SEARCH_MAX_LIMIT, MILVUS_SEARCH_MAX_LIMIT + 100)],
        ]

        backend = self._make_backend(mock_client)
        assert backend.count() == MILVUS_SEARCH_MAX_LIMIT + 100
        assert mock_client.search.call_count == 0

    def test_count_query_with_partition(self):
        """分区计数透传 partition_names."""
        mock_client = MagicMock()
        mock_client.query.return_value = [{"id": 1}, {"id": 2}]

        backend = self._make_backend(mock_client)
        assert backend.count(partition_name="yuwen") == 2

        query_kwargs = mock_client.query.call_args.kwargs
        assert query_kwargs["partition_names"] == ["yuwen"]


# ══════════════════════════════════════════════════════════
# Schema 测试
# ══════════════════════════════════════════════════════════

class TestSchemaExtension:
    """Schema 扩展验证."""

    def test_subject_enum_values(self):
        """Subject enum 包含全部 9 个学科."""
        from src.models.schemas import Subject
        values = [s.value for s in Subject]
        assert "chinese" in values
        assert "math" in values
        assert "english" in values
        assert "physics" in values
        assert "chemistry" in values
        assert "biology" in values
        assert "history" in values
        assert "geography" in values
        assert "politics" in values

    def test_subject_enum_chinese_alias(self):
        """Subject enum 支持中文别名."""
        from src.models.schemas import Subject
        assert Subject("语文") == Subject.CHINESE
        assert Subject("数学") == Subject.MATH
        assert Subject("地理") == Subject.GEOGRAPHY

    def test_subject_enum_case_insensitive(self):
        """Subject enum 大小写不敏感."""
        from src.models.schemas import Subject
        assert Subject("Geography") == Subject.GEOGRAPHY
        assert Subject("MATH") == Subject.MATH

    def test_subject_enum_unknown(self):
        """未知学科名抛 ValueError."""
        from src.models.schemas import Subject
        with pytest.raises(ValueError):
            Subject("unknown_subject_xyz")

    def test_image_record_subject_normalization(self):
        """ImageRecord 自动规范化 subject 字段."""
        from src.models.schemas import ImageRecord
        rec = ImageRecord(
            image_id=1, prompt="test", score=0.9, image_path="/test.png",
            subject="地理",  # 中文 → 应转为 "geography"
        )
        assert rec.subject == "geography"

    def test_image_record_empty_subject(self):
        """ImageRecord subject=None 保持 None."""
        from src.models.schemas import ImageRecord
        rec = ImageRecord(
            image_id=1, prompt="test", score=0.9, image_path="/test.png",
            subject=None,
        )
        assert rec.subject is None

    def test_image_record_tags_default(self):
        """ImageRecord tags 默认为空列表."""
        from src.models.schemas import ImageRecord
        rec = ImageRecord(
            image_id=1, prompt="test", score=0.9, image_path="/test.png",
        )
        assert rec.tags == []

    def test_search_request_subject_field(self):
        """SearchRequest 包含 subject 字段."""
        from src.models.schemas import SearchRequest
        req = SearchRequest(prompt="test", subject="geography")
        assert req.subject == "geography"

    def test_search_response_total_in_partition(self):
        """SearchResponse 包含 total_in_partition 字段."""
        from src.models.schemas import SearchResponse, ImageRecord
        resp = SearchResponse(results=[], query_time_ms=1.5, total_in_partition=42)
        assert resp.total_in_partition == 42


# ══════════════════════════════════════════════════════════
# CachedCLIPClient 测试
# ══════════════════════════════════════════════════════════

class TestCachedCLIPClient:
    """CachedCLIPClient 缓存测试."""

    @pytest.fixture
    def mock_clip_client(self):
        """Mock LocalCLIPClient."""
        client = MagicMock()
        client.embedding_dim = 512
        client.encode_text.return_value = np.array([1.0] + [0.0] * 511, dtype=np.float32)
        client.encode_texts.return_value = np.array([[1.0] + [0.0] * 511], dtype=np.float32)
        client.encode_image.return_value = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)
        client.encode_image_patches.return_value = np.zeros((256, 512), dtype=np.float32)
        return client

    def test_text_cache_hit(self, mock_clip_client):
        """相同文本两次编码 → 第二次命中缓存."""
        from src.evaluate.cached_clip_client import CachedCLIPClient
        cc = CachedCLIPClient(mock_clip_client, cache_size=10)

        emb1 = cc.encode_text("一幅地理教学插图")
        emb2 = cc.encode_text("一幅地理教学插图")

        np.testing.assert_array_equal(emb1, emb2)
        # encode_text 只应被底层调用一次
        assert mock_clip_client.encode_text.call_count == 1
        assert cc.cache_stats["hits"] == 1
        assert cc.cache_stats["misses"] == 1

    def test_text_cache_miss(self, mock_clip_client):
        """不同文本 → 两次都 miss."""
        from src.evaluate.cached_clip_client import CachedCLIPClient
        cc = CachedCLIPClient(mock_clip_client, cache_size=10)

        cc.encode_text("地理")
        cc.encode_text("数学")

        assert mock_clip_client.encode_text.call_count == 2
        assert cc.cache_stats["misses"] == 2
        assert cc.cache_stats["hits"] == 0

    def test_lru_eviction(self, mock_clip_client):
        """超出 cache_size 时 LRU 淘汰."""
        from src.evaluate.cached_clip_client import CachedCLIPClient
        cc = CachedCLIPClient(mock_clip_client, cache_size=3)

        for i in range(5):
            mock_clip_client.encode_text.return_value = np.array([float(i)] + [0.0] * 511, dtype=np.float32)
            cc.encode_text(f"text_{i}")

        assert cc.cache_stats["text_entries"] <= 3
        assert cc.cache_stats["misses"] == 5
        # 最早的应该被淘汰
        # 重新编码 text_0 应该 miss
        mock_clip_client.encode_text.return_value = np.array([99.0] + [0.0] * 511, dtype=np.float32)
        cc.encode_text("text_0")
        assert mock_clip_client.encode_text.call_count == 6  # 5 + 1

    def test_image_cache_hit(self, mock_clip_client, tmp_path):
        """相同图片路径 → 缓存命中（mock hash）."""
        from src.evaluate.cached_clip_client import CachedCLIPClient

        # 创建临时图片文件
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"fake_image_data")

        cc = CachedCLIPClient(mock_clip_client, cache_size=10)
        emb1 = cc.encode_image(str(img_path))
        emb2 = cc.encode_image(str(img_path))

        np.testing.assert_array_equal(emb1, emb2)
        # 只要文件内容不变，第二次应该命中
        assert mock_clip_client.encode_image.call_count == 1

    def test_clear_cache(self, mock_clip_client):
        """clear_cache 清空缓存并重置统计."""
        from src.evaluate.cached_clip_client import CachedCLIPClient
        cc = CachedCLIPClient(mock_clip_client, cache_size=10)

        cc.encode_text("test")
        assert cc.cache_stats["text_entries"] == 1

        cc.clear_cache()
        assert cc.cache_stats["text_entries"] == 0
        assert cc.cache_stats["hits"] == 0
        assert cc.cache_stats["misses"] == 0

    def test_cache_stats_format(self, mock_clip_client):
        """cache_stats 返回正确格式."""
        from src.evaluate.cached_clip_client import CachedCLIPClient
        cc = CachedCLIPClient(mock_clip_client, cache_size=10)

        cc.encode_text("hello")
        stats = cc.cache_stats
        assert "text_entries" in stats
        assert "image_entries" in stats
        assert "patch_entries" in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert 0.0 <= stats["hit_rate"] <= 1.0


# ══════════════════════════════════════════════════════════
# 分区解析辅助函数测试
# ══════════════════════════════════════════════════════════

class TestPartitionResolution:
    """_resolve_partition 辅助函数测试."""

    def test_resolve_english_subject(self):
        """英文 subject → 拼音分区名（v6: Milvus 分区名须 ASCII 兼容）."""
        from src.milvus.vector_store import _resolve_partition
        assert _resolve_partition("geography") == "dili"
        assert _resolve_partition("math") == "shuxue"
        assert _resolve_partition("chinese") == "yuwen"

    def test_resolve_chinese_subject(self):
        """中文 subject → 经 Subject 枚举转为拼音分区名."""
        from src.milvus.vector_store import _resolve_partition
        assert _resolve_partition("地理") == "dili"
        assert _resolve_partition("数学") == "shuxue"

    def test_resolve_none(self):
        """subject=None → _default."""
        from src.milvus.vector_store import _resolve_partition
        assert _resolve_partition(None) == "_default"
        assert _resolve_partition("") == "_default"

    def test_resolve_unknown(self):
        """未知 subject → _default."""
        from src.milvus.vector_store import _resolve_partition
        assert _resolve_partition("astronomy") == "_default"

    def test_resolve_case_insensitive(self):
        """大小写不敏感（v6: 返回拼音分区名）."""
        from src.milvus.vector_store import _resolve_partition
        assert _resolve_partition("GEOGRAPHY") == "dili"


# ══════════════════════════════════════════════════════════
# 迁移脚本测试
# ══════════════════════════════════════════════════════════

class TestMigrationInference:
    """迁移脚本的学科推断功能测试."""

    def test_infer_geography_from_prompt(self):
        """从地理相关 prompt 推断 geography."""
        from scripts.migrate_add_subject import infer_subject_from_prompt
        result = infer_subject_from_prompt("一幅中学地理教材风格的火山地貌示意图，包含火山锥、岩浆通道")
        assert result == "geography"

    def test_infer_math_from_prompt(self):
        """从数学相关 prompt 推断 math."""
        from scripts.migrate_add_subject import infer_subject_from_prompt
        result = infer_subject_from_prompt("几何三角形内角和定理证明示意图")
        assert result == "math"

    def test_infer_physics_from_prompt(self):
        """从物理相关 prompt 推断 physics."""
        from scripts.migrate_add_subject import infer_subject_from_prompt
        result = infer_subject_from_prompt("牛顿第二定律力学实验装置示意图")
        assert result == "physics"

    def test_infer_chemistry_from_prompt(self):
        """从化学相关 prompt 推断 chemistry."""
        from scripts.migrate_add_subject import infer_subject_from_prompt
        result = infer_subject_from_prompt("氧化还原反应化学方程式配平示意图")
        assert result == "chemistry"

    def test_infer_none_from_short_prompt(self):
        """短 prompt 无法推断 → None."""
        from scripts.migrate_add_subject import infer_subject_from_prompt
        result = infer_subject_from_prompt("一幅图片")
        assert result is None

    def test_infer_none_from_empty(self):
        """空 prompt → None."""
        from scripts.migrate_add_subject import infer_subject_from_prompt
        assert infer_subject_from_prompt("") is None
