"""Tests for the Draw module."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def test_base_drawer_abstract():
    """验证 BaseDrawer 是抽象类，不能直接实例化."""
    from src.draw.base import BaseDrawer

    with pytest.raises(TypeError):
        BaseDrawer()  # type: ignore[abstract]


@patch("src.draw.doubao.post_image_generation", new_callable=AsyncMock)
def test_doubao_drawer_generate(mock_api, tmp_path):
    """验证 DoubaoDrawer 调用 API 并保存图片."""
    mock_api.return_value = FAKE_PNG
    from src.draw import doubao as doubao_mod

    with patch.object(doubao_mod, "IMAGE_DIR", tmp_path):
        drawer = doubao_mod.DoubaoDrawer()
        path = asyncio.run(drawer.generate("地球仪上的纬线与经线教学插图"))
        assert path.endswith(".png")
        assert "doubao" in path
        assert Path(path).exists()
        mock_api.assert_called_once()


@patch("src.draw.tongyi.post_dashscope_image_generation", new_callable=AsyncMock)
def test_tongyi_drawer_generate(mock_api, tmp_path):
    """验证 TongyiDrawer 调用 API 并保存图片."""
    mock_api.return_value = FAKE_PNG
    from src.draw import tongyi as tongyi_mod

    with patch.object(tongyi_mod, "IMAGE_DIR", tmp_path):
        drawer = tongyi_mod.TongyiDrawer()
        path = asyncio.run(drawer.generate("世界地图时区示意图"))
        assert path.endswith(".png")
        assert "tongyi" in path
        assert Path(path).exists()
        mock_api.assert_called_once()


def test_record_store_create_and_feedback(tmp_path):
    """验证记录存储与反馈流程."""
    from src.models.schemas import FeedbackInput
    from src.storage.record_store import RecordStore

    store = RecordStore(records_file=tmp_path / "records.json")
    record = store.create(
        prompt="测试 prompt",
        model="doubao",
        image_path=str(tmp_path / "test.png"),
        metadata={"demo": "geography"},
    )
    assert record.id
    assert record.feedback is None

    updated = store.add_feedback(
        record.id,
        FeedbackInput(rating=4, comment="纬线清晰", tags=["accurate"]),
    )
    assert updated.feedback is not None
    assert updated.feedback.rating == 4
    assert updated.feedback_submitted_at

    fetched = store.get(record.id)
    assert fetched is not None
    assert fetched.feedback.comment == "纬线清晰"
