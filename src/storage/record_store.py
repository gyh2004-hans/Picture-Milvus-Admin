"""阶段 1 数据存储 —— 保存 prompt / image / feedback 记录."""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import re

from src.config import RECORDS_FILE
from src.models.schemas import DrawRecord, FeedbackInput

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """简单中英文分词：中文逐字、英文按空格/标点切.

    用于 Jaccard 相似度计算，不依赖 jieba 等额外库.
    """
    # 提取中文字符
    chinese = re.findall(r"[一-鿿]", text)
    # 提取英文单词（2 字符以上）
    english = re.findall(r"[a-zA-Z]{2,}", text.lower())
    return chinese + english


class RecordStore:
    """JSON 文件存储，阶段 1 使用；后续可迁移 SQLite / PostgreSQL."""

    def __init__(self, records_file: Path = RECORDS_FILE) -> None:
        self._path = records_file
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("[]", encoding="utf-8")

    def _load(self) -> list[dict]:
        raw = self._path.read_text(encoding="utf-8")
        return json.loads(raw or "[]")

    def _save(self, records: list[dict]) -> None:
        self._path.write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create(
        self,
        *,
        prompt: str,
        model: str,
        image_path: str,
        metadata: Optional[dict] = None,
    ) -> DrawRecord:
        """保存一次生图记录."""
        t0 = time.perf_counter()
        record_id = str(uuid.uuid4())
        logger.info(
            "record_store.create.start | record_id=%s model=%s prompt_len=%d",
            record_id,
            model,
            len(prompt),
        )

        record = DrawRecord(
            id=record_id,
            prompt=prompt,
            model=model,
            image_path=image_path,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            metadata=metadata or {},
        )

        records = self._load()
        records.append(record.model_dump())
        self._save(records)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info(
            "record_store.create.end | record_id=%s duration_ms=%d image_path=%s",
            record_id,
            duration_ms,
            image_path,
        )
        return record

    def list_records(self, limit: int = 50) -> list[DrawRecord]:
        records = self._load()
        items = records[-limit:]
        items.reverse()
        return [DrawRecord.model_validate(r) for r in items]

    def get(self, record_id: str) -> Optional[DrawRecord]:
        for raw in self._load():
            if raw.get("id") == record_id:
                return DrawRecord.model_validate(raw)
        return None

    def search_similar(
        self,
        query_text: str,
        top_k: int = 3,
        min_score: float | None = None,
    ) -> list[dict]:
        """检索与 query 最相似的历史成功记录（基于关键词 Jaccard 相似度）.

        用于跨迭代 few-shot：找到历史上 prompt 相似且得分高的记录，
        将其优化策略复用到当前 prompt 优化中。

        Args:
            query_text: 当前 prompt 文本.
            top_k: 返回 top-k 条最相似记录.
            min_score: 最低全局 CLIP 得分过滤（None = 不过滤）.

        Returns:
            [{prompt, image_path, global_score, similarity}, ...]
        """
        records = self._load()
        if not records:
            return []

        query_tokens = set(_tokenize(query_text))

        scored: list[dict] = []
        for raw in records:
            # 提取记录中的 prompt 文本
            record_text = raw.get("prompt", "")
            if not record_text:
                continue

            record_tokens = set(_tokenize(record_text))
            if not query_tokens or not record_tokens:
                continue

            # Jaccard 相似度
            intersection = query_tokens & record_tokens
            union = query_tokens | record_tokens
            similarity = len(intersection) / len(union) if union else 0.0

            # 最低分过滤
            global_score = raw.get("overall_score") or raw.get("metadata", {}).get("overall_score", 0)
            if min_score is not None and global_score < min_score:
                continue

            scored.append({
                "prompt": record_text,
                "image_path": raw.get("image_path", ""),
                "global_score": global_score,
                "similarity": round(similarity, 4),
            })

        # 按相似度降序
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]

    def add_feedback(self, record_id: str, feedback: FeedbackInput) -> DrawRecord:
        """追加人工反馈."""
        t0 = time.perf_counter()
        logger.info("record_store.feedback.start | record_id=%s", record_id)

        records = self._load()
        updated: Optional[DrawRecord] = None

        for i, raw in enumerate(records):
            if raw.get("id") != record_id:
                continue
            record = DrawRecord.model_validate(raw)
            record.feedback = feedback
            record.feedback_submitted_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            records[i] = record.model_dump()
            updated = record
            break

        if updated is None:
            logger.error("record_store.feedback.error | record_id=%s reason=not_found", record_id)
            raise KeyError(f"Record not found: {record_id}")

        self._save(records)
        duration_ms = int((time.perf_counter() - t0) * 1000)
        logger.info("record_store.feedback.end | record_id=%s duration_ms=%d", record_id, duration_ms)
        return updated
