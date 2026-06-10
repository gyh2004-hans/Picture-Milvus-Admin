#!/usr/bin/env python3
"""Milvus 向量检索性能压测基准.

对齐优化计划 §5.3 / §5.5:
  - 不同索引类型 (FLAT / HNSW / IVF_FLAT) 的 QPS 对比
  - 不同数据规模 (1K / 10K / 100K) 的延迟分布
  - 分区检索 vs 全库检索的精度对比

用法:
  python scripts/benchmark_milvus.py                    # 全部基准
  python scripts/benchmark_milvus.py --scale 1k         # 仅 1K 规模
  python scripts/benchmark_milvus.py --index HNSW        # 仅测试 HNSW 索引
  python scripts/benchmark_milvus.py --no-partition      # 跳过分区精度测试
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.milvus.vector_store import (
    DEFAULT_PARTITION,
    SUBJECT_PARTITION_MAP,
    VectorStore,
    get_vector_store,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark_milvus")

# ── 压测参数 ──────────────────────────────────────────────

SCALES: dict[str, int] = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
}

INDEX_TYPES = ["FLAT", "HNSW", "IVF_FLAT"]
DIM = 512  # Chinese-CLIP base-patch16

# 测试用的 9 个学科（均匀分布）
ALL_SUBJECTS = list(SUBJECT_PARTITION_MAP.keys())  # ["chinese", "math", ...]


# ── 辅助 ──────────────────────────────────────────────────

def _make_random_embedding(seed: int | None = None) -> np.ndarray:
    """生成随机 L2 归一化 embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _make_metadata(subject: str, idx: int) -> dict:
    """生成一条测试元数据."""
    return {
        "prompt": f"[benchmark] {subject} prompt #{idx}",
        "optimized_prompt": f"[benchmark] optimized {subject} #{idx}",
        "score": round(float(np.random.RandomState(idx).uniform(0.6, 1.0)), 3),
        "image_path": f"/benchmark/img/{subject}/{idx:06d}.png",
        "subject": subject,
        "category": "benchmark",
        "tags": [subject, "benchmark", f"batch_{idx // 1000}"],
    }


# ── 基准测试 ──────────────────────────────────────────────

class MilvusBenchmark:
    """Milvus 性能基准测试器."""

    def __init__(self, store: VectorStore):
        self._store = store
        self._store.connect()

    # ── 1. 插入性能 ─────────────────────────────────

    def bench_insert(self, scale: int) -> dict:
        """测试批量插入的吞吐量."""
        logger.info("Benchmark: INSERT | scale=%d", scale)
        per_subject = scale // len(ALL_SUBJECTS)

        t0 = time.perf_counter()
        total = 0
        for subject in ALL_SUBJECTS:
            emb = _make_random_embedding()
            for i in range(per_subject):
                from src.models.schemas import ImageRecord
                record = ImageRecord(
                    image_id=0,
                    prompt=f"[bench] {subject} #{i}",
                    score=0.85,
                    image_path=f"/bench/{subject}/{i}.png",
                    embedding=emb.tolist(),
                    subject=subject,
                    tags=["benchmark"],
                )
                self._store.insert(record, subject=subject)
                total += 1
        duration_s = time.perf_counter() - t0
        return {
            "operation": "insert",
            "count": total,
            "duration_s": round(duration_s, 3),
            "qps": round(total / max(duration_s, 0.001), 1),
        }

    # ── 2. 检索性能 ─────────────────────────────────

    def bench_search(self, top_k: int = 5, warmup: int = 10, rounds: int = 50) -> dict:
        """测试检索 QPS 与延迟分布."""
        logger.info("Benchmark: SEARCH | top_k=%d warmup=%d rounds=%d",
                     top_k, warmup, rounds)

        # 生成查询向量
        query_vecs = [_make_random_embedding(i).tolist() for i in range(rounds + warmup)]

        latencies: list[float] = []
        for i, qv in enumerate(query_vecs):
            t0 = time.perf_counter()
            _ = self._store.search_by_text(
                text_embedding=qv,
                text=f"benchmark query {i}",
                top_k=top_k,
                subject=None,  # 全库检索
            )
            lat = (time.perf_counter() - t0) * 1000  # ms
            if i >= warmup:
                latencies.append(lat)

        latencies_sorted = sorted(latencies)
        n = len(latencies_sorted)
        return {
            "operation": "search",
            "top_k": top_k,
            "total_rounds": rounds,
            "latency_ms": {
                "p50": round(latencies_sorted[n // 2], 2),
                "p95": round(latencies_sorted[int(n * 0.95)], 2),
                "p99": round(latencies_sorted[int(n * 0.99)], 2),
                "mean": round(sum(latencies) / n, 2),
                "min": round(latencies_sorted[0], 2),
                "max": round(latencies_sorted[-1], 2),
            },
            "qps": round(1000.0 * n / sum(latencies), 1),
        }

    # ── 3. 分区检索 vs 全库检索 ────────────────────

    def bench_partition_vs_full(
        self, subject: str = "geography", top_k: int = 5, rounds: int = 30
    ) -> dict:
        """对比分区检索与全库检索的延迟与结果一致性."""
        logger.info("Benchmark: PARTITION vs FULL | subject=%s", subject)

        query_vec = _make_random_embedding(42).tolist()

        # 分区检索
        t0 = time.perf_counter()
        for _ in range(rounds):
            result_part = self._store.search_by_text(
                text_embedding=query_vec, text="分区测试",
                top_k=top_k, subject=subject,
            )
        lat_part = (time.perf_counter() - t0) / rounds * 1000

        # 全库检索
        t0 = time.perf_counter()
        for _ in range(rounds):
            result_full = self._store.search_by_text(
                text_embedding=query_vec, text="全库测试",
                top_k=top_k, subject=None,
            )
        lat_full = (time.perf_counter() - t0) / rounds * 1000

        # 精度对比：分区内结果中属于目标学科的占比
        part_subject_match = sum(
            1 for r in result_part.results if r.subject == subject
        )
        full_subject_match = sum(
            1 for r in result_full.results if r.subject == subject
        )

        return {
            "operation": "partition_vs_full",
            "subject": subject,
            "rounds": rounds,
            "partition_search": {
                "avg_latency_ms": round(lat_part, 2),
                "results_count": len(result_part.results),
                "subject_match_rate": round(
                    part_subject_match / max(len(result_part.results), 1), 3
                ),
            },
            "full_search": {
                "avg_latency_ms": round(lat_full, 2),
                "results_count": len(result_full.results),
                "subject_match_rate": round(
                    full_subject_match / max(len(result_full.results), 1), 3
                ),
            },
        }

    # ── 4. 索引类型对比 ─────────────────────────────

    def bench_index_switch(self, index_type: str) -> dict | None:
        """测试切换索引类型后的检索性能."""
        if self._store.backend_type == "local_numpy":
            return {"operation": "index_switch", "error": "not supported on local_numpy backend"}

        logger.info("Benchmark: INDEX switch to %s", index_type)
        t0 = time.perf_counter()
        ok = self._store.create_index(index_type=index_type)
        duration_s = time.perf_counter() - t0

        if not ok:
            return {"operation": "index_switch", "index_type": index_type, "error": "create_index failed"}

        info = self._store.get_index_info()
        return {
            "operation": "index_switch",
            "index_type": index_type,
            "duration_s": round(duration_s, 3),
            "index_info": info,
        }

    # ── 5. 全部基准 ─────────────────────────────────

    def run_all(self, scales: list[int], indices: list[str]) -> list[dict]:
        """运行全部基准测试并返回结果列表."""
        results: list[dict] = []

        for scale in scales:
            logger.info("=" * 50)
            logger.info("SCALE: %d", scale)
            logger.info("=" * 50)

            # 清理旧数据
            self._store.drop_all()
            self._store.connect()

            # 插入
            r = self.bench_insert(scale)
            r["scale"] = scale
            results.append(r)

            # 检索
            r = self.bench_search(top_k=5)
            r["scale"] = scale
            results.append(r)

            # 分区 vs 全库
            r = self.bench_partition_vs_full(subject="geography")
            r["scale"] = scale
            results.append(r)

            # 不同 top_k
            for k in [1, 10, 50]:
                r = self.bench_search(top_k=k, rounds=20)
                r["scale"] = scale
                r["note"] = f"top_k={k}"
                results.append(r)

            # 索引切换（仅 MilvusLite / MilvusServer 后端）
            if self._store.backend_type != "local_numpy":
                for idx_type in indices:
                    r = self.bench_index_switch(idx_type)
                    if r:
                        r = dict(r)  # type: ignore[assignment]
                        r["scale"] = scale
                        results.append(r)
                    # 检索（新索引）
                    r = self.bench_search(top_k=5, rounds=20)
                    r["scale"] = scale
                    r["note"] = f"index={idx_type}"
                    results.append(r)

        return results


# ── CLI ────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milvus 向量检索性能压测基准",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scale", choices=list(SCALES), default=None,
        help="测试数据规模 (默认: 全部)",
    )
    parser.add_argument(
        "--index", choices=INDEX_TYPES, default=None,
        help="仅测试指定索引类型",
    )
    parser.add_argument(
        "--no-partition", action="store_true",
        help="跳过分区精度对比测试",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="结果输出 JSON 文件路径",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="压测后清理所有数据",
    )
    args = parser.parse_args()

    scales = [SCALES[args.scale]] if args.scale else [SCALES["1k"]]
    indices = [args.index] if args.index else INDEX_TYPES[:2]

    store = get_vector_store()
    bench = MilvusBenchmark(store)

    logger.info("Backend: %s", store.backend_type)
    logger.info("Scales: %s", scales)
    logger.info("Indices: %s", indices)

    try:
        results = bench.run_all(scales, indices)
    finally:
        if args.cleanup:
            store.drop_all()
            logger.info("Cleanup: all data dropped")

    # 输出
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.output:
        Path(args.output).write_text(
            json.dumps(results, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Results written to %s", args.output)

    # 摘要
    insert_results = [r for r in results if r["operation"] == "insert"]
    search_results = [r for r in results if r["operation"] == "search" and "note" not in r]
    print("\nSUMMARY:")
    for r in insert_results:
        print(f"  INSERT {r.get('scale', '?'):>5}: {r['qps']:>8.1f} QPS, {r['count']} records in {r['duration_s']}s")
    for r in search_results:
        lat = r["latency_ms"]
        print(f"  SEARCH {r.get('scale', '?'):>5} (top_k={r['top_k']}): "
              f"p50={lat['p50']:.1f}ms p95={lat['p95']:.1f}ms qps={r['qps']:.1f}")


if __name__ == "__main__":
    main()
