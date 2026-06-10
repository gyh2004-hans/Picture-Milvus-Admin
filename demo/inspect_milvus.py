"""查看 Milvus 数据库内容（一行命令）.

用法:
    python -m demo.inspect_milvus
    python -m demo.inspect_milvus --top 10
    python -m demo.inspect_milvus --dump   # 导出全部字段
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> None:
    # UTF-8 for Windows terminals
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="查看 Milvus / Numpy 数据库内容")
    parser.add_argument("--top", type=int, default=20, help="显示条数（默认20）")
    parser.add_argument("--dump", action="store_true", help="导出全部字段为JSON")
    parser.add_argument("--drop", action="store_true", help="清空数据库（谨慎！）")
    args = parser.parse_args(argv)

    STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"

    # ── 尝式 1: Milvus Lite ──────────────────────────
    milvus_db = STORAGE_DIR / "milvus_lite.db"
    if milvus_db.exists():
        print(f"[Milvus Lite]  {milvus_db}")
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(str(milvus_db))

            collection = "image_embeddings"
            if not client.has_collection(collection):
                print(f"  Collection '{collection}' not found")
                client.close()
                return

            # Load collection for query
            try:
                client.load_collection(collection)
            except Exception:
                pass

            stats = client.get_collection_stats(collection)
            total = stats.get("row_count", 0)
            print(f"  Total records: {total}")
            print(f"  Collection: {collection}")
            print()

            if args.drop:
                ans = input("  !!! 确认要删除所有数据? (输入 YES 确认): ")
                if ans.strip() == "YES":
                    client.drop_collection(collection)
                    print(f"  Collection '{collection}' dropped")
                else:
                    print("  Cancelled")
                client.close()
                return

            if total == 0:
                print("  (empty)")
                client.close()
                return

            # query with dynamic field filter; use score >= 0 (always true for our data)
            results = client.query(
                collection_name=collection,
                filter="score >= 0",
                output_fields=["id", "prompt", "optimized_prompt", "score", "image_path", "created_at", "model_version"],
                limit=args.top,
            )
            print(f"  Showing top {min(len(results), args.top)}/{total}")
            print(f"  {'='*58}")
            print(f"  {'='*58}")

            for i, r in enumerate(results, 1):
                img_id = r.get("id", "?")
                score = r.get("score", 0)
                prompt = (r.get("prompt", "") or "")[:80]
                opt = (r.get("optimized_prompt", "") or "")[:80]
                img = r.get("image_path", "") or ""
                created = r.get("created_at", "") or ""

                print(f"\n  [{i}] id={img_id}  score={score:.4f}")
                print(f"      prompt:    {prompt}")
                if opt:
                    print(f"      optimized: {opt}")
                if img:
                    print(f"      image:     {img}")
                if created:
                    print(f"      created:   {created}")

                if args.dump:
                    # 检查是否有 vector 字段
                    vec = r.get("vector")
                    if vec:
                        print(f"      vector:    [{len(vec)}-dim, norm={np.linalg.norm(vec):.4f}]")

            # Don't close yet — need it for stats below

            # 汇总统计（复用已获取的 results）
            print(f"\n  {'='*58}")
            if results:
                scores = [r.get("score", 0) for r in results]
                print(f"  Score stats (top {len(scores)}): min={min(scores):.4f} max={max(scores):.4f} "
                      f"mean={np.mean(scores):.4f} median={np.median(scores):.4f}")
            # Also try to get all scores for broader stats
            try:
                all_results = client.query(
                    collection_name=collection,
                    filter="score >= 0",
                    output_fields=["score"],
                    limit=10000,
                )
                if all_results:
                    scores = [r.get("score", 0) for r in all_results]
                    print(f"  Score stats (all {len(scores)}): min={min(scores):.4f} max={max(scores):.4f} "
                          f"mean={np.mean(scores):.4f} median={np.median(scores):.4f}")
            except Exception:
                pass
            client.close()
            return

        except ImportError:
            print("  pymilvus not installed, trying numpy backend...")
        except Exception as exc:
            print(f"  Error querying Milvus Lite: {exc}")
            print("  Trying numpy backend...")

    # ── 尝试 2: Local Numpy Backend (records.json) ────
    records_file = STORAGE_DIR / "records.json"
    if not records_file.exists():
        print("\n[LocalNumpy] No records found (storage/records.json also missing)")
        print("  Run a demo first, e.g.: python -m demo.clip_milvus_demo --mode dry-run")
        return

    print(f"\n[LocalNumpy]  {records_file}")
    records = json.loads(records_file.read_text(encoding="utf-8"))
    total = len(records)
    print(f"  Total records: {total}")

    if args.drop:
        ans = input("  !!! 确认要删除所有数据? (输入 YES 确认): ")
        if ans.strip() == "YES":
            records_file.write_text("[]", encoding="utf-8")
            print("  Cleared")
        else:
            print("  Cancelled")
        return

    if total == 0:
        print("  (empty)")
        return

    for i, r in enumerate(records[-args.top:], 1):
        print(f"\n  [{i}] id={r.get('id', '?')}  model={r.get('model', '')}")
        print(f"      prompt:    {(r.get('prompt', '') or '')[:80]}")
        img = r.get("image_path", "") or ""
        if img:
            print(f"      image:     {img}")
        created = r.get("created_at", "") or ""
        if created:
            print(f"      created:   {created}")

        if args.dump:
            print(f"      full: {json.dumps(r, ensure_ascii=False, default=str)[:200]}")


if __name__ == "__main__":
    main()
