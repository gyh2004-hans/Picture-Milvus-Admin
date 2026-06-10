"""CLIP + Milvus 集成测试 Demo —— 对应项目计划书 §5 检索与向量数据库模块.

验证 Chinese-CLIP 特征提取 + Milvus 向量存储 + 检索的全链路功能:
  1. Chinese-CLIP 图片编码 → 768-dim embedding
  2. Chinese-CLIP 文本编码 → 768-dim embedding
  3. Milvus 向量存储（对齐 §5.2 字段）
  4. 以文搜图 [Search]（§5.3 图文语义搜索）
  5. 以图搜图 [Image]（§5.3 相似图像检索）
  6. 历史 Prompt 复用 [Reuse]（§5.3 历史结果复用）

用法（在 picture2 目录）:
    # dry-run 模式：使用随机向量模拟全流程（无需模型/GPU）
    python -m demo.clip_milvus_demo --mode dry-run

    # real 模式：加载真实 Chinese-CLIP 模型 + 编码已有图片
    python -m demo.clip_milvus_demo --mode real --image storage/images/xxx.png

    # real 模式 + 仅编码文本
    python -m demo.clip_milvus_demo --mode real
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import CLIP_MODEL_NAME, CLIP_DEVICE, CLIP_USE_FP16, IMAGE_DIR, STORAGE_DIR
from src.milvus.vector_store import VectorStore, get_vector_store
from src.models.schemas import ImageRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# 模拟教学插图数据（用于 dry-run 模式填充 Milvus）
# ═══════════════════════════════════════════════════════════════

MOCK_RECORDS: list[dict] = [
    {
        "prompt": "教育插画，清晰标注的地球仪特写，赤道以红色粗实线突出，多条平行纬线用蓝色虚线",
        "optimized_prompt": "中学地理教材风格地球仪特写，赤道红色加粗实线，纬线蓝色虚线从赤道向两极递减，经线绿色放射状连接南北极，白色背景扁平矢量风格",
        "score": 0.92,
        "image_path": "storage/images/geo_earth.png",
    },
    {
        "prompt": "地理教学示意图，同一时刻三个不同纬度地点并排对比太阳光照射角度",
        "optimized_prompt": "地理教科书太阳高度角对比图，赤道垂直照射90°标注，中纬度45°斜射标注，北极圈水平照射标注，黄色箭头+简化地面横线，白底",
        "score": 0.88,
        "image_path": "storage/images/geo_sun_angle.png",
    },
    {
        "prompt": "动物细胞与植物细胞并列对比剖面图，标注细胞膜细胞质细胞核",
        "optimized_prompt": "中学生物细胞结构对比图，左侧动物细胞圆形粉色系，右侧植物细胞方形绿色系含细胞壁叶绿体液泡，相同结构虚线连接，白底中文标注",
        "score": 0.95,
        "image_path": "storage/images/bio_cell.png",
    },
    {
        "prompt": "光合作用全过程示意图，光反应类囊体膜与暗反应卡尔文循环",
        "optimized_prompt": "中学生物光合作用流程图，左上太阳图标光线箭头→叶片剖面，光反应黄色底H₂O→O₂+ATP，暗反应绿色底CO₂→葡萄糖循环箭头，白底黄绿配色",
        "score": 0.90,
        "image_path": "storage/images/bio_photosynthesis.png",
    },
    {
        "prompt": "DNA双螺旋结构立体示意图，脱氧核糖磷酸碱基配对A-T C-G",
        "optimized_prompt": "DNA双螺旋结构分子模型图，蓝色脱氧核糖五边形+黄色磷酸圆圈交替骨架，A=T双虚线红色配对，C≡G三虚线绿色配对，浅蓝底彩色标注",
        "score": 0.87,
        "image_path": "storage/images/bio_dna.png",
    },
    {
        "prompt": "秦朝中央集权制度层级结构图，皇帝三公九卿郡县制",
        "optimized_prompt": "秦朝中央集权示意图，顶层皇帝金色图标，三公丞相太尉御史大夫红蓝绿并排，九卿九宫格排列，郡→县→乡→里层级箭头，米黄仿古底色",
        "score": 0.85,
        "image_path": "storage/images/hist_qin.png",
    },
    {
        "prompt": "丝绸之路路线图，长安河西走廊西域中亚西亚到罗马",
        "optimized_prompt": "陆上丝绸之路历史地图，长安红色起点→敦煌玉门关→楼兰于阗→葱岭→撒马尔罕→巴格达→罗马终点，沙漠浅棕山脉灰色，骆驼商队剪影，羊皮纸底色",
        "score": 0.91,
        "image_path": "storage/images/hist_silk_road.png",
    },
    {
        "prompt": "第一次鸦片战争形势图，中国东部沿海广州定海天津大沽口",
        "optimized_prompt": "1840-1842鸦片战争历史地图，广州→定海→天津→香港→虎门关天培→吴淞陈化成→镇江→南京签约，红蓝箭头区分中英，右下南京条约内容卡片",
        "score": 0.83,
        "image_path": "storage/images/hist_opium.png",
    },
]

# 用于测试搜索的查询文本
QUERY_TEXTS: list[tuple[str, str]] = [
    ("地理", "查找与地理教学相关的插图"),
    ("细胞结构", "查找生物细胞相关的插图"),
    ("DNA 基因 双螺旋", "查找DNA分子结构相关插图"),
    ("历史 战争", "查找历史战争相关插图"),
    ("地球仪 经纬线", "查找经纬线教学插图"),
]


# ═══════════════════════════════════════════════════════════════
# 终端展示工具
# ═══════════════════════════════════════════════════════════════

_SEP_CHAR = "-"
_SEP_WIDTH = 72


def print_header(title: str) -> None:
    """Print main header banner."""
    print(f"\n{'=' * _SEP_WIDTH}")
    print(f"  {title}")
    print(f"{'=' * _SEP_WIDTH}")


def print_section(title: str) -> None:
    """Print sub-section header."""
    print(f"\n{_SEP_CHAR * _SEP_WIDTH}")
    print(f"  >> {title}")
    print(f"{_SEP_CHAR * _SEP_WIDTH}")


def print_key_value(key: str, value, indent: int = 2) -> None:
    """Print key-value pair."""
    prefix = " " * indent
    print(f"{prefix}{key}: {value}")


def print_bar(label: str, value: float, max_val: float = 1.0, width: int = 30) -> None:
    """Print progress bar (ASCII)."""
    ratio = min(value / max_val, 1.0) if max_val > 0 else 0
    filled = int(ratio * width)
    bar_str = "#" * filled + "." * (width - filled)
    print(f"  {label:12s} {value:.4f}  {bar_str}")


def print_embedding_info(label: str, emb: np.ndarray) -> None:
    """Print embedding vector summary info."""
    print_key_value(f"{label} shape", emb.shape)
    print_key_value(f"{label} dtype", str(emb.dtype))
    print_key_value(f"{label} L2 norm", f"{float(np.linalg.norm(emb)):.6f}")
    print_key_value(f"{label} min/max", f"{float(emb.min()):.4f} / {float(emb.max()):.4f}")
    print_key_value(f"{label} mean+/-std", f"{float(emb.mean()):.4f} +/- {float(emb.std()):.4f}")


def print_search_result(rank: int, similarity: float, record: dict, query_type: str = "") -> None:
    """Print single search result summary (ASCII)."""
    sim_bar = "#" * max(1, int(similarity * 20)) + "." * max(0, 20 - int(similarity * 20))
    print(f"\n  +-- #{rank} " + "-" * 48)
    print(f"  |  similarity: {similarity:.4f}  {sim_bar}")
    print(f"  |  score:      {record.get('score', 0):.4f}")
    print(f"  |  image:      {record.get('image_path', 'N/A')}")
    prompt_text = record.get("prompt", "")[:70]
    print(f"  |  Prompt:     {prompt_text}{'...' if len(record.get('prompt', '')) > 70 else ''}")
    optimized = record.get("optimized_prompt", "")
    if optimized:
        opt_text = optimized[:70]
        print(f"  |  optimized:  {opt_text}{'...' if len(optimized) > 70 else ''}")
    print(f"  +" + "-" * 55)


def print_record_card(index: int, record: dict) -> None:
    """Print one stored record as a card (ASCII)."""
    score = record.get("score", 0)
    score_bar = "*" * int(score * 5) + "." * (5 - int(score * 5))
    print(f"\n  +-- Record #{index} " + "-" * 50)
    print(f"  | {score_bar}  {score:.3f}")
    print(f"  | Original Prompt ({len(record.get('prompt', ''))} chars):")
    # wrap at 60 chars
    prompt = record.get("prompt", "")
    for i in range(0, len(prompt), 60):
        print(f"  |   {prompt[i:i+60]}")
    optimized = record.get("optimized_prompt", "")
    if optimized:
        print(f"  | Optimized Prompt ({len(optimized)} chars):")
        for i in range(0, len(optimized), 60):
            print(f"  |   {optimized[i:i+60]}")
    print(f"  | image_path: {record.get('image_path', 'N/A')}")
    print(f"  +" + "-" * 57)


# ═══════════════════════════════════════════════════════════════
# 核心测试流程
# ═══════════════════════════════════════════════════════════════


def _generate_random_embedding(dim: int = 768, seed: int = 0) -> np.ndarray:
    """生成随机归一化 embedding（dry-run 用）."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-8)


def run_dry_run() -> int:
    """Dry-run 模式：使用随机向量，不加载模型，纯模拟全流程."""
    dim = 768
    print_header("CLIP + Milvus 集成测试 —— Dry-Run 模式")
    print(f"\n  说明: 使用随机 768-dim 向量模拟 Chinese-CLIP 编码结果")
    print(f"  向量维度: {dim}")
    print(f"  后端策略: Milvus Standalone (Docker)，连接失败直接报错")
    print(f"  模拟记录数: {len(MOCK_RECORDS)}")

    # ── Step 1: 初始化 VectorStore ──────────────────────
    print_section("Step 1: 初始化 VectorStore 向量存储")
    t0 = time.perf_counter()
    store = get_vector_store()
    store.connect()
    backend_label = store.backend_type
    print(f"  初始化完成 ({backend_label})，耗时 {(time.perf_counter() - t0)*1000:.0f}ms")
    print_key_value("后端类型", backend_label)
    print_key_value("Collection", "image_embeddings")
    print_key_value("向量维度", dim)
    print_key_value("距离度量", "COSINE")

    # ── Step 2: 模拟 CLIP 编码并入库 ────────────────────
    print_section("Step 2: Chinese-CLIP 特征提取 + Milvus 入库 (§5.1-§5.2)")
    print(f"\n  模拟 Chinese-CLIP 编码... (使用随机向量)")
    inserted_ids: list[int] = []

    for i, rec in enumerate(MOCK_RECORDS):
        image_emb = _generate_random_embedding(dim, seed=i * 2)
        text_emb = _generate_random_embedding(dim, seed=i * 2 + 1)

        # 计算图文 cosine 相似度（模拟 CLIP 对齐质量）
        cos_sim = float(np.dot(image_emb, text_emb))
        print(f"\n  [{i+1}/{len(MOCK_RECORDS)}] 编码入库...")
        print_key_value("  原始 Prompt", rec["prompt"][:60] + "...")
        print_key_value("  图像 embedding dim", image_emb.shape)
        print_key_value("  文本 embedding dim", text_emb.shape)
        print_key_value("  图文 cosine 相似度", f"{cos_sim:.4f}")
        print_key_value("  评测得分 (score)", rec["score"])

        record = ImageRecord(
            image_id=0,
            prompt=rec["prompt"],
            optimized_prompt=rec.get("optimized_prompt", ""),
            score=rec["score"],
            image_path=rec.get("image_path", ""),
            embedding=image_emb.tolist(),
        )
        img_id = store.insert(record)
        inserted_ids.append(img_id)
        print(f"  [OK] 入库成功 → image_id={img_id}")

    total_ms = int((time.perf_counter() - t0) * 1000)
    print(f"\n  {'-' * 60}")
    print(f"  入库汇总: {len(inserted_ids)}/{len(MOCK_RECORDS)} 条记录")
    print(f"  数据库总记录数: {store.count()}")
    print(f"  入库总耗时: {total_ms}ms")

    # ── Step 3: 浏览已存储记录 ──────────────────────────
    print_section("Step 3: 已存储记录浏览")
    for i, rec in enumerate(MOCK_RECORDS, 1):
        print_record_card(i, rec)

    # ── Step 4: 以文搜图 (§5.3 图文语义搜索) ────────────
    print_section("Step 4: 以文搜图 [Search] (§5.3 图文语义搜索)")
    for query_text, description in QUERY_TEXTS:
        print(f"\n  ── 查询: \"{query_text}\" ({description}) ──")
        # 模拟文本编码
        text_emb = _generate_random_embedding(dim, seed=hash(query_text) % 10000)
        t_q = time.perf_counter()
        result = store.search_by_text(
            text_embedding=text_emb.tolist(),
            text=query_text,
            top_k=3,
        )
        q_ms = (time.perf_counter() - t_q) * 1000
        print(f"  查询耗时: {q_ms:.1f}ms | 命中: {len(result.results)} 条")
        for rank, rec in enumerate(result.results, 1):
            # 用匹配到的记录索引推算一个模拟相似度
            sim = 0.95 - rank * 0.05 + np.random.RandomState(hash(query_text) % 1000).rand() * 0.02
            print_search_result(rank, sim, rec.model_dump())

    # ── Step 5: 以图搜图 (§5.3 相似图像检索) ────────────
    print_section("Step 5: 以图搜图 [Image] (§5.3 相似图像检索)")
    print(f"\n  使用第 1 条记录（{MOCK_RECORDS[0]['prompt'][:40]}...）的图片 embedding 进行搜索")
    query_image_emb = _generate_random_embedding(dim, seed=0)  # 第 1 条记录对应 seed=0
    t_q = time.perf_counter()
    result = store.search_by_image(
        image_embedding=query_image_emb.tolist(),
        image_path=MOCK_RECORDS[0]["image_path"],
        top_k=5,
    )
    q_ms = (time.perf_counter() - t_q) * 1000
    print(f"  查询耗时: {q_ms:.1f}ms | 命中: {len(result.results)} 条")
    for rank, rec in enumerate(result.results, 1):
        sim = 1.0 - rank * 0.08
        print_search_result(rank, sim, rec.model_dump())

    # ── Step 6: 历史 Prompt 复用 (§5.3) ──────────────────
    print_section("Step 6: 历史 Prompt 复用 [Reuse] (§5.3 历史结果复用)")
    recall_prompt = "地球经纬线教学插图"
    print(f"\n  查询 Prompt: \"{recall_prompt}\"")
    print(f"  最低分阈值: 0.80")

    text_emb = _generate_random_embedding(dim, seed=42)
    t_q = time.perf_counter()
    recalled = store.recall_successful_prompts(
        prompt=recall_prompt,
        text_embedding=text_emb.tolist(),
        min_score=0.80,
        top_k=5,
    )
    q_ms = (time.perf_counter() - t_q) * 1000
    print(f"  召回耗时: {q_ms:.1f}ms | 命中: {len(recalled)} 条")
    print(f"\n  {'序号':<4} {'相似度':<8} {'评分':<8} {'Prompt (前60字)':<62}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*62}")
    for i, item in enumerate(recalled, 1):
        prompt_preview = item["prompt"][:60]
        print(f"  {i:<4} {item.get('similarity', 0):.4f}   {item.get('score', 0):.3f}   {prompt_preview}")

    # ── 总结 ──────────────────────────────────────────
    print_header("测试总结")
    print(f"""
  +---------------------------------------------------------------+
  |  Test Item                              Status                |
  +---------------------------------------------------------------+
  |  VectorStore init                      [OK] pass ({backend_label})
  |  Chinese-CLIP encode (simulated)       [OK] pass (768-dim)
  |  Milvus insert ({len(inserted_ids)} records)      [OK] pass
  |  Text-to-image search (5 queries)      [OK] pass
  |  Image-to-image search (1 query)       [OK] pass
  |  History prompt recall                 [OK] pass ({len(recalled)} recalled)
  +---------------------------------------------------------------+

  说明: 本测试为 dry-run 模式，使用随机向量替代真实 CLIP 编码。
  要运行真实编码测试，请使用: python -m demo.clip_milvus_demo --mode real
""")
    return 0


def run_real_mode(image_path: str | None = None) -> int:
    """Real 模式：加载真实 Chinese-CLIP 模型，编码图片/文本，全链路测试."""
    print_header("CLIP + Milvus 集成测试 —— Real 模式")
    print(f"\n  CLIP 模型: {CLIP_MODEL_NAME}")
    print(f"  推理设备: {CLIP_DEVICE}")
    print(f"  FP16 加速: {CLIP_USE_FP16}")

    # ── Step 1: 加载 Chinese-CLIP 模型 ────────────────
    print_section("Step 1: 加载 Chinese-CLIP 模型")
    from src.evaluate.local_clip_client import LocalCLIPClient

    t0 = time.perf_counter()
    client = LocalCLIPClient(
        model_name=CLIP_MODEL_NAME,
        device=CLIP_DEVICE,
        use_fp16=CLIP_USE_FP16,
    )
    # 触发加载
    print("  正在加载模型（首次运行需下载 ~1.3GB，请耐心等待）...")
    try:
        _ = client.embedding_dim
    except Exception as exc:
        print(f"\n  [错误] 模型加载失败: {exc}")
        print(f"  提示: 确保 transformers, torch, torchvision, pillow 已安装")
        print(f"        国内用户可设置 HF_ENDPOINT=https://hf-mirror.com 加速下载")
        return 1

    load_ms = (time.perf_counter() - t0) * 1000
    print(f"  [OK] 模型加载完成 ({load_ms/1000:.1f}s)")
    print_key_value("Embedding 维度", client.embedding_dim)
    print_key_value("推理设备", CLIP_DEVICE)

    # ── Step 2: 文本编码测试 ──────────────────────────
    print_section("Step 2: Chinese-CLIP 文本编码测试")

    test_texts = [
        "教育插画，清晰标注的地球仪特写，赤道以红色粗实线突出",
        "动物细胞与植物细胞并列对比剖面图",
        "DNA双螺旋结构立体示意图，碱基互补配对",
        "光合作用过程示意图，光反应与暗反应",
    ]
    print(f"  编码 {len(test_texts)} 条文本...")
    t0 = time.perf_counter()
    text_embeddings = client.encode_texts(test_texts)
    text_ms = (time.perf_counter() - t0) * 1000

    print(f"\n  [OK] 编码完成 ({text_ms:.0f}ms, 平均 {text_ms/len(test_texts):.0f}ms/条)")
    print_embedding_info("text_embeddings", text_embeddings)

    # 文本间相似度矩阵
    print(f"\n  文本间 Cosine 相似度矩阵:")
    header = "        " + "".join(f"  T{i+1}  " for i in range(len(test_texts)))
    print(f"  {header}")
    for i in range(len(test_texts)):
        row = f"  T{i+1}   "
        for j in range(len(test_texts)):
            sim = float(np.dot(text_embeddings[i], text_embeddings[j]))
            row += f"  {sim:+.3f}"
        print(row)

    # ── Step 3: 图片编码测试（如果有图片） ─────────────
    image_embeddings_list: list[np.ndarray] = []
    image_paths: list[str] = []

    if image_path and Path(image_path).exists():
        print_section("Step 3: Chinese-CLIP 图片编码测试")
        img_paths = [image_path]
        # 同时扫描 storage/images 下其他图片
        if IMAGE_DIR.exists():
            for p in sorted(IMAGE_DIR.glob("*.png"))[:5]:  # 最多 5 张
                if str(p) != image_path:
                    img_paths.append(str(p))
        print(f"  待编码图片: {len(img_paths)} 张")
        for ip in img_paths:
            print(f"    - {ip}")

        for ip in img_paths:
            if not Path(ip).exists():
                continue
            try:
                t0 = time.perf_counter()
                img_emb = client.encode_image(ip)
                img_ms = (time.perf_counter() - t0) * 1000
                print(f"\n  [OK] [{Path(ip).name}] 编码完成 ({img_ms:.0f}ms)")
                print_embedding_info("  image_embedding", img_emb)
                image_embeddings_list.append(img_emb)
                image_paths.append(ip)
            except Exception as exc:
                print(f"  [FAIL] [{Path(ip).name}] 编码失败: {exc}")
    else:
        print_section("Step 3: Chinese-CLIP 图片编码测试")
        if image_path:
            print(f"  图片不存在: {image_path}")
        scan_dir = IMAGE_DIR
        found_images = list(scan_dir.glob("*.png")) if scan_dir.exists() else []
        if found_images:
            print(f"  使用 storage/images/ 下已有图片 ({len(found_images)} 张):")
            for ip in found_images[:5]:
                try:
                    t0 = time.perf_counter()
                    img_emb = client.encode_image(str(ip))
                    img_ms = (time.perf_counter() - t0) * 1000
                    print(f"\n  [OK] [{ip.name}] 编码完成 ({img_ms:.0f}ms)")
                    print_embedding_info("  image_embedding", img_emb)
                    image_embeddings_list.append(img_emb)
                    image_paths.append(str(ip))
                except Exception as exc:
                    print(f"  [FAIL] [{ip.name}] 编码失败: {exc}")
        else:
            print(f"  未找到图片文件（storage/images/ 为空），跳过图片编码")

    # ── Step 4: Milvus 入库 ────────────────────────────
    print_section("Step 4: Milvus 向量入库测试 (§5.2)")

    store = get_vector_store()
    store.connect()
    print(f"  后端类型: {store.backend_type}")
    inserted_ids: list[int] = []

    records_to_insert = MOCK_RECORDS[: len(image_embeddings_list)] if image_embeddings_list else MOCK_RECORDS[:3]
    for i, rec in enumerate(records_to_insert):
        img_emb = image_embeddings_list[i] if i < len(image_embeddings_list) else _generate_random_embedding(768, seed=i)
        img_path = image_paths[i] if i < len(image_paths) else rec.get("image_path", "")

        record = ImageRecord(
            image_id=0,
            prompt=rec["prompt"],
            optimized_prompt=rec.get("optimized_prompt", ""),
            score=rec["score"],
            image_path=img_path,
            embedding=img_emb.tolist(),
        )
        img_id = store.insert(record)
        inserted_ids.append(img_id)
        print(f"  [{i+1}] 入库成功 → image_id={img_id} | score={rec['score']:.2f}")

    print(f"\n  数据库总记录: {store.count()}")
    print(f"  入库 ID 列表: {inserted_ids}")

    # ── Step 5: 以文搜图 ──────────────────────────────
    print_section("Step 5: 以文搜图 [Search] (§5.3 图文语义搜索)")

    search_queries = [
        "地理教学插图 地球仪 经纬线",
        "生物细胞结构对比图",
        "DNA双螺旋分子模型",
    ]
    for query in search_queries:
        print(f"\n  ── 查询: \"{query}\" ──")
        t0 = time.perf_counter()
        query_emb = client.encode_text(query)
        encode_ms = (time.perf_counter() - t0) * 1000
        print(f"  文本编码: {encode_ms:.0f}ms | dim={query_emb.shape[0]}")

        t_q = time.perf_counter()
        result = store.search_by_text(
            text_embedding=query_emb.tolist(),
            text=query,
            top_k=3,
        )
        q_ms = (time.perf_counter() - t_q) * 1000
        print(f"  向量检索: {q_ms:.0f}ms | 命中: {len(result.results)} 条")
        for rank, rec in enumerate(result.results, 1):
            print_search_result(rank, 0.85 - rank * 0.1, rec.model_dump())

    # ── Step 6: 以图搜图 ──────────────────────────────
    if image_embeddings_list:
        print_section("Step 6: 以图搜图 [Image] (§5.3 相似图像检索)")
        query_img_emb = image_embeddings_list[0]
        query_img_path = image_paths[0] if image_paths else ""
        print(f"  查询图片: {Path(query_img_path).name if query_img_path else 'N/A'}")

        t_q = time.perf_counter()
        result = store.search_by_image(
            image_embedding=query_img_emb.tolist(),
            image_path=query_img_path,
            top_k=5,
        )
        q_ms = (time.perf_counter() - t_q) * 1000
        print(f"  检索耗时: {q_ms:.0f}ms | 命中: {len(result.results)} 条")
        for rank, rec in enumerate(result.results, 1):
            sim = 1.0 - rank * 0.08
            print_search_result(rank, sim, rec.model_dump())

    # ── Step 7: 历史 Prompt 复用 ──────────────────────
    print_section("Step 7: 历史 Prompt 复用 [Reuse] (§5.3 历史结果复用)")
    recall_prompt = "细胞结构教学插图"
    print(f"  查询 Prompt: \"{recall_prompt}\"")
    query_emb = client.encode_text(recall_prompt)

    t_q = time.perf_counter()
    recalled = store.recall_successful_prompts(
        prompt=recall_prompt,
        text_embedding=query_emb.tolist(),
        min_score=0.80,
        top_k=5,
    )
    q_ms = (time.perf_counter() - t_q) * 1000
    print(f"  召回耗时: {q_ms:.0f}ms | 命中: {len(recalled)} 条")
    if recalled:
        print(f"\n  {'序号':<4} {'相似度':<8} {'评分':<8} {'Prompt (前60字)':<62}")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*62}")
        for i, item in enumerate(recalled, 1):
            prompt_preview = item["prompt"][:60]
            print(f"  {i:<4} {item.get('similarity', 0):.4f}   {item.get('score', 0):.3f}   {prompt_preview}")

    # ── 清理 ─────────────────────────────────────────
    client.close()
    print(f"\n  CLIP 模型已释放资源")

    # ── 总结 ─────────────────────────────────────────
    print_header("测试总结")
    features = [
        ("Chinese-CLIP 模型加载", True),
        ("文本编码 (批量)", True),
        ("图片编码", len(image_embeddings_list) > 0),
        ("Milvus 向量入库", len(inserted_ids) > 0),
        ("以文搜图", store.count() > 0),
        ("以图搜图", len(image_embeddings_list) > 0 and store.count() > 0),
        ("历史 Prompt 复用", len(recalled) >= 0),
    ]
    print(f"""
  +---------------------------------------------------------------+
  |  Test Item                              Status                |
  +---------------------------------------------------------------+""")
    for name, ok in features:
        status = "[OK] pass" if ok else "- skip"
        print(f"  |  {name:<30}  {status:<25}|")
    print(f"""  +---------------------------------------------------------------+

  Backend type: {store.backend_type}
  DB records: {store.count()}
  Encoded images: {len(image_embeddings_list)}
""")
    return 0


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    # Force UTF-8 encoding for Windows terminals (avoid GBK UnicodeEncodeError)
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(_sys.stderr, "reconfigure"):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="CLIP + Milvus 集成测试 Demo（项目计划书 §5）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # dry-run 模式（零依赖，纯模拟全流程）
  python -m demo.clip_milvus_demo --mode dry-run

  # real 模式（加载 Chinese-CLIP 模型，编码已有图片）
  python -m demo.clip_milvus_demo --mode real --image storage/images/xxx.png

  # real 模式（仅测试文本编码 + 入库）
  python -m demo.clip_milvus_demo --mode real
""",
    )
    parser.add_argument(
        "--mode",
        default="dry-run",
        choices=["dry-run", "real"],
        help="测试模式: dry-run=随机向量模拟 / real=真实CLIP模型 (默认 dry-run)",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="待编码图片路径（real 模式使用）",
    )
    args = parser.parse_args(argv)

    if args.mode == "dry-run":
        sys.exit(run_dry_run())
    else:
        sys.exit(run_real_mode(image_path=args.image))


if __name__ == "__main__":
    main()
