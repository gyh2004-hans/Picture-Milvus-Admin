"""闭环 Pipeline —— CLIP 检索 + 评测迭代 + 入库.

流程:
  CLIP向量检索 → 命中则直接复用 → 未命中则 Evaluate循环 → 生图 → 入库

实现项目计划书 §4 "闭环迭代机制" 的停止条件逻辑:
  - §4.2 终止规则: (1) overall_score ≥ eval_threshold (默认 0.82) → 提前结束
                 (2) |本轮分 - 上轮分| < 0.01 → 收敛停止
                 (3) 分数较历史最佳下降 > SCORE_ROLLBACK_DELTA → 回滚 best_prompt 并停止
                 (4) 达到最大迭代次数 (默认 3 次) → 停止

流程（对齐项目计划书 §4）:
  CLIP检索 → 命中复用 / 未命中 → Draw → VLMEvaluator.evaluate() → 检查停止
  → PromptRefiner.analyze() → LLMAdjuster.adjust() → 循环 → CLIP编码入库
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

from src.config import (
    CLIP_DEVICE,
    CLIP_MODEL_NAME,
    CLIP_USE_FP16,
    CONVERGENCE_DELTA,
    MAX_ITERATIONS,
    MAX_ISSUES_PER_ROUND,
    SCORE_ROLLBACK_DELTA,
)
from src.draw import DRAWER_REGISTRY
from src.evaluate import VLMEvaluator
from src.models.schemas import (
    ImageRecord,
    PipelineIteration,
    PipelineRequest,
    PipelineResponse,
)
from src.refiner import LLMAdjuster, PromptRefiner
from src.storage import RecordStore

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# 分类自动检测关键词表（v6: 从 prompt 文本推断分类，自动分区限定）
#
# 以 config.CATEGORIES 中的分类名为 key，value 为扩展关键词列表。
# 运行时会动态合并 config 中的实际分类名，确保与分区体系一致。
# ══════════════════════════════════════════════════════════════
_CATEGORY_KEYWORD_SYNONYMS: dict[str, list[str]] = {
    # ── 自然风光 ──
    "自然风光": ["山水", "河流", "山脉", "湖泊", "海洋", "森林", "日出", "日落",
              "天空", "云彩", "草原", "沙漠", "雪景", "瀑布", "海滩", "自然",
              "风光", "风景", "田园", "峡谷", "冰川", "极光", "星空", "夜景"],
    # ── 人物人像 ──
    "人物人像": ["肖像", "人脸", "自拍", "合影", "模特", "半身像", "全身像",
              "头像", "证件照", "街拍", "人物", "人物照", "人像", "儿童",
              "婴儿", "老人", "家庭照", "写真"],
    # ── 城市建筑 ──
    "城市建筑": ["建筑", "房屋", "大楼", "桥梁", "室内", "装修", "家具", "景观",
              "园林", "城市", "街道", "广场", "建筑外观", "摩天楼", "地标",
              "教堂", "寺庙", "古镇", "现代建筑", "房屋设计"],
    # ── 美食饮品 ──
    "美食饮品": ["美食", "食物", "料理", "烹饪", "餐厅", "菜品", "甜点", "饮品",
              "水果", "蔬菜", "面食", "火锅", "烧烤", "西餐", "日料", "中餐",
              "咖啡", "茶", "蛋糕", "面包", "披萨", "拉面", "刺身"],
    # ── 动植物 ──
    "动植物": ["动物", "宠物", "猫", "狗", "鸟", "鱼", "野生动物", "昆虫", "狮子",
             "老虎", "熊猫", "兔子", "马", "蝴蝶", "鸟类", "海洋生物", "植物",
             "花", "树", "叶子", "花卉", "花园", "玫瑰", "樱花", "森林动物"],
    # ── 办公商务 ──
    "办公商务": ["办公", "商务", "会议室", "办公室", "写字楼", "笔记本", "键盘",
              "鼠标", "文具", "文件夹", "图表", "报告", "PPT", "白板", "名片"],
    # ── 数码科技 ──
    "数码科技": ["科技", "电路", "芯片", "机器人", "代码", "程序", "AI", "数据",
              "计算机", "手机", "电子", "数码", "网络", "软件", "硬件", "人工智能",
              "数码产品", "VR", "AR", "无人机", "3D打印"],
    # ── 服饰穿搭 ──
    "服饰穿搭": ["服饰", "穿搭", "衣服", "裙子", "裤子", "鞋子", "包包", "首饰",
              "手表", "墨镜", "帽子", "围巾", "时尚", "潮流", "时装", "礼服"],
    # ── 家居家装 ──
    "家居家装": ["家居", "家装", "客厅", "卧室", "厨房", "卫生间", "沙发", "床",
              "灯具", "窗帘", "地毯", "装修风格", "宜家", "北欧风", "日式"],
    # ── 节日庆典 ──
    "节日庆典": ["节日", "庆典", "春节", "圣诞", "元旦", "中秋", "国庆", "婚礼",
              "生日", "派对", "烟花", "灯笼", "彩带", "气球", "礼花"],
    # ── 手绘插画 ──
    "手绘插画": ["手绘", "插画", "绘画", "油画", "水彩", "素描", "雕塑", "书法",
              "抽象", "海报", "设计", "艺术品", "美术", "涂鸦", "卡通", "漫画",
              "二次元", "动画", "CG", "板绘", "艺术"],
    # ── 纹理背景 ──
    "纹理背景": ["纹理", "图案", "背景", "色块", "渐变", "标识", "图标", "壁纸",
              "材质", "大理石", "木纹", "金属", "布料", "纸张"],
    # ── 交通出行 ──
    "交通出行": ["交通", "出行", "汽车", "火车", "飞机", "地铁", "公交", "自行车",
              "电动车", "高铁", "轮船", "高速公路", "航拍道路", "立交桥"],
    # ── 教育培训 ──
    "教育培训": ["教育", "培训", "学校", "教室", "课本", "黑板", "学生", "老师",
              "图书", "学习", "考试", "实验", "图书馆", "讲座"],
    # ── 运动休闲 ──
    "运动休闲": ["运动", "休闲", "跑步", "篮球", "足球", "游泳", "健身", "瑜伽",
              "滑雪", "登山", "骑行", "高尔夫", "网球", "冲浪", "太极"],
}

# 向后兼容旧分类名 → 新分类名映射（当 prompt 用旧名称关键词时能映射到新分类）
_LEGACY_CATEGORY_ALIASES: dict[str, str] = {
    "风景": "自然风光",
    "人物": "人物人像",
    "建筑": "城市建筑",
    "美食": "美食饮品",
    "动物": "动植物",
    "科技": "数码科技",
    "艺术": "手绘插画",
    "服饰": "服饰穿搭",
    "家居": "家居家装",
    "节日": "节日庆典",
    "交通": "交通出行",
    "教育": "教育培训",
    "运动": "运动休闲",
    "纹理": "纹理背景",
    "办公": "办公商务",
}


class ImagePipeline:
    """闭环控制器 —— CLIP 检索 + 评测迭代 + 入库.

    流程:
      CLIP 检索 → 命中复用（相似度≥阈值）→ 未命中则 Evaluate 循环 → 生图 → 入库
    """

    def __init__(self) -> None:
        self._evaluator = VLMEvaluator()
        self._refiner = PromptRefiner()
        self._adjuster = LLMAdjuster()
        # Mode 3 懒加载组件
        self._clip_client: Optional[object] = None  # LocalCLIPClient
        self._vector_store: Optional[object] = None  # VectorStore
        self._record_store = RecordStore()

    # ── RecordStore 兜底存储 ──────────────────────────

    def _save_to_record_store(
        self, request: PipelineRequest, response: PipelineResponse, model: str,
    ) -> bool:
        """将生成/复用的图片信息存入本地 JSON RecordStore，确保即使 Milvus 不可用也有记录.

        Returns:
            True 保存成功，False 失败.
        """
        try:
            self._record_store.create(
                prompt=request.prompt,
                model=model,
                image_path=response.final_image_path,
                metadata={
                    "final_prompt": response.final_prompt,
                    "final_score": response.final_score,
                    "stopped_reason": response.stopped_reason,
                    "db_record_id": response.db_record_id,
                    "reused_from_record_id": response.reused_from_record_id,
                },
            )
            logger.info(
                "Pipeline record_store.save | path=%s score=%.3f reason=%s",
                response.final_image_path, response.final_score,
                response.stopped_reason,
            )
            return True
        except Exception as exc:
            logger.error("Pipeline record_store.save failed: %s", exc)
            return False

    # ── 公开入口：分发器 ──────────────────────────────────

    async def run(self, request: PipelineRequest) -> PipelineResponse:
        """执行 CLIP 检索 + 评测迭代 + 入库的完整闭环.

        Args:
            request: 含原始 prompt + 模型选择 + 停止参数.

        Returns:
            PipelineResponse: 含最终图片 + 评分 + 迭代历史.

        Raises:
            ValueError: 未知模型.
        """
        if request.model not in DRAWER_REGISTRY:
            raise ValueError(
                f"Unknown model: {request.model}. "
                f"Available: {list(DRAWER_REGISTRY)}"
            )

        return await self._run_clip_enrich(request)

    # ── Evaluate-Refine 循环（核心） ───────────────────

    async def _run_evaluate_loop(
        self,
        request: PipelineRequest,
        initial_prompt: str | None = None,
    ) -> PipelineResponse:
        """Draw → Evaluate → 停止检查 → Refine → Adjust → 循环.

        停止条件（对齐项目计划书 §4.2 + 最小编辑优化）:
          1. overall_score ≥ eval_threshold → "threshold_met"
          2. |本轮分 - 上轮分| < CONVERGENCE_DELTA → "converged"
          3. 分数较历史最佳下降 > SCORE_ROLLBACK_DELTA → "score_regression"（回滚 best_prompt）
          4. 达到 max_iterations → "max_iterations"

        每轮仅处理 top-k severe issues；LLM 最小编辑模式；全程跟踪 best_prompt/best_score.

        Args:
            request: 含原始 prompt + 模型选择 + 停止参数.
            initial_prompt: 起始 prompt（可选覆盖 request.prompt）.
                           为 None 时使用 request.prompt.

        Returns:
            PipelineResponse: 含最终图片 + 评分 + 迭代历史.
        """
        prompt = initial_prompt if initial_prompt else request.prompt
        model = request.model
        max_iter = request.max_iterations
        threshold = request.eval_threshold

        drawer = DRAWER_REGISTRY[model]

        history: list[PipelineIteration] = []
        prev_score = 0.0
        stopped_reason = "max_iterations"

        best_prompt = prompt
        best_score = 0.0
        best_image_path = ""
        best_history_idx = 0

        for iteration in range(1, max_iter + 1):
            logger.info(
                "Pipeline iteration %d/%d | prompt_len=%d",
                iteration, max_iter, len(prompt),
            )

            # Step 1: 生图
            image_path = await drawer.generate(prompt)

            # Step 2: VLM 五维度评测
            eval_result = await self._evaluator.evaluate(prompt, image_path)
            score = eval_result.overall_score

            history.append(
                PipelineIteration(
                    iteration=iteration,
                    prompt=prompt,
                    image_path=image_path,
                    overall_score=score,
                    dimension_scores=eval_result.dimension_scores,
                    issues=eval_result.issues,
                    missing_elements=eval_result.missing_elements,
                    suggestions=eval_result.suggestions,
                )
            )

            # ── 更新历史最佳 ──
            if score > best_score:
                best_score = score
                best_prompt = prompt
                best_image_path = image_path
                best_history_idx = len(history) - 1

            # ── 停止条件 1: 达标 ──
            if score >= threshold:
                stopped_reason = "threshold_met"
                logger.info(
                    "Pipeline stop: threshold_met (score=%.3f >= %.2f)",
                    score, threshold,
                )
                break

            # ── 停止条件 2: 分数崩盘 → 回滚 best_prompt ──
            if iteration > 1 and score < best_score - SCORE_ROLLBACK_DELTA:
                stopped_reason = "score_regression"
                logger.info(
                    "Pipeline stop: score_regression (score=%.3f best=%.3f delta=%.3f > %.2f) "
                    "→ rollback to best_prompt",
                    score, best_score, best_score - score, SCORE_ROLLBACK_DELTA,
                )
                break

            # ── 停止条件 3: 收敛 ──
            if iteration > 1:
                delta = abs(score - prev_score)
                if delta < CONVERGENCE_DELTA:
                    stopped_reason = "converged"
                    logger.info(
                        "Pipeline stop: converged (delta=%.4f < %.3f)",
                        delta, CONVERGENCE_DELTA,
                    )
                    break

            prev_score = score

            # ── top-k severe feedback（每轮最多 MAX_ISSUES_PER_ROUND 项） ──
            top_issues, top_missing, _top_suggestions = (
                PromptRefiner.select_top_feedback(
                    eval_result, max_items=MAX_ISSUES_PER_ROUND,
                )
            )

            # ── Step 3: 策略分析（§3.3，内部已 top-k 截断） ──
            strategy = self._refiner.analyze(
                eval_result=eval_result,
                origin_prompt=prompt,
            )

            # ── Step 4: LLM 最小编辑（§3.4） ──
            if not strategy.strategies:
                logger.info(
                    "Pipeline: no strategies generated, keeping current prompt"
                )
                history[-1].optimized_prompt = prompt
                history[-1].changes_summary = "No optimization strategies identified"
                continue

            try:
                adjuster_output = await self._adjuster.adjust(
                    origin_prompt=prompt,
                    strategy=strategy,
                    issues=top_issues,
                    missing_elements=top_missing,
                    overall_score=score,
                    eval_threshold=threshold,
                )
            except (RuntimeError, ValueError) as exc:
                logger.error("Pipeline: LLMAdjuster failed: %s", exc)
                # LLM 调整失败，继续下一轮用原 prompt
                history[-1].optimized_prompt = prompt
                history[-1].changes_summary = f"LLM adjust failed: {exc}"
                continue

            history[-1].optimized_prompt = adjuster_output.optimized_prompt
            history[-1].changes_summary = adjuster_output.changes_summary

            # 检查 LLM 是否产生了有效变更
            if adjuster_output.optimized_prompt.strip() == prompt.strip():
                logger.info(
                    "Pipeline: LLMAdjuster returned identical prompt, "
                    "no effective change possible"
                )

            prompt = adjuster_output.optimized_prompt

        # ── 最终结果：取历史最佳（score_regression / max_iterations 时尤为重要） ──
        best = history[best_history_idx]
        logger.info(
            "Pipeline finished | reason=%s iterations=%d best_score=%.3f final_score=%.3f",
            stopped_reason, len(history), best_score, best.overall_score,
        )
        return PipelineResponse(
            final_image_path=best_image_path or best.image_path,
            final_prompt=best_prompt,
            final_score=best_score,
            total_iterations=len(history),
            history=history,
            stopped_reason=stopped_reason,
        )

    # ── CLIP 检索复用 + 循环 + 入库 ────────────────────

    async def _run_clip_enrich(self, request: PipelineRequest) -> PipelineResponse:
        """语义检索 → 命中复用 / 未命中则生图.

        v5 更新: 命中复用检测改为自然语言语义检索链路 (search_by_semantic),
        加权排序 0.7*semantic + 0.2*image + 0.1*tags, 自动限定学科分区.

        流程:
          1. CLIP 编码用户自然语言 → text_embedding
          2. prompt 精确匹配（同 prompt 重跑 → 复用最高分记录，跳过 CLIP 跨模态检索）
          3. VectorStore.search_by_semantic() 语义检索（近似 prompt，自动限定学科分区）
          4. 若 top-1 semantic_similarity ≥ reuse_threshold(默认 0.77) 且图片文件存在 → 直接复用
          5. 未命中 → 使用原始 prompt 执行 Evaluate-Refine 循环 → 生图
          6. 新生成的图片 → CLIP 编码 → 存入向量库（含 text + image embedding）

        容错: 任何 CLIP/VectorStore 步骤失败时，自动降级为 Evaluate-Refine 循环行为.

        Args:
            request: 含 prompt + model + CLIP 检索参数 + reuse_threshold.

        Returns:
            PipelineResponse: 含最终图片 + 评分 + 迭代历史 + 入库信息.
        """
        logger.info(
            "Pipeline mode=clip_enrich | model=%s prompt_len=%d",
            request.model, len(request.prompt),
        )

        # ── Step 1-2: 懒加载 CLIP + VectorStore ─────────
        matched_prompts: list[dict] = []

        try:
            # CLIP 模型加载 / VectorStore 连接为阻塞 I/O，卸载到线程避免阻塞事件循环
            await asyncio.to_thread(self._lazy_init_clip)
            await asyncio.to_thread(self._lazy_init_vector_store)
        except Exception as exc:
            logger.warning(
                "Pipeline clip_enrich: init failed (%s), falling back to evaluate_loop",
                exc,
            )
            response = await self._run_evaluate_loop(request)
            response.stored_in_records = self._save_to_record_store(request, response, request.model)
            return response

        # ── Step 3: CLIP 编码文本 ───────────────────────
        try:
            t0 = time.perf_counter()
            # CLIP 推理为 CPU/GPU 密集型同步调用，卸载到线程
            text_embedding = await asyncio.to_thread(
                self._clip_client.encode_text, request.prompt,
            )
            logger.info(
                "clip_enrich.encode_text | dim=%d duration_ms=%d",
                len(text_embedding),
                int((time.perf_counter() - t0) * 1000),
            )
        except Exception as exc:
            logger.warning(
                "Pipeline clip_enrich: CLIP encode_text failed (%s), "
                "falling back to evaluate_loop",
                exc,
            )
            response = await self._run_evaluate_loop(request)
            response.stored_in_records = self._save_to_record_store(request, response, request.model)
            return response

        # ── Step 4: 解析分类分区（v6: 同时支持 subject 和 category） ──
        search_subject = request.category or request.subject
        if not search_subject:
            detected = self._detect_category_from_prompt(request.prompt)
            if detected:
                search_subject = detected
                logger.info(
                    "clip_enrich.search: auto-detected category=%s from prompt",
                    search_subject,
                )
            else:
                logger.warning(
                    "clip_enrich.search: category not set and could not be "
                    "auto-detected from prompt — searching across all partitions"
                )

        # ── Step 5: prompt 精确匹配（同 prompt 重跑优先复用） ──
        try:
            exact_hit = await asyncio.to_thread(
                self._vector_store.find_best_by_exact_prompt,
                request.prompt,
                search_subject,
                request.clip_min_score,
            )
            if exact_hit is not None:
                exact_image_path = exact_hit.get("image_path", "")
                if exact_image_path and Path(exact_image_path).is_file():
                    logger.info(
                        "clip_enrich.reuse_hit | match=exact_prompt record_id=%d "
                        "score=%.3f image_path=%s — 跳过生图，直接复用已有图片",
                        exact_hit.get("image_id"), exact_hit.get("score", 0.0),
                        exact_image_path,
                    )
                    reuse_response = PipelineResponse(
                        final_image_path=exact_image_path,
                        final_prompt=exact_hit.get("optimized_prompt")
                        or exact_hit.get("prompt", ""),
                        final_score=exact_hit.get("score", 0.0),
                        total_iterations=0,
                        history=[],
                        stopped_reason="reused",
                        db_record_id=exact_hit.get("image_id"),
                        matched_prompts=[exact_hit],
                        reused_from_record_id=exact_hit.get("image_id"),
                        stored_in_milvus=True,
                    )
                    reuse_response.stored_in_records = self._save_to_record_store(
                        request, reuse_response, request.model,
                    )
                    return reuse_response
        except Exception as exc:
            logger.warning(
                "Pipeline clip_enrich: exact prompt lookup failed (%s), "
                "continuing with semantic search",
                exc,
            )

        # ── Step 6: CLIP 语义检索（近似 prompt；v5: 加权排序 0.7*semantic + 0.2*image + 0.1*tags） ──
        try:
            semantic_result = await asyncio.to_thread(
                self._vector_store.search_by_semantic,
                query_text=request.prompt,
                semantic_embedding=text_embedding.tolist(),
                top_k=request.clip_top_k,
                subject=search_subject,
            )

            raw_results: list[dict] = semantic_result.get("results", [])

            # 按 min_score 过滤（用评测得分，与旧逻辑一致）
            matched = [
                r for r in raw_results
                if r.get("score", 0.0) >= request.clip_min_score
            ]

            logger.info(
                "clip_enrich.search_semantic | total=%d filtered=%d min_score=%.2f "
                "subject=%s partition_total=%d",
                len(raw_results), len(matched), request.clip_min_score,
                search_subject, semantic_result.get("total_in_partition", 0),
            )

            matched_prompts = matched  # search_by_semantic 返回已是 dict 列表
        except Exception as exc:
            logger.warning(
                "Pipeline clip_enrich: semantic search failed (%s), "
                "falling back to evaluate_loop",
                exc,
            )
            response = await self._run_evaluate_loop(request)
            response.stored_in_records = self._save_to_record_store(request, response, request.model)
            return response

        # ── Step 7: 语义检索命中复用（跨模态，仅用于近似 prompt） ──
        top_result = raw_results[0] if raw_results else None
        top_semantic_sim = top_result.get("semantic_similarity", 0.0) if top_result else 0.0
        top_image_path = top_result.get("image_path", "") if top_result else ""
        if (
            top_result is not None
            and top_semantic_sim >= request.reuse_threshold
            and top_image_path
            and Path(top_image_path).is_file()  # 文件必须存在才能复用
        ):
            logger.info(
                "clip_enrich.reuse_hit | match=semantic record_id=%d semantic_similarity=%.4f "
                "final_score=%.4f threshold=%.2f image_path=%s — "
                "跳过生图，直接复用已有图片",
                top_result.get("image_id"), top_semantic_sim,
                top_result.get("final_score", 0.0),
                request.reuse_threshold, top_image_path,
            )
            reuse_response = PipelineResponse(
                final_image_path=top_image_path,
                final_prompt=top_result.get("prompt", ""),
                final_score=top_result.get("score", 0.0),
                total_iterations=0,
                history=[],
                stopped_reason="reused",
                db_record_id=top_result.get("image_id"),
                matched_prompts=[top_result],
                reused_from_record_id=top_result.get("image_id"),
                stored_in_milvus=True,
            )
            reuse_response.stored_in_records = self._save_to_record_store(
                request, reuse_response, request.model,
            )
            return reuse_response

        # 未命中或文件缺失 → 继续生图流程
        if top_result is not None:
            logger.info(
                "clip_enrich.reuse_miss | top_semantic_sim=%.4f final_score=%.4f "
                "threshold=%.2f — 进入生图流程",
                top_semantic_sim, top_result.get("final_score", 0.0),
                request.reuse_threshold,
            )
        else:
            logger.info("clip_enrich.reuse_miss | no search results — 进入生图流程")

        # ── Step 8: 执行 Evaluate-Refine 循环（始终使用原始 prompt）──
        response = await self._run_evaluate_loop(request)

        # ── Step 9: VLM 图片内容解析（best-effort，不阻断后续流程）──
        vlm_category: str | None = None
        vlm_tags: list[str] = []
        vlm_main_objects: list[str] = []
        vlm_scene_description: str = ""
        vlm_style: str = ""
        vlm_color_palette: list[str] = []
        vlm_content_type: str = ""
        semantic_text: str = ""
        semantic_embedding: list[float] | None = None

        try:
            from src.milvus.image_content_parser import ImageContentParser
            parser = ImageContentParser()
            parse_result = await parser.parse(response.final_image_path)
            vlm_category = parse_result.category
            vlm_tags = parse_result.tags[:8]
            vlm_main_objects = parse_result.main_objects[:6]
            vlm_scene_description = parse_result.scene_description
            vlm_style = parse_result.style
            vlm_color_palette = parse_result.color_palette[:5]
            vlm_content_type = parse_result.content_type
            semantic_text = ImageContentParser.build_semantic_text(parse_result)
            logger.info(
                "clip_enrich.vlm_parse | category=%s content_type=%s "
                "objects=%d tags=%d semantic_text_len=%d",
                vlm_category, vlm_content_type,
                len(vlm_main_objects), len(vlm_tags), len(semantic_text),
            )
        except Exception as exc:
            logger.warning(
                "Pipeline clip_enrich: VLM content parsing failed (%s), "
                "using prompt-based category detection as fallback", exc,
            )
            fallback_category = search_subject or "其他"
            semantic_text = (
                f"分类: {fallback_category}\n"
                f"检索描述: {request.prompt}"
            )

        # 确定分区路由分类：VLM 解析 > prompt 自动检测 > API 传入
        effective_category = (
            vlm_category
            or search_subject
            or request.category
            or request.subject
        )
        # 归一化 category 到 config.CATEGORIES
        if effective_category and vlm_category is None:
            # 仅对非 VLM 来源的 category 做额外归一化（VLM 已通过 _normalize_category 归一化）
            from src.config import CATEGORIES
            if effective_category not in CATEGORIES:
                matched = None
                for cat in CATEGORIES:
                    if cat in effective_category or effective_category in cat:
                        matched = cat
                        break
                if matched:
                    effective_category = matched

        # ── Step 10: CLIP 编码最终图片 ────────────────────
        db_record_id: Optional[int] = None
        try:
            t0 = time.perf_counter()
            image_embedding = await asyncio.to_thread(
                self._clip_client.encode_image,
                response.final_image_path,
            )
            logger.info(
                "clip_enrich.encode_image | path=%s duration_ms=%d",
                response.final_image_path,
                int((time.perf_counter() - t0) * 1000),
            )
        except Exception as exc:
            logger.error(
                "Pipeline clip_enrich: encode_image failed for %s: %s",
                response.final_image_path, exc,
            )
            image_embedding = None

        # CLIP 编码 semantic_text → semantic_embedding（best-effort）
        if image_embedding is not None and semantic_text:
            try:
                t0 = time.perf_counter()
                sem_emb = await asyncio.to_thread(
                    self._clip_client.encode_text, semantic_text,
                )
                semantic_embedding = sem_emb.tolist()
                logger.info(
                    "clip_enrich.encode_semantic_text | len=%d duration_ms=%d",
                    len(semantic_text),
                    int((time.perf_counter() - t0) * 1000),
                )
            except Exception as exc:
                logger.warning(
                    "Pipeline clip_enrich: encode_semantic_text failed: %s", exc,
                )

        # ── Step 11: 存入向量库 ──────────────────────────
        if image_embedding is not None:
            # 策略 A: 优先使用含 VLM 内容字段的完整记录
            try:
                record = ImageRecord(
                    image_id=0,
                    prompt=request.prompt,
                    optimized_prompt=response.final_prompt,
                    score=response.final_score,
                    image_path=response.final_image_path,
                    embedding=image_embedding.tolist(),
                    text_embedding=text_embedding.tolist(),
                    # 分区路由
                    category=effective_category or "",
                    subject=effective_category or request.subject or "",
                    # VLM 内容字段
                    tags=vlm_tags,
                    semantic_text=semantic_text,
                    semantic_embedding=semantic_embedding,
                    main_objects=vlm_main_objects,
                    scene_description=vlm_scene_description,
                    style=vlm_style,
                    color_palette=vlm_color_palette,
                    content_type=vlm_content_type,
                    diagram_type=vlm_content_type,
                    visual_elements=vlm_main_objects,
                    keywords=vlm_tags,
                    knowledge_points=vlm_tags[:6],
                    topic=vlm_scene_description[:100] if vlm_scene_description else "",
                    source_type="generated",
                )
                db_record_id = await asyncio.to_thread(
                    self._vector_store.insert, record,
                    subject=effective_category or request.subject,
                )
                logger.info(
                    "clip_enrich.db_insert | record_id=%d score=%.3f "
                    "category=%s partition=%s tags=%d semantic_text_len=%d",
                    db_record_id, response.final_score,
                    effective_category,
                    self._vector_store._resolve_partition_name(effective_category)
                    if effective_category else "_default",
                    len(vlm_tags), len(semantic_text),
                )
            except Exception as exc:
                logger.error(
                    "Pipeline clip_enrich: full insert failed (%s), "
                    "retrying with minimal fields", exc,
                )
                # 策略 B: 回退到最小字段插入（兼容旧行为，确保数据不丢失）
                try:
                    fallback_record = ImageRecord(
                        image_id=0,
                        prompt=request.prompt,
                        optimized_prompt=response.final_prompt,
                        score=response.final_score,
                        image_path=response.final_image_path,
                        embedding=image_embedding.tolist(),
                        text_embedding=text_embedding.tolist(),
                        subject=request.subject,
                        source_type="generated",
                    )
                    db_record_id = await asyncio.to_thread(
                        self._vector_store.insert, fallback_record,
                        subject=request.subject,
                    )
                    logger.info(
                        "clip_enrich.db_insert_fallback | record_id=%d score=%.3f",
                        db_record_id, response.final_score,
                    )
                except Exception as fallback_exc:
                    logger.error(
                        "Pipeline clip_enrich: fallback insert also failed: %s",
                        fallback_exc,
                    )

        # ── 返回增强响应 ────────────────────────────────
        final_response = PipelineResponse(
            final_image_path=response.final_image_path,
            final_prompt=response.final_prompt,
            final_score=response.final_score,
            total_iterations=response.total_iterations,
            history=response.history,
            stopped_reason=response.stopped_reason,
            db_record_id=db_record_id,
            matched_prompts=matched_prompts,
            stored_in_milvus=(db_record_id is not None),
        )
        # 无论 Milvus 是否成功，都存入本地 RecordStore 作为兜底
        final_response.stored_in_records = self._save_to_record_store(
            request, final_response, request.model,
        )
        return final_response

    # ── Mode 3 辅助方法 ──────────────────────────────────

    def _lazy_init_clip(self) -> None:
        """懒加载 LocalCLIPClient.

        v4: 使用 CachedCLIPClient 包装，避免重复编码浪费算力.
        """
        if self._clip_client is not None:
            return

        from src.evaluate.cached_clip_client import CachedCLIPClient
        from src.evaluate.local_clip_client import LocalCLIPClient

        logger.info(
            "Pipeline lazy-init CLIP | model=%s device=%s fp16=%s",
            CLIP_MODEL_NAME, CLIP_DEVICE, CLIP_USE_FP16,
        )
        base_client = LocalCLIPClient(
            model_name=CLIP_MODEL_NAME,
            device=CLIP_DEVICE,
            use_fp16=CLIP_USE_FP16,
        )
        self._clip_client = CachedCLIPClient(base_client, cache_size=1024)

    def _lazy_init_vector_store(self) -> None:
        """懒加载 VectorStore."""
        if self._vector_store is not None:
            return

        from src.milvus import get_vector_store

        logger.info("Pipeline lazy-init VectorStore")
        self._vector_store = get_vector_store()
        self._vector_store.connect()
        logger.info(
            "VectorStore connected | backend=%s count=%d",
            self._vector_store.backend_type,
            self._vector_store.count(),
        )

    @staticmethod
    def _detect_category_from_prompt(prompt: str) -> str | None:
        """从 prompt 文本自动推断分类标签（v6 自动分区限定）.

        使用关键词加权匹配：每个匹配词按其长度平方计分（长词更特异，
        避免短词误匹配），得分最高的分类即为检测结果.

        关键词表基于 config.CATEGORIES 动态构建，确保检测结果与分区体系一致.

        Args:
            prompt: 用户原始 prompt 文本.

        Returns:
            分类名 (如 "自然风光")，无法判断时返回 None.
        """
        if not prompt or not prompt.strip():
            return None

        text = prompt.strip()

        # 动态构建关键词表：以 config.CATEGORIES 为基础，
        # 合并 _CATEGORY_KEYWORD_SYNONYMS 中的扩展关键词
        from src.config import CATEGORIES
        scores: dict[str, float] = {}

        for cat in CATEGORIES:
            score = 0.0
            # 分类名本身作为关键词（优先匹配，计分权重高）
            if cat in text:
                score += len(cat) ** 2 * 2  # 分类名本身匹配加权×2
            # 扩展同义词
            synonyms = _CATEGORY_KEYWORD_SYNONYMS.get(cat, [])
            for kw in synonyms:
                if kw in text:
                    score += len(kw) ** 2
            if score > 0:
                scores[cat] = score

        # 向后兼容：旧分类名关键词也能命中 → 映射到新分类名
        if not scores:
            legacy_scores: dict[str, float] = {}
            for legacy_name, new_name in _LEGACY_CATEGORY_ALIASES.items():
                if legacy_name in text:
                    legacy_scores[new_name] = len(legacy_name) ** 2 * 3
            if legacy_scores:
                scores = legacy_scores

        if not scores:
            return None

        # 得分最高的分类
        best = max(scores, key=lambda k: scores[k])
        logger.info(
            "_detect_category_from_prompt | prompt_len=%d detected=%s "
            "top_scores=%s",
            len(prompt), best,
            {k: round(v, 1) for k, v in
             sorted(scores.items(), key=lambda x: -x[1])[:3]},
        )
        return best

    # 向后兼容别名
    _detect_subject_from_prompt = _detect_category_from_prompt
