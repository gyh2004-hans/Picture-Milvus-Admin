"""Pydantic 数据模型 —— 系统内所有模块的输入/输出结构."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from src.config import detect_default_model


# ═══════════════════════════════════════════════════════════
# 学科枚举（Milvus 分区路由） — v6: 已由动态分类体系替代，保留向后兼容
# ═══════════════════════════════════════════════════════════

class Subject(str, Enum):
    """[DEPRECATED v6] 学科标签 — 已由 config.CATEGORIES 动态分类体系替代.

    保留此枚举仅为向后兼容旧 Milvus 数据，新代码应使用 config.CATEGORIES.
    """
    CHINESE = "chinese"       # 语文
    MATH = "math"             # 数学
    ENGLISH = "english"       # 英语
    PHYSICS = "physics"       # 物理
    CHEMISTRY = "chemistry"   # 化学
    BIOLOGY = "biology"       # 生物
    HISTORY = "history"       # 历史
    GEOGRAPHY = "geography"   # 地理
    POLITICS = "politics"     # 政治

    @classmethod
    def _missing_(cls, value):
        """大小写不敏感 + 中英文别名匹配."""
        if isinstance(value, str):
            v = value.strip().lower()
            for member in cls:
                if member.value == v:
                    return member
            _cn_map = {
                "语文": cls.CHINESE, "数学": cls.MATH, "英语": cls.ENGLISH,
                "物理": cls.PHYSICS, "化学": cls.CHEMISTRY, "生物": cls.BIOLOGY,
                "历史": cls.HISTORY, "地理": cls.GEOGRAPHY, "政治": cls.POLITICS,
            }
            if v in _cn_map:
                return _cn_map[v]
        return None


# ═══════════════════════════════════════════════════════════
# Draw 模块
# ═══════════════════════════════════════════════════════════

class DrawRequest(BaseModel):
    """生图请求."""
    prompt: str = Field(..., description="自然语言图像描述")
    model: str = Field(default_factory=detect_default_model, description="模型: doubao | tongyi")


class DrawResponse(BaseModel):
    """生图响应."""
    record_id: str = Field(..., description="记录 ID，用于提交反馈")
    image_path: str = Field(..., description="生成图片的本地路径")
    model: str = Field(..., description="使用的模型")
    prompt: str = Field(..., description="原始 prompt")


class FeedbackInput(BaseModel):
    """人工反馈."""
    rating: Optional[int] = Field(default=None, ge=1, le=5, description="1-5 分")
    comment: str = Field(default="", description="文字反馈")
    tags: list[str] = Field(default_factory=list, description="标签，如 accurate / blurry")


class FeedbackRequest(BaseModel):
    """提交反馈请求."""
    record_id: str
    feedback: FeedbackInput


class DrawRecord(BaseModel):
    """阶段 1 生图记录."""
    id: str
    prompt: str
    model: str
    image_path: str
    created_at: str
    metadata: dict = Field(default_factory=dict)
    feedback: Optional[FeedbackInput] = None
    feedback_submitted_at: Optional[str] = None


# ═══════════════════════════════════════════════════════════
# Evaluate 模块（§3.2 VLM 五维度主评测器）
# ═══════════════════════════════════════════════════════════

class DimensionScore(BaseModel):
    """单个维度的得分 + 评语."""
    dimension: str = Field(..., description="维度名称")
    score: float = Field(..., ge=0.0, le=1.0, description="维度得分 0-1")
    comment: str = Field(default="", description="维度评语")


class EvalResult(BaseModel):
    """VLM 五维度评测完整输出 —— 对齐项目计划书 §3.2.

    五个评测维度:
      - 主体对象一致性: 图像是否包含描述中的关键主体
      - 属性一致性: 颜色、大小、形状、数量等属性是否正确
      - 空间关系一致性: 对象位置关系是否满足描述
      - 场景完整性: 背景与环境是否符合要求
      - 整体语义匹配度: 图像与文本之间的综合相关程度
    """
    overall_score: float = Field(..., ge=0.0, le=1.0, description="综合评分 0-1")
    dimension_scores: list[DimensionScore] = Field(
        default_factory=list, description="五维度得分列表"
    )
    issues: list[str] = Field(
        default_factory=list, description="发现的问题列表"
    )
    missing_elements: list[str] = Field(
        default_factory=list, description="缺失的视觉元素"
    )
    suggestions: list[str] = Field(
        default_factory=list, description="优化建议列表"
    )


class AttributeScore(BaseModel):
    """单个属性的评分（供 VLM 复核使用）."""
    attribute: str
    score: float


class EvalRequest(BaseModel):
    """评测请求."""
    prompt: str = Field(..., description="原始 prompt")
    image_path: str = Field(..., description="待评测图片路径")


# ═══════════════════════════════════════════════════════════
# Prompt Refiner 模块（§3.3 策略分析 + §3.4 LLM 调整）
# ═══════════════════════════════════════════════════════════

class StrategyItem(BaseModel):
    """单条优化策略."""
    category: str = Field(..., description="问题分类: missing/attribute_error/composition/style")
    target: str = Field(..., description="目标元素或属性")
    action: str = Field(..., description="优化动作: 增强主体/强化约束/位置描述/风格限制词")


class StrategyAnalysis(BaseModel):
    """策略分析结果 —— PromptRefiner.analyze() 的输出."""
    strategies: list[StrategyItem] = Field(default_factory=list, description="优化策略列表")
    summary: str = Field(default="", description="策略分析总结")


class RefinerInput(BaseModel):
    """Refiner 的输入 —— 原始 prompt + 评测结果."""
    origin_prompt: str
    overall_score: float
    issues: list[str] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)


class RefinerOutput(BaseModel):
    """LLM Adjuster 的输出 —— 优化后的 prompt."""
    optimized_prompt: str = Field(..., description="优化后的 prompt")
    changes_summary: str = Field(default="", description="LLM 对修改的说明")


# ═══════════════════════════════════════════════════════════
# 向量存储 (Milvus)
# ═══════════════════════════════════════════════════════════

class ImageContentParseResult(BaseModel):
    """VLM 图片内容解析结果（v6: 通用版，替代 EducationParseResult）.

    将图片送入视觉 LLM，解析为通用内容结构 JSON，用于构建 semantic_text 并存入 Milvus.
    """
    category: str = Field(default="", description="图片分类（开放域，如风景/人物/科技/美食等）")
    content_type: str = Field(default="", description="图片类型: 照片/插图/图表/截图/海报/其他")
    main_objects: list[str] = Field(default_factory=list, description="主体对象列表")
    scene_description: str = Field(default="", description="场景描述")
    style: str = Field(default="", description="风格描述（如写实/卡通/极简/复古等）")
    color_palette: list[str] = Field(default_factory=list, description="主色调列表")
    tags: list[str] = Field(default_factory=list, description="通用关键词/标签")
    retrieval_prompt: str = Field(default="", description="未来用户可能输入的搜索描述")


# 向后兼容别名
EducationParseResult = ImageContentParseResult


class ImageRecord(BaseModel):
    """存入 Milvus 的图像记录."""
    image_id: int
    prompt: str
    optimized_prompt: Optional[str] = None
    score: float = Field(default=0.0, description="评测得分（对齐项目计划书 §5.2）")
    image_path: str
    embedding: Optional[list[float]] = None
    text_embedding: Optional[list[float]] = Field(
        default=None, description="原始 prompt 的 CLIP 文本向量（clip_enrich 复用检索用）"
    )
    similarity: Optional[float] = Field(
        default=None, description="检索时与本记录的 CLIP 余弦相似度，仅检索响应填充，入库时为空"
    )
    # ── 分类/分区字段（v4 新增, v6 泛化） ──
    subject: Optional[str] = Field(
        default=None, description="[DEPRECATED v6] 学科标签，已由 category 替代，保留向后兼容"
    )
    category: Optional[str] = Field(
        default=None, description="图片分类（开放域，如风景/人物/科技/美食等），用于分区路由和筛选"
    )
    tags: list[str] = Field(
        default_factory=list, description="自由标签/关键词，用于筛选和加权排序"
    )
    # ── 语义入库字段（v5 新增, v6 泛化） ──
    semantic_text: Optional[str] = Field(
        default=None, description="语义文本（结构化拼接，供 Chinese-CLIP 编码）"
    )
    semantic_embedding: Optional[list[float]] = Field(
        default=None, description="语义向量（Chinese-CLIP encode semantic_text）"
    )
    topic: Optional[str] = Field(
        default=None, description="主题（v5 原主知识点，v6 泛化为通用主题字段）"
    )
    knowledge_points: list[str] = Field(
        default_factory=list, description="[DEPRECATED v6] 教材知识点，已由 keywords 替代。向后兼容，新代码使用 tags"
    )
    content_type: Optional[str] = Field(
        default=None, description="图片类型: 照片/插图/图表/截图/海报/其他"
    )
    diagram_type: Optional[str] = Field(
        default=None, description="[DEPRECATED v6] 图类型，已由 content_type 替代"
    )
    grade_level: Optional[str] = Field(
        default=None, description="[DEPRECATED v6] 学段信息（不再使用）"
    )
    visual_elements: list[str] = Field(
        default_factory=list, description="核心视觉元素"
    )
    # v6 新增通用字段
    main_objects: list[str] = Field(
        default_factory=list, description="主体对象列表（v6 新增）"
    )
    scene_description: Optional[str] = Field(
        default=None, description="场景描述（v6 新增）"
    )
    style: Optional[str] = Field(
        default=None, description="风格描述（v6 新增）"
    )
    color_palette: list[str] = Field(
        default_factory=list, description="主色调列表（v6 新增）"
    )
    keywords: list[str] = Field(
        default_factory=list, description="通用关键词（v6 新增，替代 knowledge_points）"
    )
    source_type: str = Field(
        default="generated", description="素材来源: generated(AI生成) / uploaded(人工上传)"
    )

    @field_validator("subject", mode="before")
    @classmethod
    def _normalize_subject(cls, v):
        """中文/英文 → 英文值，未识别 → 保持原值或 None."""
        if v is None or v == "":
            return None
        try:
            return Subject(v).value
        except ValueError:
            return v if isinstance(v, str) else None


class SearchRequest(BaseModel):
    """相似图检索请求."""
    prompt: Optional[str] = Field(default=None, description="查询 prompt（文字检索）")
    image_path: Optional[str] = Field(default=None, description="查询图片路径（以图搜图）")
    top_k: int = Field(default=5, ge=1, le=50)
    subject: Optional[str] = Field(
        default=None, description="[DEPRECATED v6] 限定学科分区检索。新代码使用 category"
    )
    category: Optional[str] = Field(
        default=None, description="限定分类分区检索（v6 新增，不指定则全库检索）"
    )


class SearchResponse(BaseModel):
    """检索结果."""
    results: list[ImageRecord]
    query_time_ms: float
    total_in_partition: int = Field(default=0, description="检索分区内的总记录数")


# ═══════════════════════════════════════════════════════════
# 图片素材上传（v5 新增, v6 泛化）
# ═══════════════════════════════════════════════════════════

class MaterialUploadResponse(BaseModel):
    """图片素材上传响应."""
    record_id: int = Field(..., description="Milvus 入库记录 ID")
    image_path: str = Field(..., description="服务器保存的图片路径")
    parse_result: ImageContentParseResult = Field(..., description="VLM 图片内容解析结果")
    semantic_text: str = Field(..., description="构建的 semantic_text")


class SemanticSearchRequest(BaseModel):
    """语义检索请求（v5 新增, v6 泛化）."""
    text: str = Field(..., min_length=1, max_length=2000, description="自然语言搜索描述")
    top_k: int = Field(default=5, ge=1, le=50)
    subject: Optional[str] = Field(
        default=None, description="[DEPRECATED v6] 限定学科。新代码使用 category 参数"
    )
    category: Optional[str] = Field(
        default=None, description="限定分类（v6 新增），如 风景/人物/科技 等"
    )


class SemanticSearchResult(BaseModel):
    """语义检索结果项（含加权得分明细）."""
    image_id: int
    prompt: str
    optimized_prompt: Optional[str] = None
    score: float
    image_path: str
    subject: Optional[str] = None
    category: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    topic: Optional[str] = None
    knowledge_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    content_type: Optional[str] = None
    diagram_type: Optional[str] = None
    grade_level: Optional[str] = None
    main_objects: list[str] = Field(default_factory=list)
    scene_description: Optional[str] = None
    style: Optional[str] = None
    color_palette: list[str] = Field(default_factory=list)
    source_type: str = "generated"
    # 加权得分
    final_score: float = Field(default=0.0, description="最终加权得分")
    semantic_similarity: float = Field(default=0.0, description="语义相似度 (权重0.7)")
    image_similarity: float = Field(default=0.0, description="图像相似度 (权重0.2)")
    tags_overlap: float = Field(default=0.0, description="标签重叠度 (权重0.1)")


class SemanticSearchResponse(BaseModel):
    """语义检索响应."""
    results: list[SemanticSearchResult]
    query_time_ms: float
    total_in_partition: int = Field(default=0)
    query_text: Optional[str] = None
    query_subject: Optional[str] = None
    query_category: Optional[str] = Field(default=None, description="v6: 查询限定分类")


# ═══════════════════════════════════════════════════════════
# 闭环 Pipeline
# ═══════════════════════════════════════════════════════════

class PipelineRequest(BaseModel):
    """闭环请求."""
    prompt: str
    model: str = Field(default_factory=detect_default_model)
    max_iterations: int = Field(default=3, ge=1, le=10, description="最大迭代次数（项目计划书 §4.2: 最多 3 次）")
    eval_threshold: float = Field(
        default=0.82, ge=0.0, le=1.0,
        description="评测阈值（overall_score ≥ 此值提前结束，默认 0.82）",
    )
    # clip_enrich 专用参数
    clip_top_k: int = Field(default=5, ge=1, le=20, description="CLIP 检索返回的相似 prompt 数量")
    clip_min_score: float = Field(default=0.75, ge=0.0, le=1.0, description="CLIP 检索最低评分过滤阈值")
    reuse_threshold: float = Field(
        default=0.77, ge=0.0, le=1.0,
        description="检索命中复用阈值（v5: 语义检索 semantic_similarity ≥ 此值时直接复用已有图片，跳过生图）",
    )
    subject: Optional[str] = Field(
        default=None, description="[DEPRECATED v6] 学科标签。新代码使用 category"
    )
    category: Optional[str] = Field(
        default=None, description="分类标签（v6 新增，用于分区路由，推荐指定）"
    )


class AsyncPipelineResponse(BaseModel):
    """异步生图提交响应 —— 提交即返回，图在后台生成.

    image_url 为预分配占位地址，此刻文件可能还不存在（生成中 → 404），
    后台任务完成后该 URL 即可拉到图.
    """
    task_id: str = Field(..., description="后台任务 ID（uuid4 hex）")
    image_url: str = Field(..., description="预分配的图片访问 URL（生成中暂 404）")


class PipelineIteration(BaseModel):
    """单轮迭代记录."""
    iteration: int
    prompt: str
    image_path: str
    overall_score: float
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)
    missing_elements: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    # Refiner 输出（最后一轮为 None，因为没有下一轮）
    optimized_prompt: Optional[str] = None
    changes_summary: Optional[str] = None


class PipelineResponse(BaseModel):
    """闭环完整结果."""
    final_image_path: str
    final_image_base64: Optional[str] = Field(
        default=None,
        description="最终图片的 base64 编码（PNG），供前端直接展示，不依赖服务器本地路径",
    )
    final_prompt: str
    final_score: float
    total_iterations: int
    history: list[PipelineIteration]
    stopped_reason: str  # "threshold_met" | "converged" | "max_iterations" | "score_regression" | "reused"
    # clip_enrich 专用字段
    db_record_id: Optional[int] = Field(default=None, description="向量库入库记录 ID")
    matched_prompts: list[dict] = Field(default_factory=list, description="CLIP 检索到的相似 prompt 列表")
    reused_from_record_id: Optional[int] = Field(
        default=None, description="命中复用时，来源记录的 image_id（跳过生图，直接复用已有图片）"
    )
    # 数据库存储状态（v7: 告知前端图片 + prompt 是否已可靠入库）
    stored_in_milvus: bool = Field(default=False, description="是否已存入 Milvus 向量库")
    stored_in_records: bool = Field(default=False, description="是否已存入本地 JSON 记录（RecordStore）")
