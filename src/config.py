"""应用配置 —— 环境变量 + 默认值.

配置项对齐项目计划书各模块参数:
  - §3.1 Draw 生图模块: DOUBAO_*/TONGYI_* 系列
  - §3.2 Evaluate 评测模块: VLM_EVAL_MODEL / EVAL_THRESHOLD
  - §3.3 Prompt Refiner 策略分析: LLM_* 系列
  - §3.4 LLM 调整模块: LLM_* 系列
  - §4.2 终止规则: MAX_ITERATIONS / CONVERGENCE_DELTA
  - §5.2 Milvus: MILVUS_HOST / MILVUS_PORT
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# picture/ 目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 仓库根目录（与主项目 .env 共享密钥）
REPO_ROOT = PROJECT_ROOT.parent

# 先加载仓库根 .env，再加载 picture/.env（后者可覆盖）
load_dotenv(REPO_ROOT / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# 存储路径
STORAGE_DIR = PROJECT_ROOT / "storage"
IMAGE_DIR = STORAGE_DIR / "images"
RECORDS_FILE = STORAGE_DIR / "records.json"

# ── 豆包 / 火山方舟 ──
DOUBAO_API_KEY = os.getenv("DOUBAO_API_KEY") or os.getenv("ARK_API_KEY", "")
DOUBAO_BASE_URL = os.getenv(
    "DOUBAO_BASE_URL",
    os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
)
DOUBAO_IMAGE_MODEL = os.getenv("DOUBAO_IMAGE_MODEL", "doubao-seedream-4-0-250828")
DOUBAO_IMAGE_SIZE = os.getenv("DOUBAO_IMAGE_SIZE", "512x512")

# ── 通义千问 / 百炼 ──
TONGYI_API_KEY = os.getenv("TONGYI_API_KEY") or os.getenv("DASHSCOPE_API_KEY", "")
TONGYI_BASE_URL = os.getenv(
    "TONGYI_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 通义万相默认模型（与项目计划书一致，使用千问生图大模型）
TONGYI_IMAGE_MODEL = os.getenv("TONGYI_IMAGE_MODEL", "qwen-image-2.0-pro")
TONGYI_IMAGE_SIZE = os.getenv("TONGYI_IMAGE_SIZE", "512*512")

# ── LLM (Prompt Refiner + LLM Adjuster) ──
LLM_API_KEY = os.getenv("LLM_API_KEY", os.getenv("ARK_API_KEY", ""))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
LLM_MODEL = os.getenv("LLM_MODEL", os.getenv("ARK_MODEL", "doubao-seed-2-0-pro-260215"))

# ── Milvus ──
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_URI = os.getenv(
    "MILVUS_URI",
    f"http://{MILVUS_HOST}:{MILVUS_PORT}",
)

# ── 服务 ──
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))
# 图床对外可访问 URL 基址（不含尾斜杠）
PICTURE2_PUBLIC_BASE_URL = os.getenv(
    "PICTURE_ADMIN_PUBLIC_BASE_URL",
    os.getenv("PICTURE2_PUBLIC_BASE_URL", "http://localhost:8001"),
).rstrip("/")

# ── CLIP 模型（Chinese-CLIP ViT-Base Patch16，仅供 Milvus 向量检索使用） ──
# OFA-Sys/chinese-clip-vit-base-patch16: 512-dim, 224×224, 中英双语
# 首次使用自动从 HuggingFace 下载 (~0.6GB)，可通过 HF_ENDPOINT 设置镜像
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "OFA-Sys/chinese-clip-vit-base-patch16")
CLIP_MODEL_PRETRAINED = os.getenv("CLIP_MODEL_PRETRAINED", "")

# 向量维度与 patch 数由 CLIP_MODEL_NAME 推导，不单独暴露配置（避免与模型不一致）
#   (向量维度, patch 数);  patch 数 = (224 / patch_size)^2
_CLIP_MODEL_SPECS = {
    "OFA-Sys/chinese-clip-vit-base-patch16": (512, 196),         # 224/16=14 → 196
    "OFA-Sys/chinese-clip-vit-large-patch14": (768, 256),        # 224/14=16 → 256
    "OFA-Sys/chinese-clip-vit-large-patch14-336px": (768, 576),  # 336/14=24 → 576
    "OFA-Sys/chinese-clip-vit-huge-patch14": (1024, 256),        # 224/14=16 → 256
}
EMBEDDING_DIM, CLIP_NUM_PATCHES = _CLIP_MODEL_SPECS.get(
    CLIP_MODEL_NAME, (512, 196),  # 未知模型回退到 base-patch16 规格
)


def _detect_clip_device() -> str:
    """自动检测最优 CLIP 推理设备.

    RTX 3060 6GB / 其他 CUDA GPU → "cuda"，否则 "cpu".
    环境变量 CLIP_DEVICE 可覆盖自动检测结果.
    """
    _env = os.getenv("CLIP_DEVICE", "").strip()
    if _env:
        return _env

    try:
        import torch

        if torch.cuda.is_available():
            _name = torch.cuda.get_device_name(0) or "Unknown GPU"
            _mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            import logging

            _log = logging.getLogger(__name__)
            _log.info(
                "CLIP device auto-detected: cuda | gpu=%s vram=%.1fGB",
                _name,
                _mem_gb,
            )
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _detect_clip_fp16() -> bool:
    """检测是否应启用 FP16.

    CUDA 环境默认启用 FP16（节省显存 + 加速），CPU 强制关闭.
    环境变量 CLIP_USE_FP16 可覆盖.
    """
    _env = os.getenv("CLIP_USE_FP16", "").strip().lower()
    if _env:
        return _env == "true"
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


CLIP_DEVICE = _detect_clip_device()
CLIP_USE_FP16 = _detect_clip_fp16()

# ── VLM 评测（项目计划书 §3.2: VLM 作为主评测器，5 维度评分） ──
VLM_EVAL_MODEL = os.getenv("VLM_EVAL_MODEL", "doubao-seed-1-6-vision-250815")
EVAL_THRESHOLD = float(os.getenv("EVAL_THRESHOLD", "0.82"))

# ── 终止规则（项目计划书 §4.2） ──
MAX_ITERATIONS = 3
CONVERGENCE_DELTA = float(os.getenv("CONVERGENCE_DELTA", "0.01"))
# 分数相对历史最佳下降超过此值时回滚到 best_prompt 并停止
SCORE_ROLLBACK_DELTA = float(os.getenv("SCORE_ROLLBACK_DELTA", "0.05"))
# 每轮 LLM 最多处理的 severe issue 数
MAX_ISSUES_PER_ROUND = int(os.getenv("MAX_ISSUES_PER_ROUND", "3"))

# ── VLM 复核（豆包视觉） ──
VLM_VERIFY_ENABLED = os.getenv("VLM_VERIFY_ENABLED", "true").lower() == "true"
VLM_VISION_MODEL = os.getenv("VLM_VISION_MODEL", "doubao-seed-1-6-vision-250815")
VLM_VERIFY_LANG = os.getenv("VLM_VERIFY_LANG", "zh")
VLM_VERIFY_MAX_ATTRS = int(os.getenv("VLM_VERIFY_MAX_ATTRS", "6"))
# VLM promoted 属性使用的占位分
VLM_PROMOTED_SCORE = float(os.getenv("VLM_PROMOTED_SCORE", "0.45"))


# ── 模型自动检测 ──


def detect_default_model() -> str:
    """自动检测已配置的生图模型.

    根据 .env 中配置的 API Key 自动选择可用模型.
    优先级: tongyi > doubao（与项目计划书 §3.1 一致，千问为默认生图模型）
    都未配置时返回 "tongyi"（让调用方报清晰的缺少密钥错误）.
    """
    if os.getenv("TONGYI_API_KEY"):
        return "tongyi"
    if os.getenv("DOUBAO_API_KEY"):
        return "doubao"
    return "tongyi"


# ── 动态分类体系（v6: 替代硬编码 9 学科分区） ──

# 默认分类列表（用户可在 .env 中通过 CATEGORIES 覆盖，逗号分隔）
_DEFAULT_CATEGORIES = [
    "自然风光", "人物人像", "城市建筑", "美食饮品", "动植物",
    "办公商务", "数码科技", "服饰穿搭", "家居家装", "节日庆典",
    "手绘插画", "纹理背景", "交通出行", "教育培训", "运动休闲",
]

# 默认分类 → ASCII 分区名映射（Milvus Standalone 要求分区名以字母/下划线开头）
_DEFAULT_CATEGORY_PARTITION_MAP: dict[str, str] = {
    "自然风光": "ziranfengguang",
    "人物人像": "renwurenxiang",
    "城市建筑": "chengshijianzhu",
    "美食饮品": "meishiyinpin",
    "动植物": "dongzhiwu",
    "办公商务": "bangongshangwu",
    "数码科技": "shumakeji",
    "服饰穿搭": "fushichuanda",
    "家居家装": "jiajujiazhuang",
    "节日庆典": "jieriqingdian",
    "手绘插画": "shouhuichahua",
    "纹理背景": "wenlibeijing",
    "交通出行": "jiaotongchuxing",
    "教育培训": "jiaoyupeixun",
    "运动休闲": "yundongxiuxian",
}


def _load_categories() -> list[str]:
    """从环境变量加载动态分类列表.

    .env 中 CATEGORIES 用逗号分隔:
        CATEGORIES=风景,人物,动物,科技,美食,建筑,艺术,其他

    未设置时使用 DEFAULT_CATEGORIES.
    """
    env_val = os.getenv("CATEGORIES", "").strip()
    if env_val:
        cats = [c.strip() for c in env_val.split(",") if c.strip()]
        if cats:
            return cats
    return list(_DEFAULT_CATEGORIES)


#: 当前生效的分类列表（用于分区创建、API 返回、前端展示）
CATEGORIES: list[str] = _load_categories()

#: 分类 → 分区名映射（供分区路由使用，用户可在 .env 中设置 CATEGORY_PARTITIONS 自定义）
CATEGORY_PARTITION_MAP: dict[str, str] = {}
_env_partitions = os.getenv("CATEGORY_PARTITIONS", "").strip()
if _env_partitions:
    for pair in _env_partitions.split(","):
        pair = pair.strip()
        if ":" in pair:
            cn, py = pair.split(":", 1)
            CATEGORY_PARTITION_MAP[cn.strip()] = py.strip()
# Fallback: 未配置分区名映射时，使用默认 ASCII 映射；若无默认映射则用中文名
for cat in CATEGORIES:
    if cat not in CATEGORY_PARTITION_MAP:
        CATEGORY_PARTITION_MAP[cat] = _DEFAULT_CATEGORY_PARTITION_MAP.get(cat, cat)
