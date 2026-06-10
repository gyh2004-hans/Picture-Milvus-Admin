# Milvus 素材图片入库优化方案（精简版）

## 一、目标

新增：

前端直接上传各类图片素材 → 自动理解图片内容 → 存入 Milvus → 后续与历史 AI 图片一起参与 prompt 检索。

注意：

本功能不是"以图搜图"，而是"图片知识入库"。

核心目标：

让上传的图片未来能被自然语言精准召回。

例如：

上传：

日落海滩风景照片

未来用户输入：

"暖色调的日落海滩风景照片"

能够召回该图片。

## 二、核心方案

采用：

VLM + Chinese-CLIP + Milvus

流程：

```
上传图片
    ↓
VLM 内容语义解析
    ↓
生成标准 semantic_text
    ↓
Chinese-CLIP 向量化
    ↓
Milvus 入库
    ↓
未来 prompt 检索参与召回
```

原则：

VLM 负责理解图片内容

Chinese-CLIP 负责向量化

Milvus 负责统一检索

## 三、数据库修改

保留现有 schema，仅新增字段：

| 字段 | 作用 |
|------|------|
| semantic_text | 标准内容语义 |
| semantic_embedding | 主检索向量 |
| topic | 主主题 |
| keywords | 内容关键词 |
| scene_type | 场景类型 |
| visual_elements | 核心图元素 |
| source_type | generated / uploaded |

说明：

semantic_embedding 作为未来主检索向量。
vector(image embedding) 保留，仅辅助排序。
旧数据无需迁移。

## 四、VLM 内容解析（核心）

新增：

image_content_parser.py

作用：

图片 → 结构化内容描述。

固定输出 JSON：

```json
{
  "category": "",
  "topic": "",
  "scene_type": "",
  "keywords": [],
  "visual_elements": [],
  "retrieval_prompt": ""
}
```

要求：

category 使用动态分类体系（如风景/人物/动物/科技/美食/建筑/艺术等）。
keywords 必须是图片中可识别的核心内容关键词。
retrieval_prompt 必须是未来用户可能输入的搜索描述。

示例：

```json
{
  "category": "风景",
  "topic": "日落海滩",
  "keywords": [
    "日落",
    "海滩",
    "暖色调"
  ],
  "scene_type": "照片",
  "retrieval_prompt":
  "暖色调的日落海滩风景照片，金色阳光洒在海面上"
}
```

## 五、Semantic Text 构建

不要直接 embedding 图片描述。

统一构造：

```
分类: 风景
主题: 日落海滩
关键词: 日落、海滩、暖色调
场景类型: 照片
检索描述:
暖色调的日落海滩风景照片，金色阳光洒在海面上
```

然后：

```
semantic_embedding =
clip.encode_text(semantic_text)
```

作为：

未来主检索向量。

## 六、上传入库流程

新增接口：

POST /api/upload_material

流程：

```
上传图片
    ↓
VLM解析
    ↓
生成 semantic_text
    ↓
Chinese-CLIP:
    image embedding
    semantic embedding
    ↓
Milvus insert
```

写入：

```json
{
    "vector": "image_embedding",
    "semantic_embedding": "semantic_embedding",
    "prompt": "retrieval_prompt",
    "optimized_prompt": "retrieval_prompt",
    "semantic_text": "semantic_text",
    "source_type": "uploaded"
}
```

说明：

上传图片无原始 prompt。

因此：

retrieval_prompt 直接作为 prompt 存储。

## 七、检索逻辑修改

当前：

text_embedding search

改为：

semantic_embedding search

流程：

```
用户 prompt
    ↓
Chinese-CLIP encode_text
    ↓
Milvus 检索
    ↓
召回：
AI生成图 + 上传素材图
```

排序：

```
final_score =
0.7 * semantic_similarity
+ 0.2 * image_similarity
+ 0.1 * tags_overlap
```

原则：

语义优先，视觉相似辅助。

### 7.2 Pipeline 检索复用（v5.1 新增）

CLIP_ENRICH 模式下，Pipeline 增加了检索复用优先逻辑：

```
用户自然语言
    ↓
Chinese-CLIP encode_text → text_embedding
    ↓
① prompt 精确匹配（同 prompt 重跑，score ≥ clip_min_score）
    ↓ 未命中
② search_by_semantic 加权检索（0.7×semantic + 0.2×image + 0.1×tags）
    ↓
    ├─ semantic_similarity ≥ reuse_threshold (默认 0.77) 且文件存在
    │     → 直接复用已有图片，stopped_reason="reused"
    │       total_iterations=0，跳过全部生图/评测
    │
    └─ 未命中 → 原始 prompt → Draw → Evaluate → Refine
          → CLIP 编码新图 → Milvus 入库
```

参数（`PipelineRequest` 请求体字段，见 `src/models/schemas.py`）：

- `reuse_threshold`：默认 **0.77**，语义近似复用阈值（`semantic_similarity`）
- `clip_min_score`：默认 **0.75**，精确匹配与检索结果的最低 VLM 评测分过滤
- 复用需同时满足：相似度/匹配达标 + 图片文件存在

与 `/api/search/semantic` 的区别：

- `/api/search/semantic`：纯检索接口，仅返回库中已有图片列表
- `/pipeline` (clip_enrich)：检索 → 命中复用 / 未命中生图 → 入库，是完整的生产接口

## 八、预期效果

系统从：

历史 Prompt 复用库

升级为：

图片素材知识库

实现：

- 各类图片直接入库
- 自然语言精准召回
- AI生成图与上传图统一检索
- 素材越多，检索越准
