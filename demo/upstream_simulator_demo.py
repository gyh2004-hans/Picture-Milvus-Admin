"""上游模块模拟器 Demo —— 模拟上游模块生成自然语言图片描述.

模拟 picture2 的上游模块（Design / Materialize）输出：
  通用图片描述 → natural language prompt → picture2 生图评测闭环。

本 Demo 为多个分类准备了 prompt 数据，覆盖风景/人物/动物/科技/美食/建筑/艺术等。

用法（在 picture2 目录内执行）:
    # ★ 一键全链路测试：多分类 28 条 prompt → CLIP+Draw+VLM评测+Milvus入库
    python -m demo.upstream_simulator_demo --mode pipeline --category all --with-clip

    # dry-run（零 API 调用，验证流程）
    python -m demo.upstream_simulator_demo --mode dry-run

    # 列出所有 prompt
    python -m demo.upstream_simulator_demo --mode list

    # 列出指定分类的 prompt
    python -m demo.upstream_simulator_demo --mode list --category 风景

    # 单分类 Pipeline
    python -m demo.upstream_simulator_demo --mode pipeline --category 科技 --with-clip

    # 单条 prompt 快速验证
    python -m demo.upstream_simulator_demo --mode pipeline --category 美食 --id food_ramen --with-clip
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.config import EVAL_THRESHOLD, IMAGE_DIR, RECORDS_FILE, detect_default_model
from src.draw import DRAWER_REGISTRY, DoubaoDrawer, TongyiDrawer
from src.models.schemas import (
    PipelineRequest,
    PipelineResponse,
)
from src.pipeline import ImagePipeline
from src.storage import RecordStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# 上游模拟输出 —— 多分类图片自然语言 prompt（v6 泛化）
# ═══════════════════════════════════════════════════════════════════
#
# 以下 prompt 覆盖风景/人物/动物/科技/美食/建筑/艺术等多个通用分类.


@dataclass(frozen=True)
class ImageGenerationPrompt:
    """上游模块输出的一条图片生成描述."""

    id: str
    category: str  # 分类名: scenery/portrait/animal/tech/food/architecture/art
    title: str
    prompt: str  # 自然语言图像描述
    context: str = ""  # 生成场景/用途说明


# ── 语文 ──────────────────────────────────────────────────────────

SCENERY_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="cn_poetry_scene",
        category="chinese",
        title="古诗场景复原——《登高》",
        context="讲解杜甫《登高》：风急天高、渚清沙白、无边落木、不尽长江",
        prompt=(
            "杜甫《登高》诗意场景插画："
            "秋日高台之上，一位老者（杜甫）凭栏远眺，衣袂被劲风吹起；"
            "近景沙洲清冷、白沙如雪，飞鸟盘旋低徊；"
            "远方长江波涛滚滚、无边落叶萧萧而下；"
            "天空高远灰蓝，气氛苍凉悲壮；"
            "中国水墨风格，赭石+青灰+白色调，竖幅构图，"
            "右下角标注诗眼字：'悲'（红色篆书小印），"
            "中学语文古诗词鉴赏教学插图"
        ),
    ),
    ImageGenerationPrompt(
        id="cn_argument_patterns",
        category="chinese",
        title="议论文论证方法对比图",
        context="讲解论证方法：举例论证、道理论证、对比论证、比喻论证、因果论证",
        prompt=(
            "议论文五种论证方法对比图解，五列卡片式布局："
            "举例论证（绿色卡，图标：放大镜+案例文档，示例'如司马迁忍辱著《史记》'），"
            "道理论证（蓝色卡，图标：书本+引号，示例'孟子曰：天将降大任于斯人也'），"
            "对比论证（橙红卡，图标：天平两端，示例'正面vs反面 双栏对比排版'），"
            "比喻论证（紫色卡，图标：灯泡+桥梁，示例'理想是灯塔，照亮前行之路'），"
            "因果论证（青色卡，图标：箭头链A→B→C，示例'勤奋→积累→成功 因果链'），"
            "每列：上=方法名+图标，中=定义框，下=典型例句，最底部'写作建议'小贴士，"
            "信息图+卡片风格，白底，彩色分区，中文标注，中学语文作文指导插图"
        ),
    ),
    ImageGenerationPrompt(
        id="cn_vernacular_movement",
        category="chinese",
        title="新文化运动——白话文vs文言文",
        context="讲解新文化运动：胡适《文学改良刍议》、鲁迅《狂人日记》、白话文推广",
        prompt=(
            "新文化运动白话文推广对比信息图："
            "左侧'文言文'板块（复古卷轴样式，棕色仿古底）："
            "例句'学而时习之，不亦说乎'配古文排版，标注'典雅但脱离大众'，"
            "中间'变革'箭头（红色粗箭头，标注1917-1919）："
            "胡适头像+《文学改良刍议》封面（'八不主义'列表），"
            "陈独秀头像+《新青年》杂志封面，"
            "右侧'白话文'板块（现代书本样式，白底）："
            "鲁迅《狂人日记》首段摘录（'我翻开历史一查……'），标注'第一部白话小说'，"
            "底部对比结论：'我手写我口'——黄遵宪（金色大字），"
            "历史文献+信息图结合风格，米黄仿古底+白色现代区，中文标注，中学语文课插图"
        ),
    ),
]

# ── 数学 ──────────────────────────────────────────────────────────

TECH_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="math_set_operations",
        category="math",
        title="集合运算 Venn 图",
        context="讲解集合运算：并集(A∪B)、交集(A∩B)、补集(∁UA)、差集(A-B)",
        prompt=(
            "高中数学集合运算 Venn 图四联排版："
            "左上：并集 A∪B（两个重叠圆整体蓝色填充，标注'属于A或属于B'），"
            "右上：交集 A∩B（两圆重叠部分红色高亮，标注'同时属于A和B'），"
            "左下：补集 ∁UA（大方框U内，圆A以外区域绿色填充，标注'不属于A'），"
            "右下：差集 A-B（圆A中去掉与B重叠的月牙形，橙色填充，标注'属于A但不属于B'），"
            "每图下方公式+文字标注，配色统一（蓝/红/绿/橙），白底，"
            "教科书插图风格，线条清晰，中文标注，高中数学必修一插图"
        ),
    ),
    ImageGenerationPrompt(
        id="math_trig_wave",
        category="math",
        title="三角函数图像——正弦与余弦曲线",
        context="讲解正弦函数 y=sin x 与余弦函数 y=cos x 的图像与性质",
        prompt=(
            "三角函数图像对比图："
            "坐标系中并排展示两条完整周期曲线（-π 到 2π 区间）："
            "正弦曲线 y=sin x（红色实线）：过原点(0,0)→最高点(π/2,1)→(π,0)→最低点(3π/2,-1)→(2π,0)，"
            "余弦曲线 y=cos x（蓝色虚线）：最高点(0,1)→(π/2,0)→最低点(π,-1)→(3π/2,0)→最高点(2π,1)，"
            "x轴标注：-π, -π/2, 0, π/2, π, 3π/2, 2π（用 π 符号），"
            "y轴标注：-1, -0.5, 0, 0.5, 1，"
            "关键点用小圆点标记+坐标标注框，x轴上方标注y=sin x(红)/y=cos x(蓝)，"
            "右侧性质总结卡片：定义域R/值域[-1,1]/周期2π/奇偶性/最值，"
            "教科书坐标系插图风格，白底+浅灰网格，彩色曲线，中文标注，高中数学必修插图"
        ),
    ),
    ImageGenerationPrompt(
        id="math_derivative_meaning",
        category="math",
        title="导数的几何意义——切线斜率",
        context="讲解导数概念：割线趋近于切线、瞬时变化率、f'(x₀)几何意义",
        prompt=(
            "导数几何意义教学示意图："
            "坐标系中一条光滑曲线 y=f(x)（蓝色），曲线上标注一点P(x₀, f(x₀))（红色大圆点），"
            "过P点有三条线："
            "① 割线1：P与Q₁连线（Q₁在P右侧较远处，橙色虚线，标注'割线 斜率=Δy/Δx'），"
            "② 割线2：P与Q₂连线（Q₂更靠近P，浅橙色虚线，标注'Q→P'），"
            "③ 切线：红色实线，标注'切线 斜率=f\\'(x₀)'，"
            "Q₁→Q₂→P 的动态趋势用三个小箭头表示'趋近'过程，"
            "右侧放大圆：点P附近曲线+切线局部放大，展示'局部以直代曲'，"
            "底部公式框：f\\'(x₀)=lim(Δx→0)[f(x₀+Δx)-f(x₀)]/Δx，标注'极限→切线斜率→导数'，"
            "教科书坐标系插图风格，白底+浅灰网格，高中数学选修插图"
        ),
    ),
    ImageGenerationPrompt(
        id="math_probability_tree",
        category="math",
        title="概率树状图——条件概率与全概率",
        context="讲解条件概率：P(B|A)、全概率公式、贝叶斯公式初步",
        prompt=(
            "高中数学概率树状图教学插图："
            "从根节点'试验开始'（左侧，灰圆）向右分叉为两层："
            "第一层分叉：A（上分支，蓝色，概率P(A)=0.6标注在线旁）和 Ā（下分支，浅蓝，P(Ā)=0.4），"
            "第二层分叉（从A出发）：B|A（绿色，P(B|A)=0.7）和 B̄|A（浅绿，P(B̄|A)=0.3），"
            "第二层分叉（从Ā出发）：B|Ā（橙色，P(B|Ā)=0.2）和 B̄|Ā（浅橙，P(B̄|Ā)=0.8），"
            "每条路径终点（右侧）标注联合概率（路径概率相乘）："
            "A→B: 0.6×0.7=0.42 | A→B̄: 0.6×0.3=0.18 | Ā→B: 0.4×0.2=0.08 | Ā→B̄: 0.4×0.8=0.32，"
            "底部公式区：全概率公式 P(B)=P(A)P(B|A)+P(Ā)P(B|Ā) = 0.42+0.08 = 0.50（红框高亮），"
            "右侧贝叶斯公式提示框：P(A|B)=P(A)P(B|A)/P(B)，"
            "流程图+树状图风格，白底，彩色分支，中文标注，高中数学选择性必修插图"
        ),
    ),
]

# ── 英语 ──────────────────────────────────────────────────────────

PORTRAIT_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="en_tenses_timeline",
        category="english",
        title="英语时态时间轴总览",
        context="讲解英语 8 种核心时态：一般现在/过去/将来，进行时，完成时，完成进行时",
        prompt=(
            "英语时态时间轴全景图："
            "横向时间轴从左到右：Past（左侧蓝色区域）→ Now（中央红色竖线）→ Future（右侧绿色区域），"
            "时间轴上方排列 4 组时态（每组含公式+例句+图标）："
            "一般时态组：一般现在（do/does，日历图标，'I walk to school every day'），"
            "一般过去（did，时钟倒退图标，'I walked to school yesterday'），"
            "一般将来（will do，望远镜图标，'I will walk to school tomorrow'），"
            "进行时态组：现进（am/is/are doing，动态小人图标，'I am walking now'），"
            "过去进行（was/were doing，'I was walking at 8am'），"
            "完成时态组：现完（have/has done，对勾图标，'I have walked 5km'），"
            "过去完成（had done，'I had walked before it rained'），"
            "时间轴下方标注时间状语关键词（always/yesterday/tomorrow/now/since/for...），"
            "每类时态用不同色块区分（黄/蓝/绿/橙），信息图+时间轴风格，白底，中英双语标注，中学英语语法教学插图"
        ),
    ),
    ImageGenerationPrompt(
        id="en_relative_clauses",
        category="english",
        title="定语从句关系词图解",
        context="讲解定语从句：关系代词(who/whom/whose/which/that)与关系副词(when/where/why)",
        prompt=(
            "英语定语从句关系词图解："
            "中央'先行词'（大圆，金色）向外辐射箭头连接各关系词卡片："
            "指人组（蓝色卡片）：who（主语，小人图标+例句'The girl who sings is my sister'），"
            "whom（宾语，小人+箭头图标+例句'The boy whom I met yesterday'），"
            "whose（所属，小人+标签图标+例句'The man whose car is red'），"
            "指物组（绿色卡片）：which（物体图标+例句'The book which I bought'），"
            "that（万能钥匙图标+例句'The book that I bought / The girl that sings'），"
            "关系副词组（橙色卡片）：when（时间=钟表图标+例句'the day when we met'），"
            "where（地点=地图图标+例句'the place where we met'），"
            "why（原因=问号图标+例句'the reason why he left'），"
            "底部总结框：'先行词是人→who/whom/that | 先行词是物→which/that | 时间/地点/原因→when/where/why'，"
            "思维导图风格，白底，彩色卡片，中英双语标注，中学英语语法教学插图"
        ),
    ),
    ImageGenerationPrompt(
        id="en_writing_structure",
        category="english",
        title="英语议论文结构——汉堡模型",
        context="讲解英语议论文五段式结构：Introduction→Body Paragraphs→Conclusion",
        prompt=(
            "英语议论文'汉堡'结构模型图："
            "顶层面包（Introduction，浅棕色矩形）："
            "Hook（钩子：吸引读者注意）+ Background（背景）+ Thesis Statement（中心论点，红框高亮），"
            "标注'1 paragraph | 3-4 sentences'，"
            "中层肉饼×3（Body Paragraphs，肉色矩形×3 层叠）："
            "每层结构统一：Topic Sentence（主题句，蓝框）→ Supporting Details（论据：事实/数据/例子，绿框）→ Concluding Sentence（小结句，橙框），"
            "层间用 Transition Words 箭头连接（Firstly / Moreover / Furthermore / In addition），"
            "底层面包（Conclusion，浅棕色矩形）："
            "Restate Thesis（重述论点）→ Summary of Main Points（要点总结）→ Final Thought（升华/呼应力，金色星标），"
            "标注'1 paragraph | 3-4 sentences'，"
            "整体呈汉堡纵剖面图形，各层内部标注写作要点+常用句型模板（如'It is widely believed that...'/'On the other hand...'），"
            "趣味信息图风格，食物类比，白底，中英双语标注，中学英语写作教学插图"
        ),
    ),
]

# ── 历史 ──────────────────────────────────────────────────────────

ARCHITECTURE_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="hist_feudal_society",
        category="history",
        title="中国封建社会分期图",
        context="讲解中国封建社会五阶段：战国（形成）→秦汉（确立）→隋唐（繁荣）→宋元（发展）→明清（衰落）",
        prompt=(
            "中国封建社会五阶段演进时间轴："
            "横向时间线从左到右，五个阶段依次展开："
            "战国·形成期（公元前475-前221，浅绿色块）：铁器牛耕推广图标+商鞅变法竹简图标+百家争鸣思想图标，"
            "→ 秦汉·确立期（前221-220，深绿色块）：秦始皇统一六国图标+郡县制结构图+汉武帝独尊儒术图标，"
            "→ 隋唐·繁荣期（581-907，金色块）：科举制（进士科试卷图标）+三省六部制+开元盛世（长安城示意），"
            "→ 宋元·发展期（960-1368，蓝色块）：商品经济（交子纸币图标）+理学兴起+行省制度，"
            "→ 明清·衰落期（1368-1840，灰色块）：内阁制+闭关锁国（封闭大门图标）+资本主义萌芽（纺织工场图标），"
            "每个阶段下方标注：政治制度（上栏）、经济特点（中栏）、文化成就（下栏），"
            "时间轴最右端标注'1840鸦片战争→进入半殖民地半封建社会'（红色分界线），"
            "历史时间轴信息图风格，米黄仿古底，彩色分期，中文标注，中学历史复习插图"
        ),
    ),
    ImageGenerationPrompt(
        id="hist_renaissance",
        category="history",
        title="文艺复兴核心人物与成就",
        context="讲解文艺复兴：意大利三杰（达芬奇/米开朗基罗/拉斐尔）+文学三杰（但丁/彼特拉克/薄伽丘）",
        prompt=(
            "欧洲文艺复兴核心人物与成就信息图："
            "中央'文艺复兴Renaissance'标题（花体英文，金色，意大利佛罗伦萨穹顶背景剪影），"
            "左侧'文学三杰'板块（暖黄色底）："
            "但丁（《神曲》地狱篇封面+'中世纪的最后一位诗人，新时代的最初一位诗人'-恩格斯评语），"
            "彼特拉克（十四行诗手稿图标+'人文主义之父'），"
            "薄伽丘（《十日谈》封面+反教会禁欲主义标签），"
            "右侧'美术三杰'板块（暖红色底）："
            "达·芬奇（《蒙娜丽莎》+《最后的晚餐》缩略图+'全才：画家/工程师/科学家'），"
            "米开朗基罗（大卫像+西斯廷天顶画《创世纪》缩略图），"
            "拉斐尔（《雅典学院》缩略图+圣母像标签），"
            "底部'核心思想'条幅：人文主义Humanism（以人为本/反对神权/追求现世幸福），"
            "两侧箭头指向：→宗教改革→启蒙运动→近代科学（影响链），"
            "历史文化信息图风格，仿古羊皮纸底，中文标注+英文专名，中学世界历史插图"
        ),
    ),
    ImageGenerationPrompt(
        id="hist_cold_war",
        category="history",
        title="冷战格局与重大事件时间线",
        context="讲解冷战（1947-1991）：杜鲁门主义、马歇尔计划、北约vs华约、柏林危机、古巴导弹危机",
        prompt=(
            "冷战格局全景时间线图："
            "顶部两侧：美国（左，蓝底，星条旗图标，标注'资本主义阵营/北约'）vs 苏联（右，红底，锤镰图标，标注'社会主义阵营/华约'），"
            "中间'铁幕'纵向虚线（丘吉尔1946演说标注），左侧美国主导事件（蓝色箭头），右侧苏联主导事件（红色箭头）："
            "1947 杜鲁门主义（左，蓝箭头，'遏制共产主义'标签）→ 1947 马歇尔计划（左，蓝箭头，欧洲复兴美元图标），"
            "1948-49 柏林封锁与空运（中，对峙线，飞机图标），"
            "1949 北约成立（左，蓝箭头，NATO标志）⇌ 1955 华约成立（右，红箭头，华约标志），"
            "1950-53 朝鲜战争（左，蓝箭头，38°线地图局部），"
            "1961 柏林墙修建（中，砖墙图标），"
            "1962 古巴导弹危机（对峙高潮图标，美国封锁线+苏联货轮+核弹剪影，标注'人类最近核战争边缘'），"
            "1970s 缓和时期（中间浅紫色带，标注SALT I/赫尔辛基协议），"
            "1979 苏联入侵阿富汗（右，红箭头），"
            "1989 柏林墙倒塌（中，墙体碎裂图标+人群欢呼剪影），→ 1991 苏联解体（右端，苏联国旗下降图标），"
            "底部时间线标注'1947————————1991'，历史时间轴信息图风格，蓝红色调对峙，中英双语标注，中学世界历史插图"
        ),
    ),
]

# ── 地理 ──────────────────────────────────────────────────────────

ANIMAL_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="geo_climate_types",
        category="geography",
        title="世界气候类型分布图",
        context="讲解世界主要气候类型：热带雨林/热带草原/热带沙漠/地中海/温带海洋/温带大陆/亚寒带/极地",
        prompt=(
            "世界气候类型分布教学地图："
            "世界地图底图（各大洲轮廓清晰），11种气候类型用不同颜色填充标注："
            "热带雨林气候（深绿色，赤道附近：亚马孙/刚果/马来群岛），"
            "热带草原气候（浅绿，雨林两侧：非洲/巴西高原），"
            "热带沙漠气候（黄色，回归线附近：撒哈拉/阿拉伯/澳大利亚中西部），"
            "热带季风气候（橙色，南亚/东南亚），"
            "地中海气候（橄榄绿，大陆西岸30°-40°：地中海沿岸/加州/好望角），"
            "亚热带季风/湿润气候（浅蓝，大陆东岸25°-35°：中国东南/美国东南/南美东南），"
            "温带海洋性气候（蓝色，大陆西岸40°-60°：西欧），"
            "温带大陆性气候（浅棕，大陆内部：中亚/北美内陆），"
            "温带季风气候（深橙，亚洲东部），"
            "亚寒带针叶林气候（深绿灰，北纬50°-70°：西伯利亚/加拿大），"
            "极地/高山气候（白色/浅灰，南极洲/格陵兰/青藏高原），"
            "右侧图例（颜色→气候类型→气温曲线+降水柱状迷你图），"
            "教科书地理图册风格，白底，彩色填充，中英文标注，中学地理教学插图"
        ),
    ),
    ImageGenerationPrompt(
        id="geo_atmosphere_circulation",
        category="geography",
        title="三圈环流与气压带风带",
        context="讲解三圈环流（低纬/中纬/高纬）+ 七个气压带 + 六个风带 + 气压带风带季节移动",
        prompt=(
            "地球大气三圈环流示意图："
            "地球剖面图（圆形），赤道→两极标注纬度（0°/30°N/60°N/90°N），"
            "三个垂直环流圈用闭合箭头标注："
            "低纬环流（哈得来环流，0°-30°）：赤道上升→高空向北→30°下沉→近地面向南（红色箭头环），"
            "中纬环流（费雷尔环流，30°-60°）：30°近地面向北→60°上升→高空向南→30°下沉（蓝色箭头环），"
            "高纬环流（极地环流，60°-90°）：60°高空向北→极地下沉→近地面向南→60°上升（紫色箭头环），"
            "近地面气压带标注：赤道低气压带（红色条，0°，'热力原因'），"
            "副热带高气压带（橙色条，30°N/30°S，'动力原因-下沉'），"
            "副极地低气压带（蓝色条，60°N/60°S，'动力原因-爬升'），"
            "极地高气压带（白色条，90°N/90°S，'热力原因'），"
            "近地面风带箭头标注：信风带（东北信风，0°-30°N，橙色箭头），"
            "盛行西风带（西南风，30°-60°N，蓝色箭头），极地东风带（东北风，60°-90°N，紫色箭头），"
            "教科书剖面图风格，白底，彩色环流箭头，中英文标注，中学地理必修插图"
        ),
    ),
    ImageGenerationPrompt(
        id="geo_river_landforms",
        category="geography",
        title="河流地貌——上中下游对比",
        context="讲解河流侵蚀与堆积地貌：上游V型谷/瀑布→中游河曲/阶地→下游三角洲/冲积平原",
        prompt=(
            "河流上中下游地貌对比三段式插图："
            "上游（左侧，深绿山林背景）：陡峭V型谷剖面，河流下蚀强烈，瀑布（水流落差图标），"
            "河床多巨大砾石（圆石图标标注'砾石为主'），标注'流速快/下蚀为主/V形谷'，"
            "中游（中间，浅绿丘陵背景）：河道变宽开始弯曲→曲流（凸岸堆积（浅滩）+凹岸侵蚀（陡岸），"
            "用箭头示意水流侵蚀与堆积方向），河漫滩+阶地剖面标注（'地壳抬升+河流下切→阶地形成'），"
            "标注'流速适中/侧蚀为主/曲流发育'，"
            "下游（右侧，蓝色平原背景）：河道宽阔蜿蜒，大量曲流+牛轭湖形成示意（'截弯取直→废弃河道→牛轭湖'4步演化图），"
            "河口三角洲（扇形沉积，标注顶积层/前积层/底积层剖面），标注'流速慢/堆积为主/三角洲'，"
            "底部剖面图：从上游到下游河床纵剖面线（陡→缓），标注'侵蚀基准面'，"
            "教科书地质插图风格，蓝绿色调，中文标注，中学地理教学插图"
        ),
    ),
]

# ── 政治 ──────────────────────────────────────────────────────────

FOOD_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="pol_value_law",
        category="politics",
        title="价值规律——价格围绕价值波动",
        context="讲解价值规律：商品价值量由社会必要劳动时间决定，价格受供求影响围绕价值波动",
        prompt=(
            "价值规律教学示意图："
            "坐标系中一条水平红线标注'价值'（中心横线，标注'社会必要劳动时间决定'），"
            "价格曲线（黑色波浪线）围绕价值线上下波动："
            "供不应求→价格高于价值（曲线上凸，标注'价格上涨'，需求>供给图标），"
            "供过于求→价格低于价值（曲线下凹，标注'价格下跌'，供给>需求图标），"
            "四个象限标注实例："
            "① 旺季旅游酒店价格↑（春节海南酒店1000→3000，红箭头↑），"
            "② 丰产年农产品价格↓（大蒜丰产3元→0.8元，蓝箭头↓），"
            "③ 新款手机首发价格↑（需求旺盛，红箭头↑），"
            "④ 过季服装打折（清仓处理，蓝箭头↓），"
            "底部结论框：'价格围绕价值上下波动——价值规律的表现形式'（红色楷体大字），"
            "右下角'看不见的手'图标（亚当·斯密引用框），"
            "教科书经济学图表风格，白底+浅灰网格，红蓝黑配色，中文标注，高中政治必修一插图"
        ),
    ),
    ImageGenerationPrompt(
        id="pol_government_functions",
        category="politics",
        title="政府五大职能结构图",
        context="讲解政府职能：经济调节、市场监管、社会管理、公共服务、生态环境保护",
        prompt=(
            "中国政府五大职能结构图："
            "中央圆形'政府职能'（国徽图标，金色圆），向外辐射五条粗箭头到五个扇形区："
            "① 经济调节（左上，蓝色扇形）：图标=财经曲线图，"
            "内容：宏观调控/财政政策/货币政策/产业政策，例'减税降费/降准降息'，"
            "② 市场监管（右上，绿色扇形）：图标=放大镜+盾牌，"
            "内容：维护市场秩序/反垄断/质量监管/价格监管，例'食品安全监管/反不正当竞争'，"
            "③ 社会管理（左下，橙色扇形）：图标=人群+社区，"
            "内容：社会治安/户籍管理/基层治理/应急管理，例'社区网格化管理/110报警服务'，"
            "④ 公共服务（右下，紫色扇形）：图标=医院+学校+道路，"
            "内容：教育/医疗/社保/交通/文化，例'九年义务教育/医保/高铁建设'，"
            "⑤ 生态环境保护（底部，绿色扇形）：图标=绿叶+地球，"
            "内容：污染防治/生态修复/碳达峰碳中和/垃圾分类，例'长江十年禁渔/蓝天保卫战'，"
            "五扇形围绕中心构成完整圆形（360°），每扇形内细分二级职责+典型事例，"
            "思维导图+扇形图风格，白底，五色分区，中文标注，高中政治必修二插图"
        ),
    ),
    ImageGenerationPrompt(
        id="pol_dialectical_materialism",
        category="politics",
        title="辩证唯物主义认识论——实践与认识",
        context="讲解实践是认识的基础：实践是认识的来源、动力、检验标准、目的",
        prompt=(
            "辩证唯物主义认识论关系图："
            "中央'实践'大圆（红色实心圆，标注'实践是认识的基础'），"
            "向外辐射四条粗箭头指向四个'认识'相关节点："
            "① 实践是认识的来源（上，蓝框）：图标=人手+工具，"
            "例'神农尝百草→医药知识 / 天文观测→宇宙认知'，标注'一切真知都来源于实践'，"
            "② 实践是认识发展的动力（右，绿框）：图标=齿轮+前进箭头，"
            "例'工程需求推动力学发展 / 农业发展推动天文学进步'，标注'实践不断提出新问题→推动认识深化'，"
            "③ 实践是检验认识真理性的唯一标准（下，金框）：图标=天平+对勾，"
            "例'卫星导航验证相对论 / 临床实验验证药效'，标注'真理标准问题大讨论(1978)'，"
            "④ 实践是认识的目的（左，紫框）：图标=靶心，"
            "例'学以致用 / 理论服务实践'，标注'认识世界的目的是改造世界'，"
            "四个框之间用虚线连接表示'→感性认识→理性认识→实践检验→'循环螺旋上升，"
            "底部：认识反作用于实践（细虚线回路箭头从认识回到实践，标注'科学理论指导实践'），"
            "政治教材概念图风格，白底，彩色箭头+框，中文标注，高中政治必修四插图"
        ),
    ),
]

# ── 物理 ──────────────────────────────────────────────────────────

SCIENCE_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="physics_newton_laws",
        category="physics",
        title="牛顿运动三定律图解",
        context="讲解牛顿三大定律：惯性定律、F=ma、作用力与反作用力",
        prompt=(
            "牛顿运动三定律三联图解："
            "第一定律·惯性定律（左上，蓝色卡片）："
            "图示：桌面上的球静止（无外力→静止）vs 运动中的球在光滑/粗糙面（无摩擦→匀速直线/有摩擦→减速停止），"
            "公式框：ΣF=0 → v=常量，标注'力是改变运动状态的原因，不是维持运动的原因'，"
            "第二定律·加速度定律（右上，红色卡片）："
            "图示：同一辆车先后用小力度推（a₁小）和大力度推（a₂大），用双箭头标注F与a成正比，"
            "同一力推空车（m小，a大）vs 满载车（m大，a小），标注a与m成反比，"
            "公式框（特大）：F=ma，标注'1N=1kg·m/s²'，"
            "第三定律·作用与反作用（底部，绿色卡片，横向宽框）："
            "两个小人面对面站滑板上，A推B（F_A→B，红色箭头），B也推动A后退（F_B→A，蓝色箭头），"
            "标注'大小相等/方向相反/作用在不同物体上/同时产生同时消失'，"
            "火箭发射示意（燃气向下喷→箭体向上飞，配作用力/反作用力箭头），"
            "教科书物理插图风格，白底，彩色分区，公式+图示，中文标注，高中物理必修一插图"
        ),
    ),
    ImageGenerationPrompt(
        id="physics_electromagnetic_induction",
        category="physics",
        title="电磁感应——法拉第定律",
        context="讲解电磁感应：磁通量变化→感应电动势、楞次定律、右手定则",
        prompt=(
            "电磁感应教学综合图："
            "左侧：条形磁铁插入线圈（N极向下插入），线圈连接灵敏电流计，"
            "磁铁向下运动→线圈中磁通量Φ增大→电流计指针向右偏转（红色指针偏转标注'感应电流'），"
            "标注'磁通量增加→感应电流的磁场阻碍增加（楞次定律-来拒）'，"
            "右侧对称：条形磁铁拔出线圈（向上拔出），磁通量Φ减小→电流计指针向左偏转，"
            "标注'磁通量减少→感应电流的磁场阻碍减少（楞次定律-去留）'，"
            "中间核心公式框：ε=-N·ΔΦ/Δt（法拉第电磁感应定律，特大号红框），"
            "N=线圈匝数/ΔΦ=磁通量变化量/Δt=变化时间/负号=楞次定律（方向相反），"
            "底部右手定则示意图：右手握住螺线管，四指指向电流方向→拇指指向N极（即感应电流产生的磁场方向），"
            "右下角：导体棒切割磁感线模型（导体棒在磁场中向右运动→右手定则判断电流方向），"
            "教科书物理插图风格，白底，蓝红色调，公式+实验示意+手型，中文标注，高中物理选择性必修插图"
        ),
    ),
    ImageGenerationPrompt(
        id="physics_wave_interference",
        category="physics",
        title="波的干涉与衍射",
        context="讲解机械波：干涉（加强点/减弱点）、衍射（明显衍射条件）、驻波",
        prompt=(
            "波的干涉衍射教学对比图："
            "上半部·干涉（水波实验示意）："
            "两个点波源S₁、S₂（俯视图，标红点），各自发出同心圆波纹（蓝色虚线=波峰交错叠加），"
            "实线标注振动加强线（波峰+波峰或波谷+波谷相遇→振幅加倍，红色辐射线，间距均匀），"
            "虚线标注振动减弱线（波峰+波谷相遇→振幅抵消，蓝色辐射虚线），"
            "标注'加强条件：Δs=kλ (k=0,1,2...)' / '减弱条件：Δs=(2k+1)λ/2'，"
            "下半部·衍射（对比图）："
            "左侧：大缝隙（缝宽d>>波长λ，波几乎直线穿过→衍射不明显，光线几何阴影区标注），"
            "右侧：小缝隙（缝宽d≈波长λ，波绕过缝隙呈半圆形扩散→衍射明显，半圆波纹标注），"
            "标注'明显衍射条件：障碍物/缝尺寸 ≤ 波长λ'，"
            "右下角：驻波示意（入射波+反射波叠加→波节N/波腹A交替排列，标注λ/2间距），"
            "教科书物理插图风格，白底，蓝红波纹，公式+图示，中文标注，高中物理选择性必修插图"
        ),
    ),
]

# ── 化学 ──────────────────────────────────────────────────────────

SCIENCE2_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="chem_periodic_table_pattern",
        category="chemistry",
        title="元素周期律——原子半径与电离能递变",
        context="讲解元素周期律：同周期/同主族原子半径、电离能、电负性的递变规律",
        prompt=(
            "元素周期律递变趋势图："
            "简化周期表（1-18族×4周期，每格元素符号+原子序数），"
            "用渐变色+箭头标注两个递变趋势："
            "① 原子半径递变（上半图，尺寸热力图）："
            "同周期从左到右→半径逐渐减小（由大到小：Na>Mg>Al>Si>P>S>Cl，红色→蓝色渐变，"
            "同主族从上到下→半径逐渐增大（由小到大：Li<Na<K<Rb，浅色→深色渐变），"
            "标注原因：核电荷数增大（左→右吸引力↑） vs 电子层数增加（上→下半径↑），"
            "② 第一电离能递变（下半图，能量柱状图）："
            "同周期总趋势增大（但有反常：IIA族>IIIA族，VA族>VIA族，因电子排布稳定效应），"
            "同主族从上到下减小，"
            "右下角：电负性递变小图（F最大4.0，Cs最小0.7，箭头标注递变方向），"
            "教科书化学图表风格，白底，彩色渐变+箭头，中文标注，高中化学必修一插图"
        ),
    ),
    ImageGenerationPrompt(
        id="chem_electrolysis",
        category="chemistry",
        title="电解池——电解氯化铜溶液",
        context="讲解电解原理：电解池装置、阳极氧化/阴极还原、放电顺序、电解方程式",
        prompt=(
            "电解氯化铜溶液实验装置示意图："
            "U形管（或烧杯）盛CuCl₂溶液（蓝绿色），标注离子：Cu²⁺（蓝色小球+2价标注）、Cl⁻（绿色小球标注）、"
            "H⁺、OH⁻（少量，来自水电离），"
            "左侧石墨电极连接直流电源正极（红色导线，标注'阳极'）："
            "Cl⁻移向阳极→失去电子→2Cl⁻-2e⁻=Cl₂↑"
            "生成气泡（氯气，黄绿色气泡，用湿润淀粉KI试纸检测→变蓝验证），"
            "标注'阳极：氧化反应 放电顺序：Cl⁻>OH⁻'，"
            "右侧石墨电极连接电源负极（蓝色导线，标注'阴极'）："
            "Cu²⁺移向阴极→得到电子→Cu²⁺+2e⁻=Cu↓"
            "电极表面析出红色固体（铜，红色镀层），标注'阴极：还原反应 放电顺序：Cu²⁺>H⁺'，"
            "总反应方程式（底部大框）：CuCl₂ —通电→ Cu↓ + Cl₂↑"
            "条件标注'直流电/电极不参与反应（惰性电极石墨/铂）'，"
            "教科书化学实验插图风格，白底，蓝绿溶液色，中文标注，高中化学选择性必修插图"
        ),
    ),
    ImageGenerationPrompt(
        id="chem_organic_reactions",
        category="chemistry",
        title="有机物转化关系——烃及其衍生物",
        context="讲解有机化学转化：烷→烯→醇→醛→酸→酯，以及反应类型(取代/加成/消去/氧化/酯化)",
        prompt=(
            "有机物转化关系网络图："
            "核心转化链从左到右箭头连接："
            "烷烃（CH₄甲烷，灰色方框，标注'取代反应：与Cl₂光照→CH₃Cl'）→"
            "烯烃（C₂H₄乙烯，橙色方框，标注'加成反应：+H₂O→乙醇 / +Br₂→1,2-二溴乙烷'）→"
            "醇（C₂H₅OH乙醇，蓝色方框，标注'消去反应：浓H₂SO₄ 170°C→C₂H₄↑ / 氧化→乙醛'）→"
            "醛（CH₃CHO乙醛，绿色方框，标注'氧化：+O₂催化剂→乙酸 / 还原+H₂→乙醇'）→"
            "酸（CH₃COOH乙酸，红色方框，标注'酯化反应：+醇 浓H₂SO₄△→酯+H₂O'）→"
            "酯（CH₃COOC₂H₅乙酸乙酯，紫色方框，标注'水解：+H⁺/OH⁻→酸+醇'），"
            "转化链外围标注反应条件：催化剂/温度/溶剂，反应类型用不同颜色箭头区分："
            "取代=灰箭头/加成=橙箭头/消去=蓝箭头/氧化=浅绿箭头/酯化=红箭头/水解=紫箭头（虚线反向），"
            "每个方框内含：结构简式+官能团名称+典型反应方程式（小字），"
            "流程图风格，白底，彩色方框+箭头，中文标注，高中化学选择性必修插图"
        ),
    ),
]

# ── 生物 ──────────────────────────────────────────────────────────

NATURE_PROMPTS: list[ImageGenerationPrompt] = [
    ImageGenerationPrompt(
        id="bio_enzyme_reaction",
        category="biology",
        title="酶促反应——影响酶活性的因素",
        context="讲解酶的特性：高效性、专一性、作用条件温和（温度/pH对酶活性的影响）",
        prompt=(
            "影响酶活性因素教学对比图，三组实验图示并列："
            "① 温度对酶活性影响（左，温度计图标+曲线图）："
            "最适温度T_opt（37°C人体酶/60°C耐热酶），低温0°C→酶活性低但可恢复（冰晶图标+虚线箭头回暖→恢复），"
            "高温>80°C→酶变性失活不可逆（蛋白质变性示意：规则折叠→不规则聚集，标注'空间结构破坏'），"
            "活性-温度曲线呈钟形（标注最适温度），"
            "② pH对酶活性影响（中，pH标尺+曲线图）："
            "不同酶最适pH不同：胃蛋白酶最适pH≈2（强酸，标注胃液环境），胰蛋白酶最适pH≈8（弱碱，标注肠液环境），"
            "过酸/过碱均导致变性失活，活性-pH曲线呈钟形，"
            "③ 抑制剂影响（右，竞争性vs非竞争性示意）："
            "竞争性抑制：抑制剂（红色三角）与底物（蓝色圆形）结构相似→竞争活性位点（锁钥模型，抑制剂占据活性中心），"
            "可通过增加底物浓度缓解，"
            "非竞争性抑制：抑制剂结合酶的其他部位→改变活性位点构型→底物无法结合（别构抑制示意），"
            "无法通过增加底物缓解，"
            "每图下方标注实验验证方法（如气泡产生速率/剩余底物量），"
            "教科书生物实验插图风格，白底，蓝红配色，中英双语标注，高中生物必修一插图"
        ),
    ),
    ImageGenerationPrompt(
        id="bio_mitosis_meiosis",
        category="biology",
        title="有丝分裂与减数分裂对比",
        context="讲解细胞分裂：有丝分裂（体细胞，4期）vs 减数分裂（生殖细胞，两次连续分裂）",
        prompt=(
            "有丝分裂与减数分裂对比图，上下两行并排："
            "上行·有丝分裂（蓝色边框，标注'体细胞/2n→2n'）："
            "间期（DNA复制，染色质→染色体，中心体复制，标注'DNA 2n→4n'）→"
            "前期（核仁核膜消失+纺锤体形成，染色体散乱分布）→"
            "中期（染色体着丝粒排列在赤道板，纺锤丝连接，最整齐的排列形态）→"
            "后期（着丝粒分裂，姐妹染色单体分开，纺锤丝牵引移向两极，染色体数加倍瞬间）→"
            "末期（核仁核膜重新出现，染色体→染色质，细胞质分裂→形成两个子细胞 各2n），"
            "下行·减数分裂（红色边框，标注'原始生殖细胞/2n→n'）："
            "减Ⅰ：间期（同有丝分裂）→前期Ⅰ（联会+四分体+交叉互换 同源染色体配对示意，红色/蓝色大小形态相同的同源染色体配对）→"
            "中期Ⅰ（同源染色体对排列在赤道板，非同源染色体自由组合）→"
            "后期Ⅰ（同源染色体分离→移向两极，非同源染色体自由组合→遗传多样性来源）→"
            "末期Ⅰ→→减Ⅱ：前期Ⅱ→中期Ⅱ（类似有丝分裂但染色体数已减半，n条）→"
            "后期Ⅱ（姐妹染色单体分离）→末期Ⅱ（形成4个子细胞各n/基因型不同标注），"
            "底部对比总结表：有丝分裂 vs 减数分裂（分裂次数/子细胞数/染色体数变化/DNA数变化/发生部位/意义），"
            "教科书细胞生物学插图风格，白底，蓝(有丝)红(减数)区分，中文标注，高中生物必修二插图"
        ),
    ),
    ImageGenerationPrompt(
        id="bio_ecosystem_energy_flow",
        category="biology",
        title="生态系统能量流动——林德曼效率",
        context="讲解生态系统中能量流动：单向流动、逐级递减、10%-20%传递效率、能量金字塔",
        prompt=(
            "生态系统能量流动教学图："
            "左侧·能量流动示意图（纵向自上而下）："
            "太阳能（顶部太阳图标，总辐射能100%）→ 箭头向下，标注'生产者固定约1%'→"
            "生产者（第一营养级，绿色层，草本/乔木图标，标注固定能量值 e.g. 50000kJ/m²·年）→"
            "粗箭头（约10%-20%传递）→"
            "初级消费者（第二营养级，黄色层，食草动物图标（兔/蝗虫），标注 e.g. 5000kJ）→"
            "细箭头→次级消费者（第三营养级，橙色层，小型食肉动物图标（蛙/蛇），标注 e.g. 500kJ）→"
            "更细箭头→三级消费者（第四营养级，红色层，大型食肉动物图标（鹰/虎），标注 e.g. 50kJ），"
            "各营养级之间标注箭头粗细递减+能量数值递减，每级标注能量去向方框（呼吸消耗热能耗散（约50%-70%）/残体枯枝落叶→分解者/流入下一营养级），"
            "右侧·能量金字塔（三角形△，底部宽顶部窄）："
            "塔内分层：生产者（宽底层，绿色）→初级消费者→次级消费者→三级消费者（窄顶层，红色），"
            "各层标注能量数值+宽度比例≈能量比例，"
            "金字塔右侧标注'林德曼效率≈10%-20%'/'单向流动不可逆（热力学第二定律）'/'逐级递减：营养级一般不超过4-5级'，"
            "教科书生态学插图风格，白底，绿黄橙红渐变色，中文标注，高中生物选择性必修插图"
        ),
    ),
]

# ═══════════════════════════════════════════════════════════════════
# 分类 → Prompt 列表映射（v6 泛化）
# ═══════════════════════════════════════════════════════════════════

CATEGORY_ILLUSTRATIONS: dict[str, list[ImageGenerationPrompt]] = {
    "scenery": SCENERY_PROMPTS,
    "tech": TECH_PROMPTS,
    "portrait": PORTRAIT_PROMPTS,
    "architecture": ARCHITECTURE_PROMPTS,
    "animal": ANIMAL_PROMPTS,
    "food": FOOD_PROMPTS,
    "art": SCIENCE_PROMPTS,
    # 向后兼容旧学科名
    "chinese": SCENERY_PROMPTS,
    "math": TECH_PROMPTS,
    "english": PORTRAIT_PROMPTS,
    "history": ARCHITECTURE_PROMPTS,
    "geography": ANIMAL_PROMPTS,
    "politics": FOOD_PROMPTS,
    "physics": TECH_PROMPTS,
    "chemistry": SCIENCE2_PROMPTS,
    "biology": NATURE_PROMPTS,
}

# 中文分类名（用于展示，v6: 保留旧学科名向后兼容）
CATEGORY_LABELS: dict[str, str] = {
    "scenery": "风景",
    "portrait": "人物",
    "animal": "动物",
    "tech": "科技",
    "food": "美食",
    "architecture": "建筑",
    "art": "艺术",
    # 向后兼容旧学科名
    "chinese": "语文",
    "math": "数学",
    "english": "英语",
    "history": "历史",
    "geography": "地理",
    "politics": "政治",
    "physics": "物理",
    "chemistry": "化学",
    "biology": "生物",
    "science": "科学",
}
# 向后兼容别名
CATEGORY_LABELS = CATEGORY_LABELS

# ═══════════════════════════════════════════════════════════════════
# 终端展示工具
# ═══════════════════════════════════════════════════════════════════


def _sep(title: str = "", char: str = "=", width: int = 70) -> None:
    """打印分隔线."""
    if title:
        print(f"\n{char * width}")
        print(f"  {title}")
        print(f"{char * width}")
    else:
        print(char * width)


def _print_prompt_card(item: ImageGenerationPrompt, index: int) -> None:
    """打印单条 prompt 的卡片信息."""
    label = CATEGORY_LABELS.get(item.category, item.category)
    print(f"\n  ┌─ [{label}] #{index}  {item.id}")
    print(f"  ├─ 标题: {item.title}")
    print(f"  ├─ 场景: {item.context}")
    print(f"  ├─ LLM 生成 Prompt ({len(item.prompt)} 字符):")
    # 按 65 字符折行
    for i in range(0, len(item.prompt), 65):
        print(f"  │  {item.prompt[i:i+65]}")
    print(f"  └─ {'─' * 50}")


# ═══════════════════════════════════════════════════════════════════
# 批量汇总工具
# ═══════════════════════════════════════════════════════════════════


def _print_batch_summary(
    results: list[dict],
    mode_label: str,
    batch_start: float,
) -> None:
    """打印批量运行汇总表."""
    total_elapsed = time.perf_counter() - batch_start
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    total = len(results)

    # 按分类分组统计
    by_category: dict[str, dict] = {}
    for r in results:
        s = r["category"]
        if s not in by_category:
            by_category[s] = {"total": 0, "ok": 0, "fail": 0, "total_score": 0.0, "score_count": 0}
        by_category[s]["total"] += 1
        if r["status"] == "OK":
            by_category[s]["ok"] += 1
            if r.get("final_score", 0) > 0:
                by_category[s]["total_score"] += r["final_score"]
                by_category[s]["score_count"] += 1
        else:
            by_category[s]["fail"] += 1

    _sep(f"{mode_label} 批量汇总", "═")
    print(f"""
  ╔══════════════════════════════════════════════════════════════════╗
  ║  上游模块模拟器 —— {mode_label} 批量测试报告                    ║
  ╠══════════════════════════════════════════════════════════════════╣
  ║  总 prompt 数:   {total:<4}    成功: {ok:<4}    失败: {fail:<4}                       ║
  ║  总耗时:         {total_elapsed:.1f}s                                          ║
  ╠══════════════════════════════════════════════════════════════════╣""")

    for cat in sorted(by_category.keys()):
        stats = by_category[cat]
        label = CATEGORY_LABELS.get(cat, cat)
        avg_score = f"avg={stats['total_score']/stats['score_count']:.3f}" if stats["score_count"] > 0 else ""
        bar_ok = "█" * stats["ok"] + ("▒" * stats["fail"] if stats["fail"] else "")
        print(f"  ║  {label:<6} {stats['ok']}/{stats['total']}  {bar_ok:<20} {avg_score:<12}║")

    print(f"""  ╠══════════════════════════════════════════════════════════════════╣
  ║  完整链路: 上游LLM输出 → Draw生图 → VLM评测 → Refine优化         ║
  ║            → CLIP编码 → Milvus入库 → 可检索知识库                ║
  ╚══════════════════════════════════════════════════════════════════╝
""")

    # 失败明细
    failures = [r for r in results if r["status"] == "FAIL"]
    if failures:
        print(f"  ⚠ 失败明细 ({len(failures)} 条):")
        for f in failures:
            label = CATEGORY_LABELS.get(f["category"], f["category"])
            err_short = (f["error"] or "")[:80]
            print(f"    [{label}] {f['id']} | {f['title']} | {err_short}")

    # 落盘汇总 JSON
    summary_path = IMAGE_DIR.parent / "upstream_batch_summary.json"
    summary_path.write_text(
        json.dumps({
            "mode": mode_label,
            "total": total, "ok": ok, "fail": fail,
            "total_elapsed_s": round(total_elapsed, 1),
            "by_category": {
                cat: {
                    "label": CATEGORY_LABELS.get(cat, cat),
                    "total": s["total"], "ok": s["ok"], "fail": s["fail"],
                    "avg_score": round(s["total_score"] / s["score_count"], 4) if s["score_count"] > 0 else None,
                }
                for cat, s in by_category.items()
            },
            "results": results,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  批量汇总 JSON 已保存: {summary_path}")


# ═══════════════════════════════════════════════════════════════════
# 运行模式
# ═══════════════════════════════════════════════════════════════════


def _register_drawers() -> None:
    """注册生图适配器."""
    if "doubao" not in DRAWER_REGISTRY:
        DRAWER_REGISTRY["doubao"] = DoubaoDrawer()
    if "tongyi" not in DRAWER_REGISTRY:
        DRAWER_REGISTRY["tongyi"] = TongyiDrawer()


def mode_list(category: str | None = None) -> int:
    """列出所有上游 LLM 输出的 prompt，不调用 API."""
    cats_to_show = (
        list(CATEGORY_ILLUSTRATIONS.keys()) if category == "all" or category is None
        else [category]
    )

    total_count = 0
    for cat in cats_to_show:
        items = CATEGORY_ILLUSTRATIONS.get(cat)
        if items is None:
            print(f"未知分类: {cat}")
            print(f"可用分类: {', '.join(CATEGORY_ILLUSTRATIONS.keys())}")
            return 1

        label = CATEGORY_LABELS.get(cat, cat)
        _sep(f"【{label}】分类 — 上游 prompt 列表", "━")
        print(f"  共 {len(items)} 条 prompt")

        for i, item in enumerate(items, 1):
            _print_prompt_card(item, i)
            total_count += 1

    _sep()
    print(f"\n  总计: {total_count} 条 prompt，覆盖 {len(cats_to_show)} 个分类")
    print(f"  用法: python -m demo.upstream_simulator_demo --mode dry-run --category <分类>")
    print(f"         python -m demo.upstream_simulator_demo --mode pipeline --category <分类> --id <prompt_id>")
    return 0


def mode_dry_run(
    category: str = "all",
    model: str | None = None,
) -> int:
    """模拟运行：不调 API，展示完整流程."""
    model = model or detect_default_model()

    cats_to_run = (
        list(CATEGORY_ILLUSTRATIONS.keys()) if category == "all"
        else [category]
    )

    _sep("上游模块模拟器 —— Dry-Run 模式", "═")
    print(f"  模拟流程: 上游 LLM 输出 prompt → picture2 Pipeline → 生图/评测/优化")
    print(f"  分类范围: {', '.join(CATEGORY_LABELS.get(c, c) for c in cats_to_run)}")
    print(f"  生图模型: {model}")
    print(f"  Pipeline 模式: clip_enrich（CLIP检索 + 评测迭代 + 入库）")
    print(f"  说明: 不调用任何 API，仅展示 prompt + 模拟 pipeline 输入/输出结构")

    total = 0
    for cat in cats_to_run:
        items = CATEGORY_ILLUSTRATIONS.get(cat, [])
        label = CATEGORY_LABELS.get(cat, cat)
        _sep(f"【{label}】{len(items)} 条 prompt", "━")

        for i, item in enumerate(items, 1):
            _print_prompt_card(item, i)

            # 模拟 PipelineRequest
            req = PipelineRequest(
                prompt=item.prompt,
                model=model,
                max_iterations=3,
                eval_threshold=EVAL_THRESHOLD,
                category=item.category,
            )
            print(f"\n  ── 模拟 PipelineRequest ──")
            print(f"  model={req.model} | max_iter={req.max_iterations} | threshold={req.eval_threshold} | category={req.category}")

            # 模拟 PipelineResponse
            print(f"  ── 模拟 PipelineResponse（假设一次迭代达标）──")
            print(f"  final_image_path: storage/images/{item.id}.png")
            print(f"  final_prompt (优化后, {len(item.prompt)}→{len(item.prompt)+15} 字符)")
            print(f"  final_score: 0.88  (VLM 五维度评测)")
            print(f"  total_iterations: 2")
            print(f"  stopped_reason: threshold_met (score 0.88 >= {EVAL_THRESHOLD})")
            print(f"  db_record_id: {100 + total}")
            print(f"  matched_prompts: 3 条相似历史 prompt 被检索到（仅用于复用判定，不修改 prompt）")

            total += 1

    _sep("Dry-Run 汇总", "═")
    print(f"""
  ┌─────────────────────────────────────────────────────────────────┐
  │  上游模块模拟 (Upstream Simulator)                               │
  ├─────────────────────────────────────────────────────────────────┤
  │  总 prompt 数:         {total:<4}                                     │
  │  覆盖分类:             {len(cats_to_run)} ({', '.join(CATEGORY_LABELS.get(c, c) for c in cats_to_run)})  │
  │  生图模型:             {model:<10}                              │
  │  Pipeline 模式:        clip_enrich                              │
  │  API 调用:             已跳过 (dry-run)                          │
  ├─────────────────────────────────────────────────────────────────┤
  │  上游模块行为模拟说明:                                          │
  │  1. 接收内容描述 DSL（主题 + 场景 + 视觉元素）                   │
  │  2. LLM 将结构化内容素材转化为自然语言插图描述                   │
  │  3. 输出 → picture2 PipelineRequest.prompt                      │
  │  4. picture2 闭环: CLIP检索 → Draw → Evaluate → Refine → 入库   │
  └─────────────────────────────────────────────────────────────────┘
""")
    return 0


def mode_pipeline(
    category: str,
    prompt_id: str | None,
    model: str | None,
    max_iterations: int = 3,
    reuse_threshold: float = 0.77,
) -> int:
    """闭环模式：完整 Pipeline（CLIP检索 → Draw → Evaluate → Refine → 入库）—— 支持 --category all 批量."""
    model = model or detect_default_model()

    cats_to_run = (
        list(CATEGORY_ILLUSTRATIONS.keys()) if category == "all"
        else [category]
    )

    _sep("上游模块 → picture2 完整 Pipeline 批量测试", "═")
    print(f"  分类数: {len(cats_to_run)} | 生图模型: {model}")
    print(f"  Pipeline 模式: clip_enrich（CLIP检索 + 评测迭代 + 入库）")
    print(f"  最大迭代: {max_iterations} | 评测阈值: {EVAL_THRESHOLD}")
    print(f"  CLIP/Milvus: 检索复用阈值={reuse_threshold}（top-1相似度≥{reuse_threshold}→直接复用；未命中→生图+入库）")

    _register_drawers()
    pipeline = ImagePipeline()
    batch_start = time.perf_counter()
    results: list[dict] = []

    total_prompts = sum(len(CATEGORY_ILLUSTRATIONS.get(c, [])) for c in cats_to_run)
    done = 0

    for cat in cats_to_run:
        items = CATEGORY_ILLUSTRATIONS.get(cat, [])
        if prompt_id:
            items = [it for it in items if it.id == prompt_id]
            if not items:
                print(f"  未找到 prompt: {prompt_id}")
                continue

        for item in items:
            done += 1
            label = CATEGORY_LABELS.get(cat, cat)
            _sep(f"[{done}/{total_prompts}] [{label}] {item.title}")
            print(f"  场景: {item.context}")
            print(f"  上游 LLM Prompt ({len(item.prompt)} 字符): {item.prompt[:120]}...")

            req = PipelineRequest(
                prompt=item.prompt,
                model=model,
                max_iterations=max_iterations,
                eval_threshold=EVAL_THRESHOLD,
                category=item.category,
                reuse_threshold=reuse_threshold,
            )

            t0 = time.perf_counter()
            try:
                resp = asyncio.run(pipeline.run(req))
                elapsed = time.perf_counter() - t0

                # 复用命中时特别展示
                if resp.stopped_reason == "reused":
                    print(f"  [HIT] 🎯 检索命中，直接复用已有图片 ({elapsed:.1f}s)——零 API 调用！")
                    print(f"    reused_from_record_id={resp.reused_from_record_id} | similarity 已达标")
                    print(f"    final_image: {resp.final_image_path}")
                    print(f"    final_score={resp.final_score:.4f} (原始评测分)")
                else:
                    print(f"  [OK] Pipeline 完成 ({elapsed:.1f}s)")
                    print(f"    final_score={resp.final_score:.4f} | iterations={resp.total_iterations} | stopped={resp.stopped_reason}")
                    print(f"    final_image: {resp.final_image_path}")

                # 迭代历史摘要（复用命中时 history 为空，自动跳过）
                for it in resp.history:
                    dim_scores = ", ".join(
                        f"{d.dimension[:4]}={d.score:.2f}" for d in it.dimension_scores[:3]
                    )
                    print(f"      第{it.iteration}轮: score={it.overall_score:.4f} [{dim_scores}...]")
                    if it.optimized_prompt:
                        print(f"        → 优化后 prompt ({len(it.optimized_prompt)} 字符)")

                # 保存摘要
                summary_path = IMAGE_DIR.parent / f"upstream_{item.id}_summary.json"
                summary_path.write_text(
                    json.dumps(resp.model_dump(), ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )

                results.append({
                    "category": cat, "id": item.id, "title": item.title,
                    "status": "OK", "elapsed": elapsed,
                    "final_score": resp.final_score,
                    "iterations": resp.total_iterations,
                    "stopped_reason": resp.stopped_reason,
                    "image_path": resp.final_image_path,
                    "db_record_id": resp.db_record_id,
                    "reused_from_record_id": resp.reused_from_record_id,
                    "error": None,
                })
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                logger.error("pipeline.error | category=%s id=%s error=%s", cat, item.id, exc)
                print(f"  [FAIL] ({elapsed:.1f}s) {exc}")
                results.append({
                    "category": cat, "id": item.id, "title": item.title,
                    "status": "FAIL", "elapsed": elapsed,
                    "final_score": 0, "iterations": 0,
                    "stopped_reason": "error", "image_path": None,
                    "db_record_id": None, "error": str(exc),
                })

    # 批量汇总
    _print_batch_summary(results, "Pipeline 闭环", batch_start)
    ok = sum(1 for r in results if r["status"] == "OK")
    return 0 if ok > 0 else 1


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> None:
    # Windows 终端 UTF-8
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    all_categories = list(CATEGORY_ILLUSTRATIONS.keys())

    parser = argparse.ArgumentParser(
        description="上游模块模拟器 —— 模拟 LLM 生成各类场景图片描述 → picture2 生图闭环",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例:
  # 列出所有分类的 prompt
  python -m demo.upstream_simulator_demo --mode list

  # dry-run 全分类 28 条 prompt 模拟流程（零 API 调用）
  python -m demo.upstream_simulator_demo --mode dry-run

  # ★ 一键全链路：多分类28条prompt → CLIP+Draw+VLM评测+Milvus入库（调 API）
  python -m demo.upstream_simulator_demo --mode pipeline --category all

  # 单分类 Pipeline
  python -m demo.upstream_simulator_demo --mode pipeline --category 科技

  # 单条 prompt Pipeline 闭环
  python -m demo.upstream_simulator_demo --mode pipeline --category 美食 --id food_ramen

可用分类: {', '.join(all_categories)}
""",
    )

    parser.add_argument(
        "--mode",
        default="list",
        choices=["list", "dry-run", "pipeline"],
        help=(
            "运行模式: list=列prompt(无API) / dry-run=模拟流程(无API) / "
            "pipeline=CLIP检索+评测迭代+入库闭环(调API)"
        ),
    )
    parser.add_argument(
        "--category",
        "--subject",  # 向后兼容别名
        default=None,
        dest="category",
        help=f"分类: all | {' | '.join(all_categories)} （默认 all）",
    )
    parser.add_argument(
        "--id",
        default=None,
        dest="prompt_id",
        help="指定 prompt ID（不指定则运行该分类全部 prompt）",
    )
    parser.add_argument(
        "--model",
        default=detect_default_model(),
        choices=["doubao", "tongyi"],
        help="生图模型（自动检测已配置的 API Key）",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="最大迭代次数（pipeline 模式，默认 3）",
    )
    parser.add_argument(
        "--reuse-threshold",
        type=float,
        default=0.77,
        help="检索复用阈值: top-1 CLIP 相似度 ≥ 此值时直接复用已有图片（默认 0.77）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟运行（同 --mode dry-run，保留向后兼容）",
    )

    args = parser.parse_args(argv)

    # 确定分类
    category = args.category or "all"

    # --dry-run 标志 → 强制 dry-run mode
    mode = "dry-run" if args.dry_run else args.mode

    # 验证分类
    if category != "all" and category not in CATEGORY_ILLUSTRATIONS:
        print(f"未知分类: {category}")
        print(f"可用: all | {' | '.join(all_categories)}")
        sys.exit(1)

    # 分发
    if mode == "list":
        sys.exit(mode_list(category if category != "all" else None))
    elif mode == "dry-run":
        sys.exit(mode_dry_run(
            category=category,
            model=args.model,
        ))
    elif mode == "pipeline":
        sys.exit(mode_pipeline(
            category=category,
            prompt_id=args.prompt_id,
            model=args.model,
            max_iterations=args.max_iter,
            reuse_threshold=args.reuse_threshold,
        ))


if __name__ == "__main__":
    main()
