import React, { useEffect, useState, useCallback } from 'react';
import {
  Card, Image, Tag, Select, Space, Spin, Empty, Typography, Pagination, Slider,
  Tooltip,
} from 'antd';
import {
  PictureOutlined, FilterOutlined, ExpandOutlined, CloudOutlined, TagsOutlined,
} from '@ant-design/icons';
import { listData, EntityRecord, getCategories, CategoryOption } from '../api/milvusClient';

const { Text } = Typography;

const PAGE_SIZE = 20;

/** 将 image_path 字段转换为 /images/ API 的 URL */
function getImageUrl(imagePath: string): string {
  if (!imagePath) return '';
  const filename = imagePath.replace(/\\/g, '/').split('/').pop() || imagePath;
  return `/images/${filename}`;
}

/** 评分 → 颜色 */
function scoreColor(s: number): string {
  if (s >= 0.82) return 'green';
  if (s >= 0.7) return 'orange';
  return 'red';
}

const Gallery: React.FC = () => {
  const [images, setImages] = useState<EntityRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [category, setCategory] = useState('');
  const [minScore, setMinScore] = useState(0);
  const [categories, setCategories] = useState<CategoryOption[]>([]);

  // 加载动态分类列表
  useEffect(() => {
    getCategories()
      .then((res) => setCategories(res.categories || []))
      .catch(() => {
        // 静默回退 —— 至少保留"全部"选项
        setCategories([]);
      });
  }, []);

  const categoryOptions = [
    { value: '', label: '全部分类' },
    ...categories.map((c) => ({ value: c.value, label: c.label })),
  ];

  const fetchImages = useCallback(async () => {
    setLoading(true);
    try {
      const params: Record<string, unknown> = {
        limit: PAGE_SIZE,
        offset: (page - 1) * PAGE_SIZE,
      };
      // v6: 使用 category 参数（优先于旧 subject 参数）
      if (category) params.category = category;
      if (minScore > 0) params.min_score = minScore;
      const result = await listData('image_embeddings', params);
      setImages(result.data);
      setTotal(result.total);
    } catch {
      // 静默处理 —— 网络错误时保留旧数据
    } finally {
      setLoading(false);
    }
  }, [page, category, minScore]);

  useEffect(() => {
    fetchImages();
  }, [fetchImages]);

  // 切换筛选条件时回到第 1 页
  const handleCategoryChange = (val: string) => {
    setCategory(val);
    setPage(1);
  };

  const handleMinScoreChange = (val: number) => {
    setMinScore(val);
    setPage(1);
  };

  return (
    <div>
      {/* ── 顶栏：筛选 + 统计 ── */}
      <Card style={{ marginBottom: 16 }}>
        <Space size="middle" wrap align="center">
          <PictureOutlined style={{ fontSize: 18, color: '#1677ff' }} />
          <Text strong style={{ fontSize: 16 }}>照片浏览</Text>
          <Text type="secondary">浏览数据库中所有已生成的图片</Text>

          <span style={{ marginLeft: 24 }}>
            <FilterOutlined /> 分类:
          </span>
          <Select
            style={{ width: 160 }}
            value={category}
            onChange={handleCategoryChange}
            options={categoryOptions}
            placeholder="选择分类"
            showSearch
            filterOption={(input, option) =>
              (option?.label as string)?.includes(input) ?? false
            }
          />

          <span>最低评分:</span>
          <Slider
            style={{ width: 180 }}
            min={0}
            max={1}
            step={0.05}
            value={minScore}
            onChange={handleMinScoreChange}
            tooltip={{ formatter: (v) => v != null ? (v as number).toFixed(2) : '0.00' }}
          />

          <Tag color="blue" style={{ marginLeft: 16 }}>
            共 {total} 张
          </Tag>
          <Tag>第 {total > 0 ? (page - 1) * PAGE_SIZE + 1 : 0}-{Math.min(page * PAGE_SIZE, total)} 张</Tag>
        </Space>
      </Card>

      {/* ── 图片网格 ── */}
      <Spin spinning={loading} tip="加载中…">
        {images.length === 0 && !loading ? (
          <Card>
            <Empty description="数据库中暂无图片记录">
              <Text type="secondary">
                请先通过 Pipeline 或 Draw 接口生成图片并入库。
              </Text>
            </Empty>
          </Card>
        ) : (
          <>
            <div style={gridContainerStyle}>
              {images.map((record) => (
                <GalleryCard key={record.id} record={record} />
              ))}
            </div>

            {/* ── 分页 ── */}
            <Card style={{ marginTop: 16, textAlign: 'center' }}>
              <Pagination
                current={page}
                pageSize={PAGE_SIZE}
                total={total}
                onChange={setPage}
                showSizeChanger={false}
                showTotal={(t) => `共 ${t} 张图片`}
              />
            </Card>
          </>
        )}
      </Spin>
    </div>
  );
};

/** 单张图片卡片 —— 带 hover 浮层显示 prompt 和 VLM 标签 */
const GalleryCard: React.FC<{ record: EntityRecord }> = ({ record }) => {
  const [isHovered, setIsHovered] = useState(false);
  const [imgError, setImgError] = useState(false);

  const imageUrl = getImageUrl(record.image_path);
  const hasImage = imageUrl && !imgError;

  // 标签列表：category + tags
  const displayCategory = record.category || record.subject;

  return (
    <div
      style={cardOuterStyle}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* ── 图片区域 ── */}
      <div style={imageContainerStyle}>
        {hasImage ? (
          <Image
            src={imageUrl}
            alt={record.prompt}
            style={imageStyle}
            preview={{
              mask: (
                <div style={previewMaskStyle}>
                  <ExpandOutlined style={{ fontSize: 20, marginBottom: 8 }} />
                  <Text style={{ color: '#fff', fontSize: 13 }}>点击放大</Text>
                </div>
              ),
            }}
            onError={() => setImgError(true)}
            placeholder={
              <div style={placeholderStyle}>
                <PictureOutlined style={{ fontSize: 32, color: '#bbb' }} />
              </div>
            }
          />
        ) : (
          <div style={placeholderStyle}>
            <PictureOutlined style={{ fontSize: 32, color: '#ddd' }} />
            <Text type="secondary" style={{ marginTop: 8, fontSize: 12 }}>
              无图片
            </Text>
          </div>
        )}

        {/* ── Hover 浮层：显示完整 prompt + VLM 内容 ── */}
        <div
          style={{
            ...hoverOverlayStyle,
            opacity: isHovered ? 1 : 0,
            pointerEvents: 'none',
          }}
        >
          <div style={hoverContentStyle}>
            <Text
              strong
              style={{
                color: '#fff',
                fontSize: 12,
                marginBottom: 4,
                display: 'block',
                letterSpacing: 1,
              }}
            >
              PROMPT
            </Text>
            <Text
              style={{
                color: 'rgba(255,255,255,0.95)',
                fontSize: 13,
                lineHeight: 1.6,
                wordBreak: 'break-word',
              }}
            >
              {record.prompt}
            </Text>
            {record.optimized_prompt && (
              <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.25)', paddingTop: 8 }}>
                <Text
                  strong
                  style={{
                    color: '#ffd666',
                    fontSize: 11,
                    marginBottom: 4,
                    display: 'block',
                    letterSpacing: 1,
                  }}
                >
                  OPTIMIZED
                </Text>
                <Text
                  style={{
                    color: 'rgba(255,255,255,0.9)',
                    fontSize: 12,
                    lineHeight: 1.6,
                    wordBreak: 'break-word',
                  }}
                >
                  {record.optimized_prompt}
                </Text>
              </div>
            )}

            {/* ── VLM 内容解析字段 ── */}
            {(record.scene_description || record.style || (record.main_objects && record.main_objects.length > 0)) && (
              <div style={{ marginTop: 10, borderTop: '1px solid rgba(255,255,255,0.25)', paddingTop: 8 }}>
                <Text
                  strong
                  style={{
                    color: '#87e8de',
                    fontSize: 11,
                    marginBottom: 4,
                    display: 'block',
                    letterSpacing: 1,
                  }}
                >
                  VLM 内容解析
                </Text>
                {record.scene_description && (
                  <Text style={{ color: 'rgba(255,255,255,0.85)', fontSize: 11, display: 'block' }}>
                    场景: {record.scene_description.length > 80
                      ? record.scene_description.slice(0, 80) + '…'
                      : record.scene_description}
                  </Text>
                )}
                {record.style && (
                  <Text style={{ color: 'rgba(255,255,255,0.85)', fontSize: 11, display: 'block' }}>
                    风格: {record.style}
                  </Text>
                )}
                {record.main_objects && record.main_objects.length > 0 && (
                  <Text style={{ color: 'rgba(255,255,255,0.85)', fontSize: 11, display: 'block' }}>
                    主体: {record.main_objects.slice(0, 6).join('、')}
                  </Text>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── 右上角信息栏 ── */}
        <div style={infoBarStyle}>
          <Space size={4} wrap>
            <Tag
              color={scoreColor(record.score)}
              style={{ margin: 0, fontSize: 11 }}
            >
              评分: {record.score.toFixed(2)}
            </Tag>
            {displayCategory && (
              <Tag color="purple" style={{ margin: 0, fontSize: 11 }}>
                {displayCategory}
              </Tag>
            )}
            {record.source_type === 'uploaded' && (
              <Tooltip title="人工上传素材">
                <Tag color="cyan" style={{ margin: 0, fontSize: 11 }}>
                  <CloudOutlined /> 素材
                </Tag>
              </Tooltip>
            )}
          </Space>
        </div>
      </div>

      {/* ── 卡片底部：prompt 缩略 + tags ── */}
      <div style={cardFooterStyle}>
        <Text
          type="secondary"
          style={{ fontSize: 12 }}
          ellipsis={{ tooltip: record.prompt }}
        >
          {record.prompt.length > 60
            ? record.prompt.slice(0, 60) + '…'
            : record.prompt}
        </Text>

        {/* ── VLM 标签 + 类型 ── */}
        <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4, alignItems: 'center' }}>
          {record.content_type && (
            <Tag color="geekblue" style={{ fontSize: 10, margin: 0 }}>
              {record.content_type}
            </Tag>
          )}
          {record.tags && record.tags.length > 0 && (
            <>
              <TagsOutlined style={{ fontSize: 11, color: '#999' }} />
              {record.tags.slice(0, 4).map((tag, i) => (
                <Tag key={i} style={{ fontSize: 10, margin: 0 }}>{tag}</Tag>
              ))}
              {record.tags.length > 4 && (
                <Tooltip title={record.tags.slice(4).join('、')}>
                  <Text type="secondary" style={{ fontSize: 10 }}>
                    +{record.tags.length - 4}
                  </Text>
                </Tooltip>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

// ═══════════════════════════════════════════════
// 样式常量
// ═══════════════════════════════════════════════

const gridContainerStyle: React.CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))',
  gap: 16,
};

const cardOuterStyle: React.CSSProperties = {
  borderRadius: 10,
  overflow: 'hidden',
  background: '#fff',
  boxShadow: '0 1px 4px rgba(0,0,0,0.08)',
  transition: 'box-shadow 0.25s, transform 0.25s',
  cursor: 'pointer',
};

const imageContainerStyle: React.CSSProperties = {
  position: 'relative',
  width: '100%',
  aspectRatio: '1 / 1',
  overflow: 'hidden',
  background: '#f5f5f5',
};

const imageStyle: React.CSSProperties = {
  width: '100%',
  height: '100%',
  objectFit: 'cover',
  display: 'block',
};

const placeholderStyle: React.CSSProperties = {
  width: '100%',
  height: '100%',
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
  background: '#fafafa',
};

const previewMaskStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  justifyContent: 'center',
};

const hoverOverlayStyle: React.CSSProperties = {
  position: 'absolute',
  inset: 0,
  background: 'linear-gradient(to top, rgba(0,0,0,0.88) 0%, rgba(0,0,0,0.55) 60%, rgba(0,0,0,0.15) 100%)',
  transition: 'opacity 0.3s ease',
  display: 'flex',
  alignItems: 'flex-end',
  zIndex: 2,
};

const hoverContentStyle: React.CSSProperties = {
  padding: '16px 14px',
  width: '100%',
  maxHeight: '90%',
  overflowY: 'auto',
};

const infoBarStyle: React.CSSProperties = {
  position: 'absolute',
  top: 8,
  right: 8,
  zIndex: 3,
  display: 'flex',
  gap: 4,
};

const cardFooterStyle: React.CSSProperties = {
  padding: '10px 12px',
  borderTop: '1px solid #f0f0f0',
};

export default Gallery;
