import React, { useState, useEffect } from 'react';
import {
  Card, Input, Button, Select, Slider, Space, List, Tag, Typography,
  Image, Upload, message, Radio, Tooltip,
} from 'antd';
import {
  SearchOutlined, CameraOutlined, HistoryOutlined, ExperimentOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import {
  searchByText, searchByImage, searchSemantic, getSearchHistory, getCategories,
  SearchResultItem, SemanticSearchResultItem, SearchHistoryItem, CategoryOption,
} from '../api/milvusClient';

const { TextArea } = Input;
const { Title, Text } = Typography;

const Search: React.FC = () => {
  const [queryText, setQueryText] = useState('');
  const [category, setCategory] = useState<string | undefined>(undefined);
  const [topK, setTopK] = useState(5);
  const [searching, setSearching] = useState(false);
  const [searchMode, setSearchMode] = useState<'semantic' | 'text'>('semantic');
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [semanticResults, setSemanticResults] = useState<SemanticSearchResultItem[]>([]);
  const [queryTime, setQueryTime] = useState(0);
  const [totalInPartition, setTotalInPartition] = useState(0);
  const [history, setHistory] = useState<SearchHistoryItem[]>([]);
  const [categories, setCategories] = useState<CategoryOption[]>([]);

  useEffect(() => {
    getCategories().then(r => setCategories(r.categories || [])).catch(() => {});
    getSearchHistory(10).then(setHistory).catch(() => {});
  }, []);

  const handleTextSearch = async () => {
    if (!queryText.trim()) {
      message.warning('请输入搜索文本');
      return;
    }
    setSearching(true);
    setSemanticResults([]);
    setResults([]);
    try {
      if (searchMode === 'semantic') {
        const res = await searchSemantic({ text: queryText, top_k: topK, category });
        setSemanticResults(res.results);
        setQueryTime(res.query_time_ms);
        setTotalInPartition(res.total_in_partition);
      } else {
        const res = await searchByText({ text: queryText, top_k: topK, subject: category });
        setResults(res.results);
        setQueryTime(res.query_time_ms);
        setTotalInPartition(res.total_in_partition);
      }
      // 刷新历史
      getSearchHistory(10).then(setHistory).catch(() => {});
    } catch (e: unknown) {
      message.error(`检索失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setSearching(false);
    }
  };

  const handleImageSearch = async (file: File) => {
    setSearching(true);
    setResults([]);
    setSemanticResults([]);
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('top_k', String(topK));
      if (category) formData.append('category', category);
      const res = await searchByImage(formData);
      setResults(res.results);
      setQueryTime(res.query_time_ms);
      setTotalInPartition(res.total_in_partition);
    } catch (e: unknown) {
      message.error(`以图搜图失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setSearching(false);
    }
  };

  const similarityColor = (sim: number) => {
    if (sim >= 0.9) return 'green';
    if (sim >= 0.7) return 'blue';
    if (sim >= 0.5) return 'orange';
    return 'red';
  };

  return (
    <div>
      <Card title="向量检索" style={{ marginBottom: 16 }}>
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Space size="middle" wrap>
            <TextArea
              placeholder="输入检索文本（中文），例如：城市夜景航拍照片，摩天大楼灯光璀璨"
              value={queryText}
              onChange={(e) => setQueryText(e.target.value)}
              onPressEnter={handleTextSearch}
              style={{ width: 400 }}
              rows={2}
              allowClear
            />
            <Space direction="vertical" size={4}>
              <Text type="secondary">检索模式</Text>
              <Radio.Group
                value={searchMode}
                onChange={(e) => setSearchMode(e.target.value)}
                optionType="button"
                buttonStyle="solid"
                size="small"
              >
                <Tooltip title="语义检索: semantic_embedding 主检索 + 0.7/0.2/0.1 加权排序（推荐）">
                  <Radio.Button value="semantic">
                    <ThunderboltOutlined /> 语义检索
                  </Radio.Button>
                </Tooltip>
                <Tooltip title="文本检索: 传统 text_embedding 检索">
                  <Radio.Button value="text">文本检索</Radio.Button>
                </Tooltip>
              </Radio.Group>
            </Space>
            <Space direction="vertical" size={4}>
              <Text type="secondary">分类筛选</Text>
              <Select
                style={{ width: 160 }}
                placeholder="全部分类"
                allowClear
                showSearch
                value={category}
                onChange={setCategory}
                filterOption={(input, option) =>
                  (option?.label as string)?.includes(input) ?? false
                }
                options={[
                  ...categories.map(c => ({ value: c.value, label: c.label })),
                ]}
              />
            </Space>
            <Space direction="vertical" size={4}>
              <Text type="secondary">Top-K: {topK}</Text>
              <Slider
                style={{ width: 150 }}
                min={1} max={50}
                value={topK}
                onChange={setTopK}
              />
            </Space>
            <Space>
              <Button type="primary" icon={<SearchOutlined />}
                loading={searching} onClick={handleTextSearch}>
                文本检索
              </Button>
              <Upload
                accept="image/*"
                showUploadList={false}
                beforeUpload={(file) => { handleImageSearch(file); return false; }}
              >
                <Button icon={<CameraOutlined />} loading={searching}>
                  以图搜图
                </Button>
              </Upload>
            </Space>
          </Space>
        </Space>
      </Card>

      {/* 语义检索结果 (v5) */}
      {semanticResults.length > 0 && (
        <Card
          title={
            <Space>
              <ThunderboltOutlined style={{ color: '#722ed1' }} />
              <span>语义检索结果</span>
              <Tag color="purple">{totalInPartition} 条分区内记录</Tag>
              <Tag>{queryTime.toFixed(0)}ms</Tag>
              <Tag color="blue">加权排序: 0.7×语义 + 0.2×图像 + 0.1×标签</Tag>
            </Space>
          }
          style={{ marginBottom: 16 }}
        >
          <List
            dataSource={semanticResults}
            renderItem={(item, index) => (
              <List.Item
                extra={
                  <Image
                    src={`/images/${item.image_path?.split(/[/\\]/).pop() || ''}`}
                    alt={`#${index + 1} ID: ${item.image_id}`}
                    width={160}
                    style={{ borderRadius: 8, objectFit: 'cover' }}
                    fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                  />
                }
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <Text strong>#{index + 1}</Text>
                      <Text>ID: {item.image_id}</Text>
                      <Tag color="purple">
                        最终分: {(item.final_score * 100).toFixed(1)}%
                      </Tag>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" size={4}>
                      {/* 加权得分明细 */}
                      <Space size={[0, 4]} wrap>
                        <Tag color="blue">语义: {(item.semantic_similarity * 100).toFixed(1)}%</Tag>
                        <Tag color="green">图像: {(item.image_similarity * 100).toFixed(1)}%</Tag>
                        <Tag color="orange">标签: {(item.tags_overlap * 100).toFixed(1)}%</Tag>
                      </Space>
                      {/* 分类/类型字段 */}
                      <Space size={[0, 4]} wrap>
                        {item.subject && <Tag color="purple">{item.subject}</Tag>}
                        {item.category && !item.subject && <Tag color="purple">{item.category}</Tag>}
                        {item.topic && <Tag color="geekblue">{item.topic}</Tag>}
                        {item.content_type && <Tag>{item.content_type}</Tag>}
                        {item.diagram_type && !item.content_type && <Tag>{item.diagram_type}</Tag>}
                        <Tag color={item.source_type === 'uploaded' ? 'cyan' : 'default'}>
                          {item.source_type === 'uploaded' ? '素材' : 'AI生成'}
                        </Tag>
                      </Space>
                      {/* 关键词/标签 */}
                      {(item.keywords && item.keywords.length > 0) && (
                        <Space size={[0, 4]} wrap>
                          {item.keywords.map(kw => (
                            <Tag key={kw} color="lime">{kw}</Tag>
                          ))}
                        </Space>
                      )}
                      {(!item.keywords || item.keywords.length === 0) && item.knowledge_points && item.knowledge_points.length > 0 && (
                        <Space size={[0, 4]} wrap>
                          {item.knowledge_points.map(kp => (
                            <Tag key={kp} color="lime">{kp}</Tag>
                          ))}
                        </Space>
                      )}
                      {/* prompt 摘要 */}
                      <Text type="secondary" style={{ fontSize: 12 }} ellipsis>
                        {item.prompt}
                      </Text>
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* 检索结果 */}
      {results.length > 0 && (
        <Card
          title={
            <Space>
              <ExperimentOutlined />
              <span>检索结果</span>
              <Tag color="blue">{totalInPartition} 条分区内记录</Tag>
              <Tag>{queryTime.toFixed(0)}ms</Tag>
            </Space>
          }
          style={{ marginBottom: 16 }}
        >
          <List
            dataSource={results}
            renderItem={(item, index) => (
              <List.Item
                extra={
                  <Image
                    src={`/images/${item.image_path?.split(/[/\\]/).pop() || ''}`}
                    alt={`#${index + 1} ID: ${item.image_id}`}
                    width={160}
                    style={{ borderRadius: 8, objectFit: 'cover' }}
                    fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
                  />
                }
              >
                <List.Item.Meta
                  title={
                    <Space>
                      <Text strong>#{index + 1}</Text>
                      <Text>ID: {item.image_id}</Text>
                    </Space>
                  }
                  description={
                    <Space direction="vertical" size={4}>
                      <Space size={[0, 4]} wrap>
                        <Tag color={similarityColor(item.similarity)}>
                          相似度: {(item.similarity * 100).toFixed(1)}%
                        </Tag>
                        <Tag color="green">评分: {item.score?.toFixed(2)}</Tag>
                        {item.subject && <Tag color="purple">{item.subject}</Tag>}
                        {item.category && !item.subject && <Tag color="purple">{item.category}</Tag>}
                      </Space>
                      {item.tags && item.tags.length > 0 && (
                        <Space size={[0, 4]} wrap>
                          {item.tags.map(t => <Tag key={t} color="blue">{t}</Tag>)}
                        </Space>
                      )}
                    </Space>
                  }
                />
              </List.Item>
            )}
          />
        </Card>
      )}

      {/* 检索历史 */}
      <Card
        title={<Space><HistoryOutlined /><span>检索历史</span></Space>}
        style={{ width: '100%' }}
      >
        {history.length === 0 ? (
          <Text type="secondary">暂无检索历史</Text>
        ) : (
          <List
            size="small"
            dataSource={history}
            renderItem={(item) => (
              <List.Item>
                <Space>
                  <Tag color={item.mode === 'text' ? 'blue' : 'green'}>
                    {item.mode === 'text' ? '文本' : '图片'}
                  </Tag>
                  <Text
                    style={{ cursor: 'pointer', maxWidth: 300 }}
                    ellipsis
                    onClick={() => setQueryText(item.query)}
                  >
                    {item.query}
                  </Text>
                  {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
                  {(item as any).subject ? <Tag color="purple">{(item as any).subject as string}</Tag> : null}
                  <Tag>{item.result_count} 条结果</Tag>
                  <Text type="secondary">{item.timestamp}</Text>
                </Space>
              </List.Item>
            )}
          />
        )}
      </Card>
    </div>
  );
};

export default Search;
