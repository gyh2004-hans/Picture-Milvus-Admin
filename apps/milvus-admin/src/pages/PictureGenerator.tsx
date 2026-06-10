import React, { useState, useRef } from 'react';
import {
  Card, Input, Button, Space, Tag, Typography, Image, message, Spin, Empty, Row, Col,
  Divider, Tooltip,
} from 'antd';
import {
  SendOutlined, DownloadOutlined, PictureOutlined,
  CloseCircleOutlined, LoadingOutlined, CheckCircleOutlined,
  DatabaseOutlined, SaveOutlined, ExclamationCircleOutlined,
} from '@ant-design/icons';
import { runPipeline, PipelineResponse } from '../api/milvusClient';

const { TextArea } = Input;
const { Text } = Typography;

// ═══════════════════════════════════════════════════
// 主组件
// ═══════════════════════════════════════════════════

const PictureGenerator: React.FC = () => {
  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const resultRef = useRef<HTMLDivElement>(null);

  // ── 执行获取图片 ──────────────────────────────

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      message.warning('请输入图像描述');
      return;
    }

    setGenerating(true);
    setError(null);
    setResult(null);

    try {
      const res = await runPipeline({
        prompt: prompt.trim(),
        mode: 'clip_enrich',
        model: 'tongyi',
        max_iterations: 3,
        eval_threshold: 0.82,
      });
      setResult(res);

      // 滚动到结果
      setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : '未知错误';
      setError(msg);
      message.error(`获取图片失败: ${msg}`);
    } finally {
      setGenerating(false);
    }
  };

  // ── 下载图片 ──────────────────────────────────

  const handleDownload = (base64: string, filename: string) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${base64}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const isReused = result?.stopped_reason === 'reused';

  // ═══════════════════════════════════════════════════
  // 渲染
  // ═══════════════════════════════════════════════════

  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      {/* ── 输入区 ──────────────────────────────── */}
      <Card
        title={
          <Space>
            <PictureOutlined style={{ color: '#1890ff' }} />
            <span>获取图片</span>
          </Space>
        }
        style={{ marginBottom: 16 }}
      >
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          {/* Prompt 输入 */}
          <div>
            <Text strong>图像描述（自然语言）</Text>
            <TextArea
              placeholder="用自然语言描述你想要的图片，例如：城市夜景航拍照片，摩天大楼灯光璀璨，长曝光效果..."
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              rows={4}
              style={{ marginTop: 8 }}
              maxLength={2000}
              showCount
            />
          </div>

          {/* 获取按钮 */}
          <Button
            type="primary"
            size="large"
            icon={generating ? <LoadingOutlined /> : <SendOutlined />}
            loading={generating}
            onClick={handleGenerate}
            block
            style={{ height: 48, fontSize: 16 }}
          >
            {generating ? '正在处理...' : '获取图片'}
          </Button>
        </Space>
      </Card>

      {/* ── 错误提示 ────────────────────────────── */}
      {error && (
        <Card style={{ marginBottom: 16, borderColor: '#ff4d4f' }}>
          <Space>
            <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
            <Text type="danger">{error}</Text>
          </Space>
        </Card>
      )}

      {/* ── 加载状态 ────────────────────────────── */}
      {generating && (
        <Card style={{ marginBottom: 16 }}>
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Spin size="large" />
            <div style={{ marginTop: 16 }}>
              <Text type="secondary">
                正在检索相似图片并生成，请耐心等待...
              </Text>
            </div>
          </div>
        </Card>
      )}

      {/* ── 图片输出区域 ────────────────────────── */}
      <div ref={resultRef}>
        {result && (
          <Card
            title={
              <Space>
                <PictureOutlined style={{ color: '#52c41a' }} />
                <span>图片输出</span>
                {isReused ? (
                  <Tag color="purple">🎯 复用命中</Tag>
                ) : (
                  <Tag color="blue">AI 生成</Tag>
                )}
              </Space>
            }
            style={{ marginBottom: 16 }}
          >
            <Row gutter={[16, 16]}>
              {/* 图片展示 */}
              <Col xs={24} md={16}>
                <div style={{ position: 'relative' }}>
                  {result.final_image_base64 ? (
                    <Image
                      src={`data:image/png;base64,${result.final_image_base64}`}
                      alt="获取的图片"
                      style={{ width: '100%', borderRadius: 8 }}
                      preview={{ mask: '点击预览' }}
                    />
                  ) : result.final_image_path ? (
                    <Image
                      src={`/images/${result.final_image_path.split(/[/\\]/).pop() || ''}`}
                      alt="获取的图片"
                      style={{ width: '100%', borderRadius: 8 }}
                      preview={{ mask: '点击预览' }}
                    />
                  ) : null}

                  {/* 下载按钮 */}
                  <Button
                    type="primary"
                    icon={<DownloadOutlined />}
                    style={{ position: 'absolute', top: 12, right: 12 }}
                    onClick={() => {
                      if (result.final_image_base64) {
                        handleDownload(result.final_image_base64, `image_${Date.now()}.png`);
                      } else if (result.final_image_path) {
                        const filename = result.final_image_path.split(/[/\\]/).pop() || 'image.png';
                        const link = document.createElement('a');
                        link.href = `/images/${filename}`;
                        link.download = filename;
                        document.body.appendChild(link);
                        link.click();
                        document.body.removeChild(link);
                      }
                    }}
                  >
                    下载图片
                  </Button>
                </div>
              </Col>

              {/* 信息栏 */}
              <Col xs={24} md={8}>
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                  {/* 生成/复用说明 */}
                  <Text type="secondary">
                    {isReused
                      ? '该图片来自语义检索命中结果，已复用已有图片'
                      : '该图片由 AI 模型生成'}
                  </Text>
                  {result.final_score != null && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      最终评分: {(result.final_score * 100).toFixed(1)}%
                    </Text>
                  )}
                  {result.total_iterations != null && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      迭代次数: {result.total_iterations}
                    </Text>
                  )}
                  {result.reused_from_record_id != null && (
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      复用来源 ID: {result.reused_from_record_id}
                    </Text>
                  )}

                  <Divider style={{ margin: '8px 0' }} />

                  {/* ── 数据库存储状态 ── */}
                  <Text strong style={{ fontSize: 12 }}><DatabaseOutlined /> 存储状态</Text>
                  {result.stored_in_milvus ? (
                    <Tag color="green" icon={<CheckCircleOutlined />}>
                      Milvus 向量库 — 已存入
                    </Tag>
                  ) : (
                    <Tag color="orange" icon={<ExclamationCircleOutlined />}>
                      Milvus 向量库 — 未存储
                    </Tag>
                  )}
                  {result.stored_in_records ? (
                    <Tag color="blue" icon={<SaveOutlined />}>
                      本地记录 — 已存入
                    </Tag>
                  ) : (
                    <Tag color="default" icon={<ExclamationCircleOutlined />}>
                      本地记录 — 未存储
                    </Tag>
                  )}

                  {result.db_record_id != null && (
                    <Text type="secondary" style={{ fontSize: 11 }}>
                      Milvus 记录 ID: {result.db_record_id}
                    </Text>
                  )}

                  {/* 存储的 Prompt 摘要 */}
                  <Divider style={{ margin: '4px 0' }} />
                  <Text type="secondary" style={{ fontSize: 11 }}>已存入的 Prompt:</Text>
                  <Tooltip title={result.final_prompt}>
                    <Text
                      style={{
                        fontSize: 11,
                        background: '#f5f5f5',
                        padding: '4px 8px',
                        borderRadius: 4,
                        maxHeight: 80,
                        overflow: 'hidden',
                        display: 'block',
                        lineHeight: 1.5,
                      }}
                    >
                      {result.final_prompt.length > 120
                        ? result.final_prompt.slice(0, 120) + '…'
                        : result.final_prompt}
                    </Text>
                  </Tooltip>
                </Space>
              </Col>
            </Row>
          </Card>
        )}
      </div>

      {/* ── 空状态 ──────────────────────────────── */}
      {!result && !generating && !error && (
        <Card style={{ marginBottom: 16 }}>
          <Empty
            image={<PictureOutlined style={{ fontSize: 64, color: '#d9d9d9' }} />}
            description={
              <Text type="secondary">
                输入图像描述，获取图片 —— 支持 AI 生成与智能检索复用
              </Text>
            }
          />
        </Card>
      )}
    </div>
  );
};

export default PictureGenerator;
