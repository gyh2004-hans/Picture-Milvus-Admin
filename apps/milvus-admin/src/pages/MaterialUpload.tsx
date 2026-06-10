import React, { useState } from 'react';
import {
  Card, Upload, Button, Space, Typography, message, Descriptions,
  Tag, Spin, Divider, Image, Alert,
} from 'antd';
import {
  UploadOutlined, FileImageOutlined, ExperimentOutlined,
  CheckCircleOutlined, ThunderboltOutlined,
} from '@ant-design/icons';
import type { UploadFile } from 'antd';
import { uploadMaterial, MaterialUploadResponse } from '../api/milvusClient';

const { Title, Text, Paragraph } = Typography;

const CATEGORY_COLORS: Record<string, string> = {
  '风景': 'green', '人物': 'orange', '动物': 'lime', '科技': 'blue',
  '美食': 'red', '建筑': 'geekblue', '艺术': 'purple', '其他': 'default',
};

const MaterialUpload: React.FC = () => {
  const [fileList, setFileList] = useState<UploadFile[]>([]);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<MaterialUploadResponse | null>(null);

  const handleUpload = async () => {
    if (fileList.length === 0) {
      message.warning('请先选择要上传的图片');
      return;
    }
    const file = fileList[0].originFileObj;
    if (!file) {
      message.error('文件对象无效');
      return;
    }

    setUploading(true);
    setResult(null);
    try {
      const res = await uploadMaterial(file);
      setResult(res);
      message.success(`入库成功！record_id=${res.record_id}`);
      setFileList([]);
    } catch (e: unknown) {
      message.error(`上传失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setUploading(false);
    }
  };

  const categoryColor = result?.parse_result.category
    ? CATEGORY_COLORS[result.parse_result.category] || 'default'
    : 'default';

  return (
    <div>
      <Title level={4}>
        <FileImageOutlined /> 图片素材入库
      </Title>
      <Paragraph type="secondary">
        上传任意图片，自动 VLM 理解图片内容 → Chinese-CLIP 向量化 → Milvus 入库，
        后续可通过自然语言精准检索召回。
      </Paragraph>

      {/* ── 上传区域 ── */}
      <Card style={{ marginBottom: 16 }}>
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Upload
            accept="image/*"
            fileList={fileList}
            beforeUpload={(file) => {
              if (file.size > 10 * 1024 * 1024) {
                message.error('文件大小不能超过 10MB');
                return Upload.LIST_IGNORE;
              }
              setFileList([{
                uid: '-1',
                name: file.name,
                status: 'done',
                originFileObj: file as unknown as File,
              }]);
              return false;
            }}
            onRemove={() => setFileList([])}
            maxCount={1}
            listType="picture-card"
          >
            {fileList.length === 0 && (
              <div>
                <UploadOutlined style={{ fontSize: 24 }} />
                <div style={{ marginTop: 8 }}>选择图片</div>
              </div>
            )}
          </Upload>

          <Space>
            <Button
              type="primary"
              icon={<ThunderboltOutlined />}
              onClick={handleUpload}
              loading={uploading}
              disabled={fileList.length === 0}
              size="large"
            >
              开始入库
            </Button>
            {uploading && (
              <Text type="secondary">
                <Spin size="small" /> 正在进行 VLM 内容解析...
              </Text>
            )}
          </Space>

          <Alert
            type="info"
            showIcon
            message="上传流程"
            description="上传 → VLM 内容解析 → 构建 semantic_text → Chinese-CLIP 双向量编码（图像 + 语义） → Milvus 入库。整个过程约需 5-15 秒。"
            style={{ marginTop: 8 }}
          />
        </Space>
      </Card>

      {/* ── 入库结果 ── */}
      {result && (
        <Card
          title={
            <Space>
              <CheckCircleOutlined style={{ color: '#52c41a' }} />
              <span>入库成功</span>
              <Tag color="green">record_id: {result.record_id}</Tag>
            </Space>
          }
        >
          {/* VLM 解析结果 */}
          <Title level={5}>
            <ExperimentOutlined /> VLM 图片内容解析
          </Title>
          <Descriptions bordered size="small" column={2} style={{ marginBottom: 16 }}>
            <Descriptions.Item label="分类">
              <Tag color={categoryColor}>{result.parse_result.category || '未识别'}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="类型">
              <Tag>{result.parse_result.content_type || '未识别'}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="场景描述" span={2}>
              {result.parse_result.scene_description || '未识别'}
            </Descriptions.Item>
            <Descriptions.Item label="风格">
              <Tag color="blue">{result.parse_result.style || '未识别'}</Tag>
            </Descriptions.Item>
            <Descriptions.Item label="主体对象">
              <Space size={[0, 4]} wrap>
                {result.parse_result.main_objects.length > 0
                  ? result.parse_result.main_objects.map(obj => (
                      <Tag key={obj} color="purple">{obj}</Tag>
                    ))
                  : <Text type="secondary">未识别</Text>
                }
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="主色调" span={2}>
              <Space size={[0, 4]} wrap>
                {result.parse_result.color_palette.length > 0
                  ? result.parse_result.color_palette.map(c => (
                      <Tag key={c} color="cyan">{c}</Tag>
                    ))
                  : <Text type="secondary">未识别</Text>
                }
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="标签" span={2}>
              <Space size={[0, 4]} wrap>
                {result.parse_result.tags.length > 0
                  ? result.parse_result.tags.map(t => (
                      <Tag key={t} color="geekblue">{t}</Tag>
                    ))
                  : <Text type="secondary">未识别</Text>
                }
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="检索描述" span={2}>
              <Text strong style={{ color: '#1677ff' }}>
                {result.parse_result.retrieval_prompt}
              </Text>
              <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
                （未来用户搜索此描述即可召回该图片）
              </Text>
            </Descriptions.Item>
          </Descriptions>

          <Divider />

          {/* Semantic Text */}
          <Title level={5}>
            <FileImageOutlined /> Semantic Text（CLIP 编码输入）
          </Title>
          <Card size="small" style={{ background: '#f6f8fa', marginBottom: 16 }}>
            <pre style={{ margin: 0, fontSize: 13, lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>
              {result.semantic_text}
            </pre>
          </Card>

          {/* 图片预览 */}
          <Title level={5}>上传图片</Title>
          <Image
            src={`/images/${result.image_path?.split(/[/\\]/).pop() || ''}`}
            alt="上传的图片"
            style={{ maxHeight: 300, borderRadius: 8 }}
            fallback="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
          />
        </Card>
      )}
    </div>
  );
};

export default MaterialUpload;
