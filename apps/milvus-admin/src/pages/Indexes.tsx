import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Table, Button, Card, Space, Modal, Select, Popconfirm, message, Tag, Descriptions,
} from 'antd';
import { PlusOutlined, ReloadOutlined, DeleteOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { listIndexes, createIndex, dropIndex, IndexInfo } from '../api/milvusClient';

const INDEX_TYPES = [
  { value: 'FLAT', label: 'FLAT — 精确搜索（默认，< 10K 数据推荐）' },
  { value: 'HNSW', label: 'HNSW — 大规模数据推荐（10K~100K）' },
  { value: 'IVF_FLAT', label: 'IVF_FLAT — 对比测试基线' },
  { value: 'IVF_SQ8', label: 'IVF_SQ8 — 内存受限场景' },
];

const METRIC_TYPES = [
  { value: 'COSINE', label: 'COSINE — 余弦相似度（推荐）' },
  { value: 'L2', label: 'L2 — 欧氏距离' },
  { value: 'IP', label: 'IP — 内积' },
];

const Indexes: React.FC = () => {
  const { name = 'image_embeddings' } = useParams<{ name: string }>();
  const [data, setData] = useState<IndexInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [createModalOpen, setCreateModalOpen] = useState(false);
  const [newIndexType, setNewIndexType] = useState('HNSW');
  const [newMetricType, setNewMetricType] = useState('COSINE');

  const fetchData = async () => {
    setLoading(true);
    try {
      const result = await listIndexes(name);
      setData(result);
    } catch (e: unknown) {
      message.error(`加载失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [name]);

  const handleCreate = async () => {
    try {
      await createIndex(name, newIndexType, newMetricType);
      message.success(`索引 "${newIndexType}" (${newMetricType}) 已创建`);
      setCreateModalOpen(false);
      fetchData();
    } catch (e: unknown) {
      message.error(`创建失败: ${e instanceof Error ? e.message : '未知错误'}`);
    }
  };

  const handleDrop = async () => {
    try {
      await dropIndex(name);
      message.success('索引已删除');
      fetchData();
    } catch (e: unknown) {
      message.error(`删除失败: ${e instanceof Error ? e.message : '未知错误'}`);
    }
  };

  const columns: ColumnsType<IndexInfo> = [
    {
      title: '字段名',
      dataIndex: 'field_name',
      key: 'field_name',
      render: (v: string) => <code>{v}</code>,
    },
    {
      title: '索引类型',
      dataIndex: 'index_type',
      key: 'index_type',
      render: (v: string) => {
        const colors: Record<string, string> = { FLAT: 'default', HNSW: 'green', IVF_FLAT: 'blue', IVF_SQ8: 'orange' };
        return <Tag color={colors[v] || 'default'}>{v}</Tag>;
      },
    },
    {
      title: '度量类型',
      dataIndex: 'metric_type',
      key: 'metric_type',
      render: (v: string) => <Tag>{v}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (v: string) => <Tag color="green">● {v}</Tag>,
    },
    {
      title: '操作',
      key: 'actions',
      render: () => (
        <Popconfirm
          title="确认删除索引？删除后检索将回退到暴力搜索。"
          onConfirm={handleDrop}
          okText="确认删除"
          cancelText="取消"
        >
          <Button size="small" danger icon={<DeleteOutlined />}>删除索引</Button>
        </Popconfirm>
      ),
    },
  ];

  return (
    <div>
      <Card title={`索引管理 — ${name}`} style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button type="primary" icon={<PlusOutlined />}
              onClick={() => setCreateModalOpen(true)}>
              创建索引
            </Button>
            <Button icon={<ReloadOutlined />} onClick={fetchData}>刷新</Button>
          </Space>
        }>
        <Descriptions bordered size="small" column={3}>
          <Descriptions.Item label="索引演进路径">
            FLAT → HNSW（推荐）→ IVF_FLAT / IVF_SQ8（对比测试）
          </Descriptions.Item>
          <Descriptions.Item label="当前索引数">
            {data.length}
          </Descriptions.Item>
          <Descriptions.Item label="当前索引类型">
            {data[0]?.index_type || '无'}
          </Descriptions.Item>
        </Descriptions>
      </Card>

      <Card>
        <Table
          columns={columns}
          dataSource={data}
          rowKey="field_name"
          loading={loading}
          pagination={false}
        />
      </Card>

      <Modal
        title="创建索引"
        open={createModalOpen}
        onOk={handleCreate}
        onCancel={() => setCreateModalOpen(false)}
        okText="创建"
        cancelText="取消"
      >
        <div style={{ marginBottom: 16 }}>
          <label>索引类型</label>
          <Select
            style={{ width: '100%' }}
            value={newIndexType}
            onChange={setNewIndexType}
            options={INDEX_TYPES}
          />
        </div>
        <div style={{ marginBottom: 16 }}>
          <label>度量类型</label>
          <Select
            style={{ width: '100%' }}
            value={newMetricType}
            onChange={setNewMetricType}
            options={METRIC_TYPES}
          />
        </div>
      </Modal>
    </div>
  );
};

export default Indexes;
