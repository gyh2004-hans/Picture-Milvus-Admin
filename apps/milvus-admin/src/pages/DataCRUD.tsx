import React, { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import {
  Table, Button, Card, Space, Modal, Input, InputNumber, Select,
  Popconfirm, message, Tag, Slider,
} from 'antd';
import { ReloadOutlined, DeleteOutlined, EditOutlined, FilterOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import {
  listData, deleteEntity, updateEntity, getCollectionStats, EntityRecord, getCategories, CategoryOption,
} from '../api/milvusClient';

const DataCRUD: React.FC = () => {
  const { name = 'image_embeddings' } = useParams<{ name: string }>();
  const [data, setData] = useState<EntityRecord[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [categoryFilter, setCategoryFilter] = useState('');
  const [minScore, setMinScore] = useState(0);
  const [editModalOpen, setEditModalOpen] = useState(false);
  const [editRecord, setEditRecord] = useState<EntityRecord | null>(null);
  const [editPrompt, setEditPrompt] = useState('');
  const [categories, setCategories] = useState<CategoryOption[]>([]);

  // 动态加载分类列表
  useEffect(() => {
    getCategories()
      .then((r) => setCategories(r.categories || []))
      .catch(() => setCategories([]));
  }, []);

  const categoryOptions = [
    { value: '', label: '全部分类' },
    ...categories.map((c) => ({ value: c.value, label: c.label })),
  ];

  const fetchData = async () => {
    setLoading(true);
    try {
      const params: Record<string, unknown> = { limit: 50 };
      if (categoryFilter) params.category = categoryFilter;
      if (minScore > 0) params.min_score = minScore;
      const result = await listData(name, params);
      setData(result.data);
      setTotal(result.total);
    } catch (e: unknown) {
      message.error(`加载失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [name, categoryFilter, minScore]);

  const handleDelete = async (id: number) => {
    try {
      await deleteEntity(name, id);
      message.success(`记录 ${id} 已删除`);
      fetchData();
    } catch (e: unknown) {
      message.error(`删除失败: ${e instanceof Error ? e.message : '未知错误'}`);
    }
  };

  const handleEdit = (record: EntityRecord) => {
    setEditRecord(record);
    setEditPrompt(record.optimized_prompt || record.prompt);
    setEditModalOpen(true);
  };

  const handleSaveEdit = async () => {
    if (!editRecord) return;
    try {
      await updateEntity(name, editRecord.id, {
        optimized_prompt: editPrompt,
      });
      message.success(`记录 ${editRecord.id} 已更新`);
      setEditModalOpen(false);
      fetchData();
    } catch (e: unknown) {
      message.error(`更新失败: ${e instanceof Error ? e.message : '未知错误'}`);
    }
  };

  const columns: ColumnsType<EntityRecord> = [
    {
      title: 'ID',
      dataIndex: 'id',
      key: 'id',
      width: 80,
    },
    {
      title: 'Prompt',
      dataIndex: 'prompt',
      key: 'prompt',
      ellipsis: true,
      width: 250,
      render: (v: string) => (
        <span title={v}>{v.length > 60 ? v.slice(0, 60) + '...' : v}</span>
      ),
    },
    {
      title: 'Score',
      dataIndex: 'score',
      key: 'score',
      width: 80,
      render: (v: number) => {
        const color = v >= 0.82 ? 'green' : v >= 0.7 ? 'orange' : 'red';
        return <Tag color={color}>{v.toFixed(2)}</Tag>;
      },
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      width: 120,
      render: (v?: string, record?: EntityRecord) => {
        const display = v || record?.subject;
        return display ? <Tag color="purple">{display}</Tag> : <Tag>未标注</Tag>;
      },
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      width: 200,
      render: (tags: string[]) => (
        <Space size={[0, 4]} wrap>
          {tags?.slice(0, 3).map((t) => <Tag key={t} color="blue">{t}</Tag>)}
          {tags?.length > 3 && <Tag>+{tags.length - 3}</Tag>}
        </Space>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 150,
      render: (_, record) => (
        <Space>
          <Button size="small" icon={<EditOutlined />}
            onClick={() => handleEdit(record)}>编辑</Button>
          <Popconfirm
            title="确认删除此记录？"
            onConfirm={() => handleDelete(record.id)}
            okText="确认"
            cancelText="取消"
          >
            <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <Card title={`数据管理 — ${name}`} style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={fetchData}>刷新</Button>
          </Space>
        }>
        <Space style={{ marginBottom: 16 }} size="middle" wrap>
          <span><FilterOutlined /> 分类筛选:</span>
          <Select
            style={{ width: 120 }}
            value={categoryFilter}
            onChange={setCategoryFilter}
            options={categoryOptions}
            showSearch
            filterOption={(input, option) =>
              (option?.label as string)?.includes(input) ?? false
            }
          />
          <span>最低评分:</span>
          <Slider
            style={{ width: 200 }}
            min={0} max={1} step={0.05}
            value={minScore}
            onChange={setMinScore}
          />
          <Tag>{total} 条记录</Tag>
        </Space>
      </Card>

      <Card>
        <Table
          columns={columns}
          dataSource={data}
          rowKey="id"
          loading={loading}
          pagination={{ pageSize: 20, showTotal: (t) => `共 ${t} 条` }}
        />
      </Card>

      <Modal
        title="编辑 Prompt"
        open={editModalOpen}
        onOk={handleSaveEdit}
        onCancel={() => setEditModalOpen(false)}
        okText="保存"
        cancelText="取消"
        width={600}
      >
        <Input.TextArea
          rows={6}
          value={editPrompt}
          onChange={(e) => setEditPrompt(e.target.value)}
          placeholder="编辑 optimized_prompt..."
        />
        {editRecord && (
          <div style={{ marginTop: 12, color: '#888' }}>
            <Tag>ID: {editRecord.id}</Tag>
            <Tag>Score: {editRecord.score?.toFixed(2)}</Tag>
            <Tag>分类: {editRecord.category || editRecord.subject || '未标注'}</Tag>
          </div>
        )}
      </Modal>
    </div>
  );
};

export default DataCRUD;
