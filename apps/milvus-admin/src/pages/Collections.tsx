import React, { useEffect, useState } from 'react';
import { Table, Button, Card, Space, Statistic, Row, Col, Popconfirm, message, Tag, Alert } from 'antd';
import { PlusOutlined, ReloadOutlined, DeleteOutlined, EyeOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { listCollections, deleteCollection, CollectionInfo, getMilvusStatus, MilvusStatus } from '../api/milvusClient';

const BACKEND_LABELS: Record<string, string> = {
  milvus_lite: 'Milvus Lite (本地文件)',
  milvus_server: 'Milvus Server (远程)',
  local_numpy: 'Numpy 内存 (⚠️ 重启丢失)',
};

const Collections: React.FC = () => {
  const [data, setData] = useState<CollectionInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<MilvusStatus | null>(null);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [result, st] = await Promise.all([
        listCollections(),
        getMilvusStatus(),
      ]);
      setData(result);
      setStatus(st);
    } catch (e: unknown) {
      message.error(`加载失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const handleDelete = async (name: string) => {
    try {
      await deleteCollection(name);
      message.success(`集合 ${name} 已删除`);
      fetchData();
    } catch (e: unknown) {
      message.error(`删除失败: ${e instanceof Error ? e.message : '未知错误'}`);
    }
  };

  const columns: ColumnsType<CollectionInfo> = [
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => <strong>{name}</strong>,
    },
    {
      title: '实体数',
      dataIndex: 'entity_count',
      key: 'entity_count',
      render: (v: number) => <Tag color={v > 0 ? 'blue' : 'default'}>{v.toLocaleString()}</Tag>,
    },
    {
      title: '后端类型',
      dataIndex: 'backend_type',
      key: 'backend_type',
      render: (v: string) => {
        const color = v === 'local_numpy' ? 'orange' : v === 'milvus_lite' ? 'green' : 'blue';
        return <Tag color={color}>{BACKEND_LABELS[v] || v}</Tag>;
      },
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'Loaded' ? 'green' : 'red'}>● {status}</Tag>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button size="small" icon={<EyeOutlined />}
            onClick={() => message.info(`详情: ${record.name} - ${record.entity_count} 条记录 | 后端: ${BACKEND_LABELS[record.backend_type] || record.backend_type}`)}>
            查看
          </Button>
          <Popconfirm
            title={`确认删除集合 "${record.name}"？此操作不可恢复。`}
            onConfirm={() => handleDelete(record.name)}
            okText="确认删除"
            cancelText="取消"
          >
            <Button size="small" danger icon={<DeleteOutlined />}>删除</Button>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const totalEntities = data.reduce((sum, c) => sum + c.entity_count, 0);
  const backendType = data[0]?.backend_type || 'unknown';
  const isLocalNumpy = backendType === 'local_numpy';

  return (
    <div>
      {isLocalNumpy && (
        <Alert
          type="warning"
          showIcon
          message="内存存储模式 — 数据在服务重启后丢失"
          description="当前使用 LocalNumpyBackend（纯内存），Demo 写入的数据在进程退出时已丢失。如需持久化，请确保 pymilvus 正确安装并使用 Milvus Lite 后端。"
          style={{ marginBottom: 16 }}
          closable
        />
      )}

      <Card title="Collection 管理" style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={fetchData}>刷新</Button>
          </Space>
        }>
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic title="集合数" value={data.length} prefix={<span>📦</span>} />
          </Col>
          <Col span={6}>
            <Statistic title="总实体数" value={totalEntities} />
          </Col>
          <Col span={6}>
            <Statistic title="后端" value={BACKEND_LABELS[backendType] || backendType}
              valueStyle={{ color: isLocalNumpy ? '#cf1322' : '#3f8600' }} />
          </Col>
          <Col span={6}>
            <Statistic title="状态" value="已加载" valueStyle={{ color: '#3f8600' }} />
          </Col>
        </Row>
      </Card>

      {status?.warning && (
        <Alert type="warning" message={status.warning} style={{ marginBottom: 16 }} closable />
      )}

      <Card>
        <Table
          columns={columns}
          dataSource={data}
          rowKey="name"
          loading={loading}
          pagination={false}
        />
      </Card>
    </div>
  );
};

export default Collections;
