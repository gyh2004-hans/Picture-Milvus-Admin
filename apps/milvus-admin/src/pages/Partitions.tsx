import React, { useEffect, useState } from 'react';
import {
  Table, Card, Space, Tag, Statistic, Row, Col, Spin, message,
} from 'antd';
import type { ColumnsType } from 'antd/es/table';
import {
  listPartitions, getCollectionStats, getCategories, PartitionInfo,
} from '../api/milvusClient';

/** 15 个固定分类标签（不可由用户改动） */
const FIXED_CATEGORY_LABELS: string[] = [
  '自然风光',
  '人物人像',
  '城市建筑',
  '美食饮品',
  '动植物',
  '办公商务',
  '数码科技',
  '服饰穿搭',
  '家居家装',
  '节日庆典',
  '手绘插画',
  '纹理背景',
  '交通出行',
  '教育培训',
  '运动休闲',
];

const COLLECTION_NAME = 'image_embeddings';

interface PartitionRow {
  key: string;
  label: string;
  partition_name: string;
  row_count: number;
  /** 是否在后端实际存在该分区 */
  exists_in_backend: boolean;
}

const Partitions: React.FC = () => {
  const [data, setData] = useState<PartitionRow[]>([]);
  const [loading, setLoading] = useState(false);

  const fetchData = async () => {
    setLoading(true);
    try {
      // 并行请求：分区列表 + 分区统计 + 后端分类标签
      const [partitions, stats, backendCategories] = await Promise.all([
        listPartitions(COLLECTION_NAME),
        getCollectionStats(COLLECTION_NAME),
        getCategories().catch(() => ({ categories: [] })),
      ]);

      // 构建后端已有分区名集合
      const existingNames = new Set(partitions.map((p: PartitionInfo) => p.name));
      // 后端分类标签集合
      const backendLabelSet = new Set(
        backendCategories.categories?.map((c: { value: string }) => c.value) || [],
      );

      // 以固定标签为基准，合并后端统计数据
      const rows: PartitionRow[] = FIXED_CATEGORY_LABELS.map((label) => {
        // 尝试匹配后端分区名（直接匹配标签名）
        const rowCount = stats[label] || 0;
        const exists = existingNames.has(label);
        // 也检查后端分类接口是否包含该标签
        const synced = backendLabelSet.has(label);
        return {
          key: label,
          label,
          partition_name: label,
          row_count: rowCount,
          exists_in_backend: exists || synced,
        };
      });

      setData(rows);
    } catch (e: unknown) {
      message.error(`加载分区数据失败: ${e instanceof Error ? e.message : '未知错误'}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const columns: ColumnsType<PartitionRow> = [
    {
      title: '序号',
      key: 'index',
      width: 60,
      render: (_, __, idx) => idx + 1,
    },
    {
      title: '分区标签',
      dataIndex: 'label',
      key: 'label',
      render: (label: string) => (
        <Space>
          <strong>{label}</strong>
          <Tag color="purple">固定分类</Tag>
        </Space>
      ),
    },
    {
      title: '分区名',
      dataIndex: 'partition_name',
      key: 'partition_name',
      render: (name: string) => <code>{name}</code>,
    },
    {
      title: '实体数',
      dataIndex: 'row_count',
      key: 'row_count',
      render: (v: number) => (
        <Tag color={v > 0 ? 'blue' : 'default'}>{v.toLocaleString()}</Tag>
      ),
    },
    {
      title: '同步状态',
      dataIndex: 'exists_in_backend',
      key: 'exists_in_backend',
      render: (exists: boolean) =>
        exists ? (
          <Tag color="green">● 已同步</Tag>
        ) : (
          <Tag color="orange">● 待初始化</Tag>
        ),
    },
  ];

  const totalEntities = data.reduce((sum, r) => sum + r.row_count, 0);
  const syncedCount = data.filter((r) => r.exists_in_backend).length;
  const hasData = data.filter((r) => r.row_count > 0).length;

  return (
    <div>
      <Card
        title="分区管理"
        style={{ marginBottom: 16 }}
        extra={
          <Tag color="blue" style={{ marginRight: 8 }}>
            共 {FIXED_CATEGORY_LABELS.length} 个固定分区
          </Tag>
        }
      >
        <Row gutter={16} style={{ marginBottom: 16 }}>
          <Col span={6}>
            <Statistic title="分区总数" value={FIXED_CATEGORY_LABELS.length} prefix={<span>📂</span>} />
          </Col>
          <Col span={6}>
            <Statistic title="总实体数" value={totalEntities} />
          </Col>
          <Col span={6}>
            <Statistic
              title="已同步分区"
              value={syncedCount}
              suffix={`/ ${FIXED_CATEGORY_LABELS.length}`}
              valueStyle={{ color: syncedCount === FIXED_CATEGORY_LABELS.length ? '#3f8600' : '#cf1322' }}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="有数据分区"
              value={hasData}
              valueStyle={{ color: hasData > 0 ? '#3f8600' : '#cf1322' }}
            />
          </Col>
        </Row>
      </Card>

      <Card>
        <Spin spinning={loading}>
          <Table
            columns={columns}
            dataSource={data}
            rowKey="key"
            pagination={false}
            locale={{ emptyText: '暂无分区数据，请确认后端服务已启动' }}
          />
        </Spin>
      </Card>
    </div>
  );
};

export default Partitions;
