import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Layout, Menu, Typography } from 'antd';
import {
  PartitionOutlined,
  SearchOutlined,
  PictureOutlined,
  CloudUploadOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import Partitions from './pages/Partitions';
import Search from './pages/Search';
import Gallery from './pages/Gallery';
import MaterialUpload from './pages/MaterialUpload';
import PictureGenerator from './pages/PictureGenerator';
import { useWebSocket } from './hooks/useWebSocket';

const { Header, Sider, Content } = Layout;

const menuItems = [
  { key: '/generate', icon: <ThunderboltOutlined />, label: '获取图片' },
  { key: '/partitions', icon: <PartitionOutlined />, label: '分区管理' },
  { key: '/upload', icon: <CloudUploadOutlined />, label: '素材入库' },
  { key: '/gallery', icon: <PictureOutlined />, label: '照片浏览' },
  { key: '/search', icon: <SearchOutlined />, label: '向量检索' },
];

const AppLayout: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // 全局 WebSocket 连接
  useWebSocket();

  const selectedKey = menuItems
    .map((item) => item.key)
    .filter((key) => location.pathname.startsWith(key))
    .sort((a, b) => b.length - a.length)[0] || '/generate';

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header style={{ display: 'flex', alignItems: 'center', padding: '0 24px' }}>
        <Typography.Title level={4} style={{ color: '#fff', margin: 0 }}>
          🎨 Picture Milvus Admin
        </Typography.Title>
        <Typography.Text style={{ color: 'rgba(255,255,255,0.65)', marginLeft: 12 }}>
          AI 图片生成与向量检索平台
        </Typography.Text>
      </Header>
      <Layout>
        <Sider width={220} style={{ background: '#fff' }}>
          <Menu
            mode="inline"
            selectedKeys={[selectedKey]}
            items={menuItems}
            onClick={({ key }) => navigate(key)}
            style={{ height: '100%', borderRight: 0 }}
          />
        </Sider>
        <Content style={{ padding: 24, background: '#f5f5f5' }}>
          <Routes>
            <Route path="/generate" element={<PictureGenerator />} />
            <Route path="/partitions" element={<Partitions />} />
            <Route path="/upload" element={<MaterialUpload />} />
            <Route path="/gallery" element={<Gallery />} />
            <Route path="/search" element={<Search />} />
            <Route path="*" element={<Navigate to="/generate" replace />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
};

const App: React.FC = () => (
  <BrowserRouter>
    <AppLayout />
  </BrowserRouter>
);

export default App;
