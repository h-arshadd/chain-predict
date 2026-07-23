import { Outlet } from 'react-router-dom';
import { Layout } from 'antd';
import Sidebar from '../components/Sidebar';
import TopBar from '../components/TopBar';

const { Content } = Layout;

export default function DashboardLayout() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Sidebar />
      <Layout>
        <TopBar />
        <Content style={{ margin: '24px', minHeight: 280 }}>
          <Outlet />
        </Content>
      </Layout>
    </Layout>
  );
}