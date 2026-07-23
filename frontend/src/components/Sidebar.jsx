import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu } from 'antd';
import {
  DashboardOutlined,
  LineChartOutlined,
  WalletOutlined,
  RocketOutlined,
  ExperimentOutlined,
  FundOutlined,
} from '@ant-design/icons';

const { Sider } = Layout;

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: 'Dashboard' },
  { key: '/strategies', icon: <LineChartOutlined />, label: 'Strategies' },
  { key: '/wallets', icon: <WalletOutlined />, label: 'Wallets' },
  { key: '/deployment', icon: <RocketOutlined />, label: 'Deployment' },
  { key: '/backtests', icon: <ExperimentOutlined />, label: 'Backtests' },
  { key: '/models', icon: <FundOutlined />, label: 'ML Models' },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <Sider
      collapsible
      collapsed={collapsed}
      onCollapse={setCollapsed}
      width={240}
      style={{ borderRight: '1px solid #E5E4E7' }}
    >
      <div
        style={{
          height: 56,
          margin: 16,
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'flex-start',
          fontWeight: 600,
          fontSize: collapsed ? 18 : 20,
          color: '#6C5CE7',
        }}
      >
        {collapsed ? 'CP' : 'ChainPredict'}
      </div>
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={({ key }) => navigate(key)}
        style={{ borderRight: 'none' }}
      />
    </Sider>
  );
}