import { Layout, Input, Avatar, Badge, Space } from 'antd';
import { SearchOutlined, BellOutlined, UserOutlined } from '@ant-design/icons';

const { Header } = Layout;

export default function TopBar() {
  return (
    <Header
      style={{
        background: '#FFFFFF',
        borderBottom: '1px solid #E5E4E7',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 24px',
      }}
    >
      <Input
        placeholder="Search strategies, models, wallets..."
        prefix={<SearchOutlined style={{ color: '#6B7280' }} />}
        style={{ maxWidth: 360, borderRadius: 8 }}
      />
      <Space size={20}>
        <Badge dot>
          <BellOutlined style={{ fontSize: 18, color: '#1D2129' }} />
        </Badge>
        <Avatar icon={<UserOutlined />} style={{ backgroundColor: '#6C5CE7' }} />
      </Space>
    </Header>
  );
}