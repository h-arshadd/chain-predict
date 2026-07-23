import { Input, Avatar, Badge } from 'antd';
import { SearchOutlined, BellOutlined, UserOutlined, DownOutlined } from '@ant-design/icons';

export default function TopBar() {
  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long', month: 'short', day: 'numeric', year: 'numeric',
  });

  return (
    <header
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '20px 32px',
        gap: 24,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 24, flex: 1 }}>
        <span style={{ color: '#9096A0', fontSize: 14.5, fontWeight: 500, whiteSpace: 'nowrap' }}>
          {today}
        </span>
        <Input
          placeholder="Search symbol or any stock"
          prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
          style={{
            maxWidth: 340,
            borderRadius: 999,
            background: 'rgba(255,255,255,0.04)',
            border: '1px solid rgba(255,255,255,0.07)',
            padding: '9px 16px',
          }}
        />
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
        <Badge dot color="#3DDC97" offset={[-2, 2]}>
          <div
            style={{
              width: 38, height: 38, borderRadius: 12,
              background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}
          >
            <BellOutlined style={{ fontSize: 16, color: '#F5F6F7' }} />
          </div>
        </Badge>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer' }}>
          <Avatar size={36} icon={<UserOutlined />} style={{ backgroundColor: '#3DDC97', color: '#0B0E11' }} />
          <span style={{ color: '#F5F6F7', fontSize: 14.5, fontWeight: 600 }}>Account</span>
          <DownOutlined style={{ fontSize: 11, color: '#6B7280' }} />
        </div>
      </div>
    </header>
  );
}