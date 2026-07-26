import { Input } from 'antd';
import { SearchOutlined } from '@ant-design/icons';

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

    </header>
  );
}