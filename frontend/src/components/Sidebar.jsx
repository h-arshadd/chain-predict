import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  LineChartOutlined,
  WalletOutlined,
  RocketOutlined,
  ExperimentOutlined,
  FundOutlined,
  SmileOutlined,
} from '@ant-design/icons';

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: 'Dashboard' },
  { key: '/strategies', icon: <LineChartOutlined />, label: 'Strategies' },
  { key: '/wallets', icon: <WalletOutlined />, label: 'Wallets' },
  { key: '/deployment', icon: <RocketOutlined />, label: 'Deployment' },
  { key: '/backtests', icon: <ExperimentOutlined />, label: 'Backtests' },
  { key: '/models', icon: <FundOutlined />, label: 'ML Models' },
  { key: '/sentiment', icon: <SmileOutlined />, label: 'Sentiment' },
];

export default function Sidebar() {
  const navigate = useNavigate();
  const location = useLocation();

  return (
    <aside
      style={{
        width: 264,
        flexShrink: 0,
        background: 'rgba(18, 22, 27, 0.7)',
        backdropFilter: 'blur(20px)',
        borderRight: '1px solid rgba(255,255,255,0.06)',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 16px',
        height: '100vh',
        position: 'sticky',
        top: 0,
      }}
    >
      {/* Logo */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '0 8px',
          marginBottom: 32,
        }}
      >
        <div
          style={{
            width: 30,
            height: 30,
            borderRadius: 9,
            background: 'radial-gradient(circle at 30% 30%, #3DDC97, #1F9E6B)',
            boxShadow: '0 0 16px rgba(61, 220, 151, 0.5)',
          }}
        />
        <span style={{ fontSize: 19, fontWeight: 700, color: '#F5F6F7', letterSpacing: -0.3 }}>
          ChainPredict
        </span>
      </div>

      {/* Nav */}
      <nav style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: 1 }}>
        {menuItems.map((item) => {
          const active = location.pathname === item.key;
          return (
            <button
              key={item.key}
              onClick={() => navigate(item.key)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                width: '100%',
                padding: '12px 16px',
                borderRadius: 14,
                border: 'none',
                cursor: 'pointer',
                fontSize: 14.5,
                fontWeight: 600,
                textAlign: 'left',
                transition: 'background 0.15s, color 0.15s',
                background: active ? '#3DDC97' : 'transparent',
                color: active ? '#0B0E11' : '#9096A0',
              }}
              onMouseEnter={(e) => {
                if (!active) e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
              }}
              onMouseLeave={(e) => {
                if (!active) e.currentTarget.style.background = 'transparent';
              }}
            >
              <span style={{ fontSize: 17, display: 'flex' }}>{item.icon}</span>
              {item.label}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}