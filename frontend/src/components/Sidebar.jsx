import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  WalletOutlined,
  RocketOutlined,
  ExperimentOutlined,
  FundOutlined,
  SmileOutlined,
  BuildOutlined,
} from '@ant-design/icons';
import Logo from './Logo';

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: 'Dashboard' },
  { key: '/wallets', icon: <WalletOutlined />, label: 'Wallets' },
  { key: '/execution', icon: <RocketOutlined />, label: 'Execution' },
  { key: '/backtests', icon: <ExperimentOutlined />, label: 'Backtests' },
  { key: '/strategy-builder', icon: <BuildOutlined />, label: 'Strategy Builder' },
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
        background: 'rgba(40, 48, 56, 0.42)',
        backdropFilter: 'blur(28px) saturate(160%)',
        borderRight: '1px solid rgba(255,255,255,0.09)',
        display: 'flex',
        flexDirection: 'column',
        padding: '24px 16px',
        height: '100vh',
        position: 'sticky',
        top: 0,
      }}
    >
      {/* Logo */}
      <Logo style={{ marginBottom: 32 }} />

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