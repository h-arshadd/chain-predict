import { useNavigate } from 'react-router-dom';

/**
 * App logo/wordmark -- clicking it always navigates to the dashboard
 * ("/"), regardless of which page it's rendered on. Used in Sidebar
 * (rendered once in DashboardLayout, so it's on every page already).
 */
export default function Logo({ style }) {
  const navigate = useNavigate();

  return (
    <div
      onClick={() => navigate('/')}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') navigate('/');
      }}
      style={{
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '0 8px',
        cursor: 'pointer',
        ...style,
      }}
    >
      <div
        style={{
          width: 30,
          height: 30,
          borderRadius: 9,
          background: '#2FA876',
          flexShrink: 0,
        }}
      />
      <span style={{ fontSize: 19, fontWeight: 700, color: '#F5F6F7', letterSpacing: -0.3 }}>
        ChainPredict
      </span>
    </div>
  );
}