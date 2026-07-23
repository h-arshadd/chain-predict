import { Outlet } from 'react-router-dom';
import Sidebar from '../components/Sidebar';
import TopBar from '../components/Topbar';

export default function DashboardLayout() {
  return (
    <div style={{ minHeight: '100vh', background: '#0B0E11', position: 'relative' }}>
      {/* Signature ambient glow blobs, matching the reference image's lighting */}
      <div className="glow-field">
        <div className="glow-blob glow-blob--mint" />
        <div className="glow-blob glow-blob--amber" />
        <div className="glow-blob glow-blob--blue" />
      </div>

      <div className="app-shell" style={{ display: 'flex', minHeight: '100vh' }}>
        <Sidebar />
        <div style={{ flex: 1, minWidth: 0 }}>
          <TopBar />
          <main style={{ padding: '0 32px 32px' }}>
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
}