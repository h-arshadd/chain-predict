const MINT = '#3DDC97';
const RED = '#F0466B';

export default function StatBox({ label, value, positive }) {
  return (
    <div style={{ background: 'rgba(120, 130, 140, 0.16)', backdropFilter: 'blur(30px) saturate(180%)', border: '1px solid rgba(255,255,255,0.22)', borderRadius: 22, padding: 16, boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.18), inset 0 0 40px rgba(255,255,255,0.03), 0 8px 24px rgba(0,0,0,0.2)' }}>
      <div style={{ color: '#9096A0', fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 19, fontWeight: 700, color: positive === undefined ? '#F5F6F7' : positive ? MINT : RED, fontFamily: 'ui-monospace, monospace' }}>
        {value}
      </div>
    </div>
  );
}