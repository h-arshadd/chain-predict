const MINT = '#3DDC97';
const RED = '#F0466B';

export default function StatBox({ label, value, positive }) {
  return (
    <div style={{ background: 'rgba(21, 26, 31, 0.75)', backdropFilter: 'blur(16px)', border: '1px solid rgba(255,255,255,0.07)', borderRadius: 22, padding: 16 }}>
      <div style={{ color: '#9096A0', fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 19, fontWeight: 700, color: positive === undefined ? '#F5F6F7' : positive ? MINT : RED, fontFamily: 'ui-monospace, monospace' }}>
        {value}
      </div>
    </div>
  );
}