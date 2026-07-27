export default function KeyValue({ label, value, mono, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.05)', gap: 12 }}>
      <span style={{ color: '#9096A0', fontSize: 13 }}>{label}</span>
      <span style={{ color: color || '#F5F6F7', fontSize: 13, fontWeight: 600, fontFamily: mono ? 'ui-monospace, monospace' : undefined, textAlign: 'right' }}>
        {value}
      </span>
    </div>
  );
}