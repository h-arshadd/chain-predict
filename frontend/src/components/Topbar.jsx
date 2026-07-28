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
      </div>

    </header>
  );
}