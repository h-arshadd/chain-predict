/**
 * Placeholder shown inside a chart panel when there's no data to plot.
 * Pass `centered` to add text-align/padding for panels where the message
 * might wrap onto multiple lines (used on Simulation/Model details).
 */
export default function EmptyChart({ text, centered = false }) {
  return (
    <div
      style={{
        height: 200,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        color: '#6B7280',
        fontSize: 13,
        ...(centered ? { textAlign: 'center', padding: '0 20px' } : {}),
      }}
    >
      {text}
    </div>
  );
}