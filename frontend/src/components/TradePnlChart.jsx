import { BarChart, Bar, Cell, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';

const MINT = '#3DDC97';
const RED = '#F0466B';

/**
 * Per-trade PnL bar chart. One bar per trade, colored green/red by
 * whether that trade was a win or a loss.
 *
 * trades: raw trade objects (e.g. data.trades) with a net_pnl field.
 * Pass `reverse` when the source list is newest-first, so the chart
 * still reads left-to-right in chronological order.
 */
export default function TradePnlChart({ trades, reverse = false, height = 200 }) {
  const list = trades || [];
  const ordered = reverse ? [...list].reverse() : list;
  const data = ordered.map((t, i) => ({ idx: `#${i + 1}`, pnl: t.net_pnl ?? 0 }));

  if (data.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6B7280', fontSize: 13 }}>
        No trades yet.
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="idx" tick={axisStyle} axisLine={false} tickLine={false} />
        <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`} />
        <Bar dataKey="pnl" radius={[4, 4, 4, 4]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.pnl >= 0 ? MINT : RED} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}