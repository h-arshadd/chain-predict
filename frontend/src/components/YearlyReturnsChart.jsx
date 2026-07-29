import { BarChart, Bar, Cell, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';
import EmptyChart from './EmptyChart';

const MINT = '#3DDC97';
const RED = '#F0466B';

/**
 * Per-year return bar chart, colored green/red by sign.
 * data: [{ year, ret }], ret as a percentage number (e.g. 12.5 for +12.5%).
 */
export default function YearlyReturnsChart({ data, height = 200, emptyText = 'Not enough history to compute yearly returns.', centered = false }) {
  const list = data || [];

  if (list.length === 0) {
    return <EmptyChart text={emptyText} centered={centered} />;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={list} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="year" tick={axisStyle} axisLine={false} tickLine={false} />
        <YAxis tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => `${v.toFixed(0)}%`} />
        <Tooltip cursor={false} contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v.toFixed(2)}%`} />
        <Bar dataKey="ret" radius={[6, 6, 6, 6]}>
          {list.map((entry, i) => (
            <Cell key={i} fill={entry.ret >= 0 ? MINT : RED} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}