import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';
import EmptyChart from './EmptyChart';

const RED = '#F0466B';

/**
 * Drawdown-over-time area chart. series: [{ label, dd }], dd as a
 * fraction (e.g. -0.12 for -12%).
 * `gradientId` must be unique per chart instance on the page.
 */
export default function DrawdownChart({ series, gradientId = 'ddGrad', height = 220, emptyText = 'No drawdown periods.' }) {
  const data = series || [];

  if (data.length <= 1) {
    return <EmptyChart text={emptyText} />;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={RED} stopOpacity={0.35} />
            <stop offset="95%" stopColor={RED} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${(v * 100).toFixed(2)}%`} />
        <Area type="monotone" dataKey="dd" stroke={RED} strokeWidth={2} fill={`url(#${gradientId})`} />
      </AreaChart>
    </ResponsiveContainer>
  );
}