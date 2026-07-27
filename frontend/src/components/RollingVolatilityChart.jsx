import { LineChart, Line, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';
import EmptyChart from './EmptyChart';

const AMBER = '#FF8A5C';

/** Rolling volatility line chart. series: [{ label, vol }]. */
export default function RollingVolatilityChart({ series, height = 200, emptyText = 'Not enough history for rolling volatility.' }) {
  const data = series || [];

  if (data.length <= 1) {
    return <EmptyChart text={emptyText} />;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
        <Line type="monotone" dataKey="vol" stroke={AMBER} strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}