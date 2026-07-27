import { AreaChart, Area, ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';
import EmptyChart from './EmptyChart';

const MINT = '#3DDC97';

/**
 * Account balance over time. series: [{ label, balance }].
 * `gradientId` must be unique per chart instance on the page (SVG <defs>
 * ids are global to the document, so two mounted charts sharing one id
 * would silently reuse whichever gradient rendered first).
 */
export default function EquityCurveChart({ series, gradientId = 'equityGrad', height = 220, emptyText = 'Not enough data to plot an equity curve.', centered = false }) {
  const data = series || [];

  if (data.length <= 1) {
    return <EmptyChart text={emptyText} centered={centered} />;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={MINT} stopOpacity={0.35} />
            <stop offset="95%" stopColor={MINT} stopOpacity={0} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
        <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
        <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
        <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
        <Area type="monotone" dataKey="balance" stroke={MINT} strokeWidth={2.5} fill={`url(#${gradientId})`} />
      </AreaChart>
    </ResponsiveContainer>
  );
}