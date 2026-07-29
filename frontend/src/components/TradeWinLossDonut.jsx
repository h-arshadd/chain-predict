import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend } from 'recharts';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle } from '../lib/chartStyle';
import { fmtUsd } from '../lib/format';

const MINT = '#3DDC97';
const RED = '#F0466B';
const GREY = '#6B7280';

/**
 * Win/Loss donut chart. Replaces the old per-trade PnL bar sequence
 * with a simple breakdown of how many trades were wins vs losses.
 *
 * trades: raw trade objects (e.g. data.trades) with a net_pnl field.
 */
export default function TradeWinLossDonut({ trades, height = 220 }) {
  const list = trades || [];

  const winTrades = list.filter((t) => (t.net_pnl ?? 0) > 0);
  const lossTrades = list.filter((t) => (t.net_pnl ?? 0) < 0);
  const wins = winTrades.length;
  const losses = lossTrades.length;
  const flat = list.length - wins - losses;

  const winPnl = winTrades.reduce((sum, t) => sum + (t.net_pnl ?? 0), 0);
  const lossPnl = lossTrades.reduce((sum, t) => sum + (t.net_pnl ?? 0), 0);

  const data = [
    { name: 'Wins', value: wins, color: MINT, pnl: winPnl },
    { name: 'Losses', value: losses, color: RED, pnl: lossPnl },
  ];
  if (flat > 0) data.push({ name: 'Breakeven', value: flat, color: GREY, pnl: 0 });

  if (list.length === 0) {
    return (
      <div style={{ height, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6B7280', fontSize: 13 }}>
        No trades yet.
      </div>
    );
  }

  const winRate = list.length > 0 ? (wins / list.length) * 100 : 0;

  return (
    <div style={{ position: 'relative', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            innerRadius="65%"
            outerRadius="90%"
            paddingAngle={data.some((d) => d.value === 0) ? 0 : 2}
            stroke="none"
          >
            {data.map((entry, i) => (
              <Cell key={i} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            position={{ y: 0 }}
            wrapperStyle={{ zIndex: 10 }}
            contentStyle={tooltipStyle}
            labelStyle={tooltipLabelStyle}
            itemStyle={tooltipItemStyle}
            content={({ active, payload }) => {
              if (!active || !payload || !payload.length) return null;
              const entry = payload[0].payload;
              const pct = list.length > 0 ? (entry.value / list.length) * 100 : 0;
              return (
                <div style={{ ...tooltipStyle, padding: '10px 12px' }}>
                  <div style={{ ...tooltipLabelStyle, fontWeight: 700, marginBottom: 4 }}>{entry.name}</div>
                  <div style={{ ...tooltipItemStyle }}>
                    {entry.value} trade{entry.value === 1 ? '' : 's'} &middot; {pct.toFixed(1)}%
                  </div>
                  {entry.name !== 'Breakeven' && (
                    <div style={{ color: entry.pnl >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', marginTop: 2 }}>
                      {fmtUsd(entry.pnl)}
                    </div>
                  )}
                </div>
              );
            }}
          />
          <Legend
            verticalAlign="bottom"
            height={24}
            iconType="circle"
            iconSize={8}
            wrapperStyle={{ fontSize: 12, color: '#9096A0' }}
          />
        </PieChart>
      </ResponsiveContainer>
      <div
        style={{
          position: 'absolute',
          top: '44%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          pointerEvents: 'none',
        }}
      >
        <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>
          {winRate.toFixed(0)}%
        </div>
        <div style={{ fontSize: 11, color: '#9096A0', marginTop: 2 }}>Win Rate</div>
      </div>
    </div>
  );
}