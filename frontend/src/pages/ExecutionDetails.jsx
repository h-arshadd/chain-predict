import { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table, Spin, Alert } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { api } from '../lib/api';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const STATUS_META = {
  running: { label: 'Running', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  paused: { label: 'Paused', bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  unassigned: { label: 'Unassigned', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  never_run: { label: 'Never Run', bg: 'rgba(255,255,255,0.06)', fg: '#6B7280' },
};

const panel = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

const tooltipStyle = { background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 };
const axisStyle = { fill: '#6B7280', fontSize: 11 };
const fmtUsd = (v) => (v == null ? '—' : v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }));
const pnlColor = (v) => (v == null ? '#6B7280' : v > 0 ? MINT : v < 0 ? RED : '#9096A0');

function Panel({ title, children, style, right }) {
  return (
    <div style={{ ...panel, padding: 22, ...style }}>
      {(title || right) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          {title && <h3 style={{ fontSize: 15.5, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{title}</h3>}
          {right}
        </div>
      )}
      {children}
    </div>
  );
}

function EmptyChart({ text }) {
  return (
    <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6B7280', fontSize: 13 }}>
      {text}
    </div>
  );
}

function KeyValue({ label, value, mono, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.05)', gap: 12 }}>
      <span style={{ color: '#9096A0', fontSize: 13 }}>{label}</span>
      <span style={{ color: color || '#F5F6F7', fontSize: 13, fontWeight: 600, fontFamily: mono ? 'ui-monospace, monospace' : undefined, textAlign: 'right' }}>
        {value}
      </span>
    </div>
  );
}

function StatBox({ label, value, positive }) {
  return (
    <div style={{ ...panel, padding: 16 }}>
      <div style={{ color: '#9096A0', fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 19, fontWeight: 700, color: positive === undefined ? '#F5F6F7' : positive ? MINT : RED, fontFamily: 'ui-monospace, monospace' }}>
        {value}
      </div>
    </div>
  );
}

// ---- helpers to turn compute_stats()'s {timestamp: value} records into recharts arrays ----
function recordsToSeries(records, valueKey = 'value') {
  if (!records) return [];
  return Object.entries(records)
    .map(([ts, val]) => ({ ts, [valueKey]: val, label: new Date(ts).toLocaleDateString() }))
    .filter((d) => d[valueKey] != null);
}

function monthlyHeatmapToRows(monthly) {
  if (!monthly) return [];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return Object.entries(monthly)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([year, monthVals]) => ({
      year,
      cells: months.map((_, i) => monthVals[i + 1] ?? monthVals[String(i + 1)] ?? null),
    }));
}

function heatColor(v) {
  if (v == null) return 'rgba(255,255,255,0.03)';
  const intensity = Math.min(Math.abs(v) * 6, 1);
  return v >= 0 ? `rgba(61,220,151,${0.15 + intensity * 0.6})` : `rgba(240,70,107,${0.15 + intensity * 0.6})`;
}

export default function ExecutionDetails() {
  const { exchange, symbol } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.get(`/api/executions/${exchange}/${symbol}`)
      .then((res) => setData(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [exchange, symbol]);

  useEffect(() => {
    load();
  }, [load]);

  const equitySeries = useMemo(
    () => (data?.equity_curve || []).map((p) => ({ ts: p.timestamp, balance: p.balance, label: new Date(p.timestamp).toLocaleDateString() })),
    [data]
  );

  const tradesForChart = useMemo(
    () => [...(data?.trades || [])].reverse().map((t, i) => ({ idx: `#${i + 1}`, pnl: t.net_pnl ?? 0 })),
    [data]
  );

  const plots = data?.stats?.plots || {};
  const drawdownSeries = recordsToSeries(plots.drawdown?.drawdown_series, 'dd');
  const rollingSharpeSeries = recordsToSeries(plots.rolling_sharpe?.rolling_sharpe, 'sharpe');
  const rollingVolSeries = recordsToSeries(plots.rolling_volatility?.rolling_volatility, 'vol');
  const yearlyReturns = plots.yearly_returns?.yearly_returns
    ? Object.entries(plots.yearly_returns.yearly_returns).map(([year, ret]) => ({ year, ret: ret * 100 }))
    : [];
  const monthlyRows = monthlyHeatmapToRows(plots.monthly_heatmap?.monthly_returns);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '100px 0' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ paddingTop: 8 }}>
        <Alert
          type="error"
          message="Couldn't load this execution"
          description={error}
          action={<button onClick={load} style={backBtnStyle}>Retry</button>}
          showIcon
        />
      </div>
    );
  }

  if (!data) return null;

  const statusStyle = STATUS_META[data.status] || STATUS_META.never_run;
  const sc = data.strategy_config || {};
  const live = data.live_position;
  const winLoss = data.win_loss;

  return (
    <div style={{ paddingTop: 8 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
        <button onClick={() => navigate(-1)} style={backBtnStyle}>
          <ArrowLeftOutlined />
        </button>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <h2 style={{ fontSize: 22, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{data.strategy_name}</h2>
            <Tag style={{ background: statusStyle.bg, color: statusStyle.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
              {statusStyle.label}
            </Tag>
          </div>
          <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2, textTransform: 'uppercase' }}>
            {data.symbol} &middot; {data.exchange} &middot; {data.account_name || 'No wallet assigned'}
          </div>
          {data.status === 'unassigned' && (
            <button
              onClick={() => navigate(`/strategies?coin=${data.symbol}`)}
              style={{ ...linkBtnStyle, marginTop: 6 }}
            >
              No strategy enabled for this pair &rarr; go enable one
            </button>
          )}
        </div>
      </div>

      {/* Current position summary strip -- prefers live Bybit data, falls back to DB state */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox
          label="Position"
          value={live ? `${live.side} ${live.size}` : data.position ? `${data.position.direction} ${data.position.quantity}` : 'Flat'}
        />
        <StatBox label="Entry / Avg Price" value={live ? live.avg_price.toLocaleString() : data.position?.entry_price?.toLocaleString() ?? '—'} />
        <StatBox label="Balance" value={fmtUsd(data.balance)} />
        <StatBox label="Cumulative PnL" value={`${data.cumulative_pnl >= 0 ? '+' : ''}${fmtUsd(data.cumulative_pnl)}`} positive={data.cumulative_pnl >= 0} />
        <StatBox label="Take Profit" value={(live?.take_profit ?? data.position?.take_profit)?.toLocaleString() ?? '—'} />
        <StatBox label="Stop Loss" value={(live?.stop_loss ?? data.position?.stop_loss)?.toLocaleString() ?? '—'} />
      </div>

      {/* Strategy info / Wallet info / Risk statistics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginBottom: 20 }}>
        <Panel title="Strategy Information">
          <KeyValue label="Symbol" value={data.symbol.toUpperCase()} />
          <KeyValue label="Timeframe" value={data.time_horizon || '—'} />
          <KeyValue label="Entry Logic (Long)" value={sc.entry_logic_long || 'No long rule'} />
          <KeyValue label="Entry Logic (Short)" value={sc.entry_logic_short || 'No short rule'} />
          <KeyValue label="Indicators" value={sc.indicators?.length ? sc.indicators.join(', ') : '—'} />
        </Panel>
        <Panel title="Wallet Information">
          <KeyValue label="Wallet" value={data.account_name || 'Unassigned'} />
          <KeyValue label="Exchange" value={data.exchange} />
          <KeyValue
            label="Wallet Status"
            value={data.wallet_enabled == null ? '—' : data.wallet_enabled ? 'Enabled' : 'Disabled'}
            color={data.wallet_enabled == null ? undefined : data.wallet_enabled ? MINT : AMBER}
          />
          <KeyValue label="Last Processed" value={data.last_processed ? new Date(data.last_processed).toLocaleString() : '—'} />
        </Panel>
        <Panel title="Risk Statistics">
          <KeyValue label="Take Profit" value={sc.take_profit_value != null ? `${sc.take_profit_value}${sc.take_profit_type === 'percentage' ? '%' : ''}` : '—'} />
          <KeyValue label="Stop Loss" value={sc.stop_loss_value != null ? `${sc.stop_loss_value}${sc.stop_loss_type === 'percentage' ? '%' : ''}` : '—'} />
          <KeyValue label="Commission" value={data.commission != null ? `${(data.commission * 100).toFixed(3)}%` : '—'} />
          <KeyValue label="Slippage" value={data.slippage != null ? `${(data.slippage * 100).toFixed(3)}%` : '—'} />
          <KeyValue label="Long / Short Allowed" value={`${data.allow_long ? 'Long' : ''}${data.allow_long && data.allow_short ? ' / ' : ''}${data.allow_short ? 'Short' : ''}` || '—'} />
        </Panel>
      </div>

      {/* Trade summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Total Trades" value={data.total_trades ?? 0} />
        <StatBox label="Total Net Profit" value={`${data.total_net_profit >= 0 ? '+' : ''}${fmtUsd(data.total_net_profit)}`} positive={data.total_net_profit >= 0} />
        <StatBox label="Win Rate" value={winLoss ? `${(winLoss.win_rate * 100).toFixed(1)}%` : '—'} />
        <StatBox label="Wins / Losses" value={winLoss ? `${winLoss.wins} / ${winLoss.losses}` : '—'} />
      </div>

      {/* Equity Curve + Drawdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Equity Curve">
          {equitySeries.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={equitySeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="eqGrad2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={MINT} stopOpacity={0.35} />
                    <stop offset="95%" stopColor={MINT} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area type="monotone" dataKey="balance" stroke={MINT} strokeWidth={2.5} fill="url(#eqGrad2)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text="Not enough closed trades yet to plot an equity curve." />
          )}
        </Panel>
        <Panel title="Drawdown">
          {drawdownSeries.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={drawdownSeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={RED} stopOpacity={0.35} />
                    <stop offset="95%" stopColor={RED} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(2)}%`} />
                <Area type="monotone" dataKey="dd" stroke={RED} strokeWidth={2} fill="url(#ddGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text={data.stats ? 'No drawdown periods yet.' : 'Not enough trade history for stats yet.'} />
          )}
        </Panel>
      </div>

      {/* Rolling Sharpe + Rolling Volatility */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Rolling Sharpe">
          {rollingSharpeSeries.length > 1 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={rollingSharpeSeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="sharpe" stroke={MINT} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text="Not enough history for a rolling Sharpe window yet." />
          )}
        </Panel>
        <Panel title="Rolling Volatility">
          {rollingVolSeries.length > 1 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={rollingVolSeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="vol" stroke={AMBER} strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text="Not enough history for rolling volatility yet." />
          )}
        </Panel>
      </div>

      {/* Yearly Returns + Trade PnL sequence */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Yearly Returns">
          {yearlyReturns.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={yearlyReturns} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="year" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${v.toFixed(2)}%`} />
                <Bar dataKey="ret" radius={[6, 6, 6, 6]}>
                  {yearlyReturns.map((entry, i) => (
                    <Cell key={i} fill={entry.ret >= 0 ? MINT : RED} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text="Not enough history to compute yearly returns." />
          )}
        </Panel>
        <Panel title="Trade PnL Sequence">
          {tradesForChart.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={tradesForChart} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="idx" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line type="monotone" dataKey="pnl" stroke={MINT} strokeWidth={2.5} dot={{ r: 3, fill: MINT }} />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text="No trades yet." />
          )}
        </Panel>
      </div>

      {/* Monthly Returns heatmap */}
      <div style={{ marginBottom: 20 }}>
        <Panel title="Monthly Returns">
          {monthlyRows.length > 0 ? (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'separate', borderSpacing: 4, fontSize: 12 }}>
                <thead>
                  <tr>
                    <th style={{ color: '#6B7280', textAlign: 'left', fontWeight: 600 }}>Year</th>
                    {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'].map((m) => (
                      <th key={m} style={{ color: '#6B7280', fontWeight: 600 }}>{m}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {monthlyRows.map((row) => (
                    <tr key={row.year}>
                      <td style={{ color: '#F5F6F7', fontWeight: 600 }}>{row.year}</td>
                      {row.cells.map((v, i) => (
                        <td
                          key={i}
                          title={v != null ? `${(v * 100).toFixed(2)}%` : ''}
                          style={{ background: heatColor(v), borderRadius: 6, textAlign: 'center', padding: '6px 4px', color: '#F5F6F7', minWidth: 44 }}
                        >
                          {v != null ? `${(v * 100).toFixed(1)}%` : ''}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyChart text="Not enough history to compute monthly returns." />
          )}
        </Panel>
      </div>

      {/* Trade Ledger */}
      <div style={{ marginBottom: 8 }}>
        <Panel title="Trade Ledger">
          <Table
            size="small"
            pagination={{ pageSize: 10 }}
            rowKey={(r) => r.entry_date_time}
            dataSource={data.trades || []}
            locale={{ emptyText: 'No trades yet for this pair.' }}
            columns={[
              { title: 'Direction', dataIndex: 'direction', key: 'direction', render: (v) => <span style={{ color: v === 'long' ? MINT : RED, fontWeight: 600, textTransform: 'capitalize' }}>{v}</span> },
              { title: 'Entry Time', dataIndex: 'entry_date_time', key: 'entry_date_time', render: (v) => new Date(v).toLocaleString() },
              { title: 'Entry Price', dataIndex: 'entry_price', key: 'entry_price', render: (v) => v?.toLocaleString() },
              { title: 'Qty', dataIndex: 'quantity', key: 'quantity' },
              { title: 'Exit Time', dataIndex: 'exit_date_time', key: 'exit_date_time', render: (v) => (v ? new Date(v).toLocaleString() : '—') },
              { title: 'Exit Price', dataIndex: 'exit_price', key: 'exit_price', render: (v) => (v != null ? v.toLocaleString() : '—') },
              {
                title: 'Net PnL', dataIndex: 'net_pnl', key: 'net_pnl',
                render: (v) => v == null ? '—' : <span style={{ color: v >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}</span>,
              },
              { title: 'Exit Reason', dataIndex: 'exit_reason', key: 'exit_reason', render: (v) => v || '—' },
              {
                title: 'Status', dataIndex: 'status', key: 'status',
                render: (v) => (
                  <Tag style={{
                    background: v === 'open' ? 'rgba(61,220,151,0.12)' : 'rgba(255,255,255,0.06)',
                    color: v === 'open' ? MINT : '#9096A0', border: 'none', borderRadius: 8, fontWeight: 600,
                  }}>
                    {v}
                  </Tag>
                ),
              },
            ]}
          />
        </Panel>
      </div>
    </div>
  );
}

const backBtnStyle = {
  width: 36, height: 36, borderRadius: 10, border: '1px solid rgba(255,255,255,0.1)',
  background: 'rgba(255,255,255,0.04)', color: '#F5F6F7', cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
};

const linkBtnStyle = {
  background: 'none', border: 'none', padding: 0, cursor: 'pointer',
  color: '#3DDC97', fontSize: 12.5, fontWeight: 600, textDecoration: 'underline',
};