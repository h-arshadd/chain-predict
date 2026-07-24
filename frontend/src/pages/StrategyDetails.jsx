import { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table, Switch, Modal, Spin, Alert, message } from 'antd';
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
  live: { label: 'Live', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  disabled: { label: 'Disabled', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  conflicted: { label: 'Conflicted', bg: 'rgba(240,70,107,0.14)', fg: RED },
};

const panel = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

const tooltipStyle = { background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 };
const axisStyle = { fill: '#6B7280', fontSize: 11 };
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
    <div style={{ height: 200, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6B7280', fontSize: 13, textAlign: 'center', padding: '0 20px' }}>
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

// Same {timestamp: value} -> recharts-array helpers ExecutionDetails.jsx
// already established for compute_stats()'s plot output -- reused
// identically here so both pages read the exact same stats shape the
// same way.
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

export default function StrategyDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.get(`/api/strategies/${id}`)
      .then((res) => setData(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  const toggleEnabled = (nextEnabled) => {
    if (!nextEnabled) {
      submitToggle(false);
      return;
    }
    Modal.confirm({
      title: 'Make this the live strategy for this pair?',
      content: `If another strategy is currently enabled for ${data.exchange}/${data.coin.toUpperCase()}, it will be disabled automatically. This takes effect immediately.`,
      okText: 'Switch strategy',
      okButtonProps: { danger: true },
      onOk: () => submitToggle(true),
    });
  };

  const submitToggle = (nextEnabled) => {
    api.patch(`/api/strategies/${id}/enabled`, { execution_enabled: nextEnabled })
      .then((res) => {
        setData(res.data);
        message.success(nextEnabled ? 'Strategy enabled — now live for this pair' : 'Strategy disabled');
      })
      .catch((err) => message.error(err.message));
  };

  const plots = data?.stats?.plots || {};
  const metrics = data?.stats?.metrics || {};

  const equitySeries = useMemo(
    () => (data?.equity_curve || []).map((p) => ({ ts: p.timestamp, equity: p.balance, label: new Date(p.timestamp).toLocaleDateString() })),
    [data]
  );
  const drawdownSeries = useMemo(() => recordsToSeries(plots.drawdown?.drawdown_series, 'dd'), [plots]);
  const rollingSharpeSeries = useMemo(() => recordsToSeries(plots.rolling_sharpe?.rolling_sharpe, 'sharpe'), [plots]);
  const monthlyRows = useMemo(() => monthlyHeatmapToRows(plots.monthly_heatmap?.monthly_returns), [plots]);
  const tradeDistribution = data
    ? [
        { name: 'Wins', value: data.trade_stats?.wins ?? 0 },
        { name: 'Losses', value: data.trade_stats?.losses ?? 0 },
      ]
    : [];
  const hasTradeDistribution = tradeDistribution.some((d) => d.value > 0);

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
          message="Couldn't load this strategy"
          description={error}
          action={<button onClick={load} style={backBtnStyle}>Retry</button>}
          showIcon
        />
      </div>
    );
  }

  if (!data) return null;

  const statusStyle = STATUS_META[data.pair_status] || STATUS_META.disabled;
  const sc = data.strategy_config || {};

  return (
    <div style={{ paddingTop: 8 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 14, marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
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
            <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2 }}>
              {data.coin.toUpperCase()} · {data.exchange} · {data.time_horizon} · Strategy ID {data.strategy_id}
              {data.data_source && <> · performance from {data.data_source}</>}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ color: '#9096A0', fontSize: 13 }}>Execution Enabled</span>
          <Switch checked={data.execution_enabled} onChange={toggleEnabled} />
        </div>
      </div>

      {data.pair_status === 'conflicted' && (
        <Alert
          type="warning"
          showIcon
          message="This pair has more than one strategy enabled"
          description="Execution treats this as misconfigured and skips the pair entirely until only one strategy is enabled. Use the switch above, or on another strategy for this pair, to resolve it."
          style={{ marginBottom: 20 }}
        />
      )}

      {/* Performance summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Latest Return" value={data.latest_return_pct == null ? '—' : `${data.latest_return_pct >= 0 ? '+' : ''}${data.latest_return_pct.toFixed(2)}%`} positive={data.latest_return_pct == null ? undefined : data.latest_return_pct >= 0} />
        <StatBox label="Sharpe Ratio" value={metrics.sharpe != null ? metrics.sharpe.toFixed(2) : '—'} />
        <StatBox label="Sortino Ratio" value={metrics.sortino != null ? metrics.sortino.toFixed(2) : '—'} />
        <StatBox label="Win Rate" value={data.trade_stats?.win_rate_pct != null ? `${data.trade_stats.win_rate_pct.toFixed(1)}%` : '—'} />
        <StatBox label="Profit Factor" value={metrics.profit_factor != null ? metrics.profit_factor.toFixed(2) : '—'} />
        <StatBox label="Max Drawdown" value={metrics.max_drawdown != null ? `${(metrics.max_drawdown * 100).toFixed(2)}%` : '—'} positive={false} />
      </div>

      {/* Config / Indicators / Risk */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginBottom: 20 }}>
        <Panel title="Strategy Configuration">
          <KeyValue label="Entry Logic (Long)" value={sc.entry_logic_long || '—'} />
          <KeyValue label="Entry Logic (Short)" value={sc.entry_logic_short || '—'} />
          <KeyValue label="Simulator Enabled" value={data.simulator_enabled ? 'Yes' : 'No'} />
        </Panel>
        <Panel title="Indicators Used">
          {sc.indicators?.length ? (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {sc.indicators.map((ind) => (
                <span
                  key={ind}
                  style={{
                    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)',
                    color: '#F5F6F7', fontSize: 12.5, fontWeight: 600, padding: '6px 12px', borderRadius: 999,
                  }}
                >
                  {ind}
                </span>
              ))}
            </div>
          ) : (
            <span style={{ color: '#6B7280', fontSize: 13 }}>No indicators recorded.</span>
          )}
        </Panel>
        <Panel title="Risk Management">
          <KeyValue label="Take Profit" value={sc.take_profit_value != null ? `${sc.take_profit_value} (${sc.take_profit_type})` : '—'} />
          <KeyValue label="Stop Loss" value={sc.stop_loss_value != null ? `${sc.stop_loss_value} (${sc.stop_loss_type})` : '—'} />
        </Panel>
      </div>

      {/* Equity Curve + Drawdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Equity Curve">
          {equitySeries.length > 1 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={equitySeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor={MINT} stopOpacity={0.35} />
                    <stop offset="95%" stopColor={MINT} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Area type="monotone" dataKey="equity" stroke={MINT} strokeWidth={2.5} fill="url(#eqGrad)" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text={data.data_source ? 'Not enough trade history for an equity curve yet.' : 'This strategy has never traded in execution or simulator yet.'} />
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
            <EmptyChart text="Not enough trade history for a drawdown chart yet." />
          )}
        </Panel>
      </div>

      {/* Monthly Returns + Trade Distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
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
                          style={{ background: heatColor(v), borderRadius: 6, textAlign: 'center', padding: '6px 4px', color: '#F5F6F7', minWidth: 40 }}
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
        <Panel title="Trade Distribution">
          {hasTradeDistribution ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={tradeDistribution} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" radius={[6, 6, 6, 6]}>
                  {tradeDistribution.map((entry, i) => (
                    <Cell key={i} fill={entry.name === 'Wins' ? MINT : RED} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart text="No completed trades yet." />
          )}
        </Panel>
      </div>

      {/* Backtest Summary — honest placeholder, not fabricated */}
      <div style={{ marginBottom: 20 }}>
        <Panel title="Backtest Summary">
          {data.backtest_summary ? (
            <div>{/* real backtest data would render here once the Backtests module exists */}</div>
          ) : (
            <EmptyChart text="Backtest data isn't available yet — the Backtests module hasn't been built. This section will populate automatically once it exists." />
          )}
        </Panel>
      </div>

      {/* Trade stats + Recent trades */}
      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 20 }}>
        <Panel title="Trade Statistics">
          <KeyValue label="Total Trades" value={data.trade_stats?.total_trades ?? 0} mono />
          <KeyValue label="Wins" value={data.trade_stats?.wins ?? 0} mono />
          <KeyValue label="Losses" value={data.trade_stats?.losses ?? 0} mono />
          <KeyValue label="Win Rate" value={data.trade_stats?.win_rate_pct != null ? `${data.trade_stats.win_rate_pct.toFixed(1)}%` : '—'} mono />
        </Panel>
        <Panel title="Recent Trades">
          <Table
            size="small"
            pagination={{ pageSize: 10 }}
            rowKey={(r) => r.entry_date_time}
            dataSource={data.recent_trades || []}
            locale={{ emptyText: 'No trades yet for this strategy.' }}
            columns={[
              { title: 'Direction', dataIndex: 'direction', key: 'direction', render: (v) => <span style={{ color: v === 'long' ? MINT : RED, fontWeight: 600, textTransform: 'capitalize' }}>{v}</span> },
              { title: 'Entry Time', dataIndex: 'entry_date_time', key: 'entry_date_time', render: (v) => (v ? new Date(v).toLocaleString() : '—') },
              { title: 'Entry Price', dataIndex: 'entry_price', key: 'entry_price', render: (v) => (v != null ? v.toLocaleString() : '—') },
              { title: 'Exit Time', dataIndex: 'exit_date_time', key: 'exit_date_time', render: (v) => (v ? new Date(v).toLocaleString() : '—') },
              { title: 'Exit Price', dataIndex: 'exit_price', key: 'exit_price', render: (v) => (v != null ? v.toLocaleString() : '—') },
              {
                title: 'Net PnL', dataIndex: 'net_pnl', key: 'net_pnl',
                render: (v) => (v == null ? '—' : <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}</span>),
              },
              { title: 'Exit Reason', dataIndex: 'exit_reason', key: 'exit_reason', render: (v) => v || '—' },
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