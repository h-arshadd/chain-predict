import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table, Spin, Alert } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { api } from '../lib/api';
import TradePnlChart from '../components/TradePnlChart';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

// Real status values from metadata.backtest.status (see
// metadata_utils.create_backtest_table / start_backtest / complete_backtest
// / fail_backtest) -- a lifecycle this table didn't have until this page
// needed one to run backtests on demand instead of only ever showing
// pre-finished rows.
const STATUS_META = {
  pending: { label: 'Pending', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  running: { label: 'Running', bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  completed: { label: 'Completed', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  failed: { label: 'Failed', bg: 'rgba(240,70,107,0.14)', fg: RED },
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

// Subset of compute_stats()'s dynamically-discovered quantstats metrics
// dict (data.stats.metrics) to headline as cards -- the full ~55-metric
// dict still exists underneath, these are just the ones worth a card.
// "pct" metrics are ratios quantstats returns as fractions (0.42 -> 42%),
// "ratio" metrics are shown as-is to 2 decimals. `invert` means a LOWER
// number is the better outcome (drawdown, losses, risk measures), so the
// green/red coloring flips for those.
const METRIC_CARDS = [
  { key: 'sharpe', label: 'Sharpe Ratio', kind: 'ratio' },
  { key: 'smart_sharpe', label: 'Smart Sharpe', kind: 'ratio' },
  { key: 'sortino', label: 'Sortino Ratio', kind: 'ratio' },
  { key: 'smart_sortino', label: 'Smart Sortino', kind: 'ratio' },
  { key: 'adjusted_sortino', label: 'Adjusted Sortino', kind: 'ratio' },
  { key: 'calmar', label: 'Calmar Ratio', kind: 'ratio' },
  { key: 'omega', label: 'Omega Ratio', kind: 'ratio' },
  { key: 'max_drawdown', label: 'Max Drawdown', kind: 'pct', invert: true },
  { key: 'cagr', label: 'CAGR', kind: 'pct' },
  { key: 'comp', label: 'Total Compounded Return', kind: 'pct' },
  { key: 'volatility', label: 'Volatility (ann.)', kind: 'pct', invert: true },
  { key: 'win_rate', label: 'Win Rate', kind: 'pct' },
  { key: 'win_loss_ratio', label: 'Win/Loss Ratio', kind: 'ratio' },
  { key: 'profit_factor', label: 'Profit Factor', kind: 'ratio' },
  { key: 'profit_ratio', label: 'Profit Ratio', kind: 'ratio' },
  { key: 'payoff_ratio', label: 'Payoff Ratio', kind: 'ratio' },
  { key: 'gain_to_pain_ratio', label: 'Gain to Pain Ratio', kind: 'ratio' },
  { key: 'best', label: 'Best Period', kind: 'pct' },
  { key: 'worst', label: 'Worst Period', kind: 'pct', invert: true },
  { key: 'avg_return', label: 'Avg Return', kind: 'pct' },
  { key: 'avg_win', label: 'Avg Win', kind: 'pct' },
  { key: 'avg_loss', label: 'Avg Loss', kind: 'pct', invert: true },
  { key: 'expected_return', label: 'Expected Return', kind: 'pct' },
  { key: 'expected_shortfall', label: 'Expected Shortfall (CVaR)', kind: 'pct', invert: true },
  { key: 'value_at_risk', label: 'Value at Risk', kind: 'pct', invert: true },
  { key: 'conditional_value_at_risk', label: 'Conditional VaR', kind: 'pct', invert: true },
  { key: 'kelly_criterion', label: 'Kelly Criterion', kind: 'pct' },
  { key: 'risk_of_ruin', label: 'Risk of Ruin', kind: 'pct', invert: true },
  { key: 'tail_ratio', label: 'Tail Ratio', kind: 'ratio' },
  { key: 'recovery_factor', label: 'Recovery Factor', kind: 'ratio' },
  { key: 'ulcer_index', label: 'Ulcer Index', kind: 'ratio', invert: true },
  { key: 'ulcer_performance_index', label: 'Ulcer Performance Index', kind: 'ratio' },
  { key: 'serenity_index', label: 'Serenity Index', kind: 'ratio' },
  { key: 'common_sense_ratio', label: 'Common Sense Ratio', kind: 'ratio' },
  { key: 'exposure', label: 'Exposure', kind: 'pct' },
  { key: 'consecutive_wins', label: 'Max Consecutive Wins', kind: 'plain' },
  { key: 'consecutive_losses', label: 'Max Consecutive Losses', kind: 'plain', invert: true },
  { key: 'skew', label: 'Skew', kind: 'ratio' },
  { key: 'kurtosis', label: 'Kurtosis', kind: 'ratio' },
];

function fmtMetric(value, kind) {
  if (value == null || Number.isNaN(value)) return '—';
  if (kind === 'pct') return `${(value * 100).toFixed(2)}%`;
  if (kind === 'plain') return `${value}`;
  return value.toFixed(2);
}

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

export default function BacktestDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);

  const load = useCallback(() => {
    api.get(`/api/backtests/${id}`)
      .then((res) => {
        setData(res.data);
        setError(null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  // Backtests now run as a background job (see routers/backtests.py) --
  // a freshly submitted request comes back 'pending', then flips to
  // 'running', then 'completed'/'failed'. Poll every 3s while it's still
  // in flight so this page updates itself without a manual refresh, and
  // stop the moment it reaches a final state.
  useEffect(() => {
    setLoading(true);
    load();
  }, [load]);

  useEffect(() => {
    if (!data) return undefined;
    if (data.status === 'pending' || data.status === 'running') {
      pollRef.current = setInterval(load, 3000);
      return () => clearInterval(pollRef.current);
    }
    return undefined;
  }, [data?.status, load]);

  const equitySeries = useMemo(
    () => (data?.equity_curve || []).map((p) => ({ ts: p.timestamp, balance: p.balance, label: new Date(p.timestamp).toLocaleDateString() })),
    [data]
  );

  const plots = data?.stats?.plots || {};
  const metrics = data?.stats?.metrics || {};
  const metricCards = METRIC_CARDS.filter((m) => metrics[m.key] !== undefined);
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
          showIcon
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load this backtest</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={backBtnStyle}>Retry</button>}
          style={{
            background: 'rgba(240, 70, 107, 0.08)',
            border: '1px solid rgba(240, 70, 107, 0.3)',
          }}
        />
      </div>
    );
  }

  if (!data) return null;

  const statusStyle = STATUS_META[data.status] || STATUS_META.pending;
  const sc = data.strategy_config || {};
  const cfg = data.backtest_config || {};
  const winLoss = data.win_loss;

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
            <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2, textTransform: 'uppercase' }}>
              {data.coin ? `${data.coin} · ` : ''}{data.exchange || ''} {data.exchange ? '·' : ''} Backtest #{data.backtest_id}
            </div>
          </div>
        </div>
      </div>

      {data.status === 'pending' && (
        <Alert
          type="info"
          showIcon
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Queued</span>}
          description={<span style={{ color: '#C9CDD3' }}>This backtest hasn't started running yet.</span>}
          style={{
            marginBottom: 20,
            background: 'rgba(61, 220, 151, 0.08)',
            border: '1px solid rgba(61, 220, 151, 0.25)',
          }}
        />
      )}
      {data.status === 'running' && (
        <Alert
          type="warning"
          showIcon
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Running</span>}
          description={<span style={{ color: '#C9CDD3' }}>Pulling data and running the backtest engine — this page refreshes automatically.</span>}
          style={{
            marginBottom: 20,
            background: 'rgba(255, 138, 92, 0.08)',
            border: '1px solid rgba(255, 138, 92, 0.3)',
          }}
        />
      )}
      {data.status === 'failed' && (
        <Alert
          type="error"
          showIcon
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Backtest failed</span>}
          description={<span style={{ color: '#C9CDD3' }}>{data.error || 'Unknown error.'}</span>}
          style={{
            marginBottom: 20,
            background: 'rgba(240, 70, 107, 0.08)',
            border: '1px solid rgba(240, 70, 107, 0.3)',
          }}
        />
      )}

      {/* Backtest configuration */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginBottom: 20 }}>
        <Panel title="Strategy Configuration">
          <KeyValue label="Symbol" value={data.coin?.toUpperCase() || '—'} />
          <KeyValue label="Timeframe" value={data.time_horizon || '—'} />
          <KeyValue label="Entry Logic (Long)" value={sc.entry_logic_long || 'No long rule'} />
          <KeyValue label="Entry Logic (Short)" value={sc.entry_logic_short || 'No short rule'} />
          <KeyValue label="Indicators" value={sc.indicators?.length ? sc.indicators.join(', ') : '—'} />
        </Panel>
        <Panel title="Backtest Settings">
          <KeyValue label="Date Range" value={`${cfg.start_date || '—'} → ${cfg.end_date || '—'}`} />
          <KeyValue label="Initial Capital" value={fmtUsd(cfg.initial_balance)} />
          <KeyValue label="Position Size" value={cfg.position_size ? `${cfg.position_size.value}% (${cfg.position_size.type})` : '—'} />
          <KeyValue label="Max Open Positions" value={cfg.max_open_positions ?? '—'} />
        </Panel>
        <Panel title="Costs & Risk">
          <KeyValue label="Commission" value={cfg.commission != null ? `${cfg.commission}%` : '—'} />
          <KeyValue label="Slippage" value={cfg.slippage != null ? `${cfg.slippage}%` : '—'} />
          <KeyValue label="Take Profit" value={cfg.take_profit ? `${cfg.take_profit.value}${cfg.take_profit.type === 'percentage' ? '%' : ''}` : '—'} />
          <KeyValue label="Stop Loss" value={cfg.stop_loss ? `${cfg.stop_loss.value}${cfg.stop_loss.type === 'percentage' ? '%' : ''}` : '—'} />
          <KeyValue label="Long / Short Allowed" value={`${cfg.allow_long ? 'Long' : ''}${cfg.allow_long && cfg.allow_short ? ' / ' : ''}${cfg.allow_short ? 'Short' : ''}` || '—'} />
        </Panel>
      </div>

      {data.status === 'completed' && (
        <>
          {/* Complete statistics strip */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
            <StatBox label="Final Balance" value={fmtUsd(data.final_balance)} />
            <StatBox label="Total Net Profit" value={`${data.total_net_profit >= 0 ? '+' : ''}${fmtUsd(data.total_net_profit)}`} positive={data.total_net_profit >= 0} />
            <StatBox label="Total Trades" value={data.total_trades ?? 0} />
            <StatBox label="Win Rate" value={winLoss && data.total_trades ? `${((winLoss.wins / data.total_trades) * 100).toFixed(1)}%` : '—'} />
            <StatBox label="Wins / Losses" value={winLoss ? `${winLoss.wins} / ${winLoss.losses}` : '—'} />
          </div>

          {/* Performance metrics -- pulled from data.stats.metrics, the
              quantstats dict compute_stats() computed and stored against
              this backtest run. */}
          {metricCards.length > 0 && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
              {metricCards.map((m) => {
                const value = metrics[m.key];
                const positive = value == null ? undefined : m.invert ? value <= 0 : value >= 0;
                return (
                  <StatBox key={m.key} label={m.label} value={fmtMetric(value, m.kind)} positive={positive} />
                );
              })}
            </div>
          )}

          {/* Equity Curve + Drawdown */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <Panel title="Equity Curve">
              {equitySeries.length > 1 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={equitySeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="btEqGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={MINT} stopOpacity={0.35} />
                        <stop offset="95%" stopColor={MINT} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={['auto', 'auto']} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Area type="monotone" dataKey="balance" stroke={MINT} strokeWidth={2.5} fill="url(#btEqGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart text="Not enough data to plot an equity curve." />
              )}
            </Panel>
            <Panel title="Drawdown">
              {drawdownSeries.length > 1 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <AreaChart data={drawdownSeries} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="btDdGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={RED} stopOpacity={0.35} />
                        <stop offset="95%" stopColor={RED} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                    <Tooltip contentStyle={tooltipStyle} formatter={(v) => `${(v * 100).toFixed(2)}%`} />
                    <Area type="monotone" dataKey="dd" stroke={RED} strokeWidth={2} fill="url(#btDdGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <EmptyChart text={data.stats ? 'No drawdown periods.' : 'Not enough trade history for stats.'} />
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
                <EmptyChart text="Not enough history for a rolling Sharpe window." />
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
                <EmptyChart text="Not enough history for rolling volatility." />
              )}
            </Panel>
          </div>

          {/* Yearly Returns + Trade Distribution */}
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
            <Panel title="Trade Distribution (Per-Trade PnL)">
              <TradePnlChart trades={data.trades} />
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

          {/* Trade List */}
          <div style={{ marginBottom: 8 }}>
            <Panel title="Trade List">
              <Table
                size="small"
                pagination={{ pageSize: 10 }}
                rowKey={(r, i) => `${r.entry_time}-${i}`}
                dataSource={data.trades || []}
                locale={{ emptyText: 'No trades for this run.' }}
                columns={[
                  { title: 'Direction', dataIndex: 'direction', key: 'direction', render: (v) => <span style={{ color: v === 'long' ? MINT : RED, fontWeight: 600, textTransform: 'capitalize' }}>{v}</span> },
                  { title: 'Entry Time', dataIndex: 'entry_time', key: 'entry_time', render: (v) => new Date(v).toLocaleString() },
                  { title: 'Entry Price', dataIndex: 'entry_price', key: 'entry_price', render: (v) => v?.toLocaleString() },
                  { title: 'Qty', dataIndex: 'quantity', key: 'quantity' },
                  { title: 'Exit Time', dataIndex: 'exit_time', key: 'exit_time', render: (v) => (v ? new Date(v).toLocaleString() : '—') },
                  { title: 'Exit Price', dataIndex: 'exit_price', key: 'exit_price', render: (v) => (v != null ? v.toLocaleString() : '—') },
                  {
                    title: 'Net PnL', dataIndex: 'net_pnl', key: 'net_pnl',
                    render: (v) => v == null ? '—' : <span style={{ color: v >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}</span>,
                  },
                  { title: 'Exit Reason', dataIndex: 'exit_reason', key: 'exit_reason', render: (v) => v || '—' },
                ]}
              />
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}

const backBtnStyle = {
  width: 36, height: 36, borderRadius: 10, border: '1px solid rgba(255,255,255,0.1)',
  background: 'rgba(255,255,255,0.04)', color: '#F5F6F7', cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
};