import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table, Spin, Alert } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { api } from '../lib/api';
import { fmtUsd } from '../lib/format';
import { recordsToSeries, monthlyHeatmapToRows, heatColor } from '../lib/quantstats';
import { METRIC_CARDS_COL_1, METRIC_CARDS_COL_2, fmtMetric } from '../lib/metricCards';
import Panel, { panelFlat as panel } from '../components/Panel';
import EmptyChart from '../components/EmptyChart';
import TradePnlChart from '../components/TradePnlChart';
import KeyValue from '../components/KeyValue';
import StatBox from '../components/StatBox';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';

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
  const metricCardsCol1 = METRIC_CARDS_COL_1.filter((m) => metrics[m.key] !== undefined);
  const metricCardsCol2 = METRIC_CARDS_COL_2.filter((m) => metrics[m.key] !== undefined);
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
              this backtest run. Split into two equal-length cards instead
              of one big wall of tiles. */}
          {(metricCardsCol1.length > 0 || metricCardsCol2.length > 0) && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
              {metricCardsCol1.length > 0 && (
                <Panel title="Performance Metrics">
                  {metricCardsCol1.map((m) => {
                    const value = metrics[m.key];
                    const positive = value == null ? undefined : m.invert ? value <= 0 : value >= 0;
                    return (
                      <KeyValue
                        key={m.key}
                        label={m.label}
                        value={fmtMetric(value, m.kind)}
                        mono
                        color={positive === undefined ? undefined : positive ? MINT : RED}
                      />
                    );
                  })}
                </Panel>
              )}
              {metricCardsCol2.length > 0 && (
                <Panel title="Performance Metrics (cont.)">
                  {metricCardsCol2.map((m) => {
                    const value = metrics[m.key];
                    const positive = value == null ? undefined : m.invert ? value <= 0 : value >= 0;
                    return (
                      <KeyValue
                        key={m.key}
                        label={m.label}
                        value={fmtMetric(value, m.kind)}
                        mono
                        color={positive === undefined ? undefined : positive ? MINT : RED}
                      />
                    );
                  })}
                </Panel>
              )}
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
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
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
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${(v * 100).toFixed(2)}%`} />
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
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
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
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
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
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v.toFixed(2)}%`} />
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