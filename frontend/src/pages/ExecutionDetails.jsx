import { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table, Spin, Alert } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { api } from '../lib/api';
import { fmtUsd } from '../lib/format';
import { recordsToSeries, monthlyHeatmapToRows, heatColor } from '../lib/quantstats';
import { METRIC_CARDS_COL_1, METRIC_CARDS_COL_2, fmtMetric } from '../lib/metricCards';
import Panel from '../components/Panel';
import EmptyChart from '../components/EmptyChart';
import TradePnlChart from '../components/TradePnlChart';
import EquityCurveChart from '../components/EquityCurveChart';
import DrawdownChart from '../components/DrawdownChart';
import RollingSharpeChart from '../components/RollingSharpeChart';
import RollingVolatilityChart from '../components/RollingVolatilityChart';
import YearlyReturnsChart from '../components/YearlyReturnsChart';
import KeyValue from '../components/KeyValue';
import StatBox from '../components/StatBox';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const STATUS_META = {
  running: { label: 'Running', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  paused: { label: 'Paused', bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  unassigned: { label: 'Unassigned', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  never_run: { label: "Didn't Start", bg: 'rgba(255,255,255,0.06)', fg: '#6B7280' },
};

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
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load this execution</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={backBtnStyle}>Retry</button>}
          showIcon
          style={{ background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
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
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h2 style={{ fontSize: 22, fontWeight: 700, color: '#F5F6F7', margin: 0, wordBreak: 'break-word' }}>{data.strategy_name}</h2>
            <Tag style={{ background: statusStyle.bg, color: statusStyle.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
              {statusStyle.label}
            </Tag>
          </div>
          <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2, textTransform: 'uppercase' }}>
            {data.symbol} &middot; {data.exchange} &middot; {data.account_name || 'No wallet assigned'}
          </div>
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

      {/* Performance metrics -- pulled from data.stats.metrics, the
          quantstats dict compute_stats() computed and stored for this
          execution's own equity curve. Split into two equal-length cards
          instead of one big wall of tiles. */}
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
          <EquityCurveChart series={equitySeries} gradientId="eqGrad2" emptyText="Not enough closed trades yet to plot an equity curve." />
        </Panel>
        <Panel title="Drawdown">
          <DrawdownChart series={drawdownSeries} gradientId="ddGrad" emptyText={data.stats ? 'No drawdown periods yet.' : 'Not enough trade history for stats yet.'} />
        </Panel>
      </div>

      {/* Rolling Sharpe + Rolling Volatility */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Rolling Sharpe">
          <RollingSharpeChart series={rollingSharpeSeries} emptyText="Not enough history for a rolling Sharpe window yet." />
        </Panel>
        <Panel title="Rolling Volatility">
          <RollingVolatilityChart series={rollingVolSeries} emptyText="Not enough history for rolling volatility yet." />
        </Panel>
      </div>

      {/* Yearly Returns + Trade PnL sequence */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Yearly Returns">
          <YearlyReturnsChart data={yearlyReturns} emptyText="Not enough history to compute yearly returns." />
        </Panel>
        <Panel title="Trade PnL Sequence">
          <TradePnlChart trades={data.trades} reverse />
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