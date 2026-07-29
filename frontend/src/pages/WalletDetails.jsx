import { useState, useEffect, useCallback, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Table, Spin, Alert } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from 'recharts';
import { api } from '../lib/api';
import { fmtUsd, pnlColor } from '../lib/format';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';
import Panel from '../components/Panel';
import StatBox from '../components/StatBox';

const MINT = '#3DDC97';
const RED = '#F0466B';
const BLUE = '#5B9CF6';

const DAY_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

function fmtNum(v, digits = 2) {
  if (v == null) return '—';
  return Number(v).toLocaleString('en-US', { maximumFractionDigits: digits });
}

function fmtDuration(seconds) {
  if (seconds == null) return '—';
  const s = Number(seconds);
  if (s < 60) return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s / 60).toFixed(1)}m`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h`;
  return `${(s / 86400).toFixed(1)}d`;
}

function parseIfJson(v) {
  if (typeof v !== 'string') return v;
  try {
    return JSON.parse(v);
  } catch {
    return v;
  }
}

const backBtnStyle = {
  display: 'flex', alignItems: 'center', gap: 8,
  background: 'rgba(255,255,255,0.04)', color: '#C9CDD3', border: '1px solid rgba(255,255,255,0.08)',
  fontSize: 13, fontWeight: 600, padding: '8px 14px', borderRadius: 10, cursor: 'pointer',
};

export default function WalletDetails() {
  const { accountName } = useParams();
  const navigate = useNavigate();

  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    setStats(null);
    api.get(`/api/wallets/${accountName}/stats`)
      .then((res) => setStats(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [accountName]);

  useEffect(() => {
    load();
  }, [load]);

  const perSymbol = useMemo(() => {
    if (!stats?.per_symbol) return [];
    const parsed = parseIfJson(stats.per_symbol) || {};
    return Object.entries(parsed).map(([symbol, v]) => ({ symbol, ...v }));
  }, [stats]);

  const hourData = useMemo(() => {
    if (!stats?.trades_by_hour_of_day) return [];
    const parsed = parseIfJson(stats.trades_by_hour_of_day) || {};
    return Array.from({ length: 24 }, (_, h) => ({ hour: `${h}:00`, count: parsed[h] ?? parsed[String(h)] ?? 0 }));
  }, [stats]);

  const dayData = useMemo(() => {
    if (!stats?.trades_by_day_of_week) return [];
    const parsed = parseIfJson(stats.trades_by_day_of_week) || {};
    return DAY_ORDER.map((day) => ({ day: day.slice(0, 3), count: parsed[day] ?? 0 }));
  }, [stats]);

  const winLossData = useMemo(() => {
    if (!stats) return [];
    const wins = stats.win_count ?? 0;
    const losses = stats.loss_count ?? 0;
    if (wins === 0 && losses === 0) return [];
    return [
      { name: 'Wins', value: wins, color: MINT },
      { name: 'Losses', value: losses, color: RED },
    ];
  }, [stats]);

  const longShortData = useMemo(() => {
    if (!stats) return [];
    const long = stats.long_trade_count ?? 0;
    const short = stats.short_trade_count ?? 0;
    if (long === 0 && short === 0) return [];
    return [
      { name: 'Long', value: long, color: BLUE },
      { name: 'Short', value: short, color: '#FF8A5C' },
    ];
  }, [stats]);

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
        <button onClick={() => navigate('/wallets')} style={backBtnStyle}>
          <ArrowLeftOutlined /> Back to Wallets
        </button>
        <Alert
          type="error"
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load wallet stats</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={backBtnStyle}>Retry</button>}
          showIcon
          style={{ marginTop: 16, background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
        />
      </div>
    );
  }

  if (!stats) {
    return (
      <div style={{ paddingTop: 8 }}>
        <button onClick={() => navigate('/wallets')} style={backBtnStyle}>
          <ArrowLeftOutlined /> Back to Wallets
        </button>
        <div style={{ color: '#6B7280', fontSize: 14, padding: '40px 2px' }}>
          No trading history yet for this wallet.
        </div>
      </div>
    );
  }

  return (
    <div style={{ paddingTop: 8 }}>
      <button onClick={() => navigate('/wallets')} style={{ ...backBtnStyle, marginBottom: 16 }}>
        <ArrowLeftOutlined /> Back to Wallets
      </button>

      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{accountName}</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          Full trading history and performance breakdown for this wallet.
        </p>
      </div>

      {/* Top-level PnL summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Net Realized PnL" value={fmtUsd(stats.net_realized_pnl)} positive={stats.net_realized_pnl > 0 ? true : stats.net_realized_pnl < 0 ? false : undefined} />
        <StatBox label="Gross Realized PnL" value={fmtUsd(stats.gross_realized_pnl)} positive={stats.gross_realized_pnl > 0 ? true : stats.gross_realized_pnl < 0 ? false : undefined} />
        <StatBox label="Win Rate" value={stats.win_rate_pct == null ? '—' : `${fmtNum(stats.win_rate_pct, 1)}%`} />
        <StatBox label="Profit Factor" value={fmtNum(stats.profit_factor)} />
        <StatBox label="Expectancy / Trade" value={fmtUsd(stats.expectancy)} positive={stats.expectancy > 0 ? true : stats.expectancy < 0 ? false : undefined} />
        <StatBox label="Max Drawdown" value={fmtUsd(stats.max_drawdown)} positive={false} />
      </div>

      {/* Win/Loss + Long/Short donuts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Win / Loss Breakdown">
          {winLossData.length === 0 ? (
            <EmptyState text="No closed trades yet." />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={winLossData} dataKey="value" nameKey="name" innerRadius="60%" outerRadius="85%" paddingAngle={2} stroke="none">
                  {winLossData.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v, n) => [`${v} trades`, n]} />
                <Legend verticalAlign="bottom" height={24} iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: '#9096A0' }} />
              </PieChart>
            </ResponsiveContainer>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 12 }}>
            <MiniStat label="Avg Win" value={fmtUsd(stats.avg_win)} color={MINT} />
            <MiniStat label="Avg Loss" value={fmtUsd(stats.avg_loss)} color={RED} />
            <MiniStat label="Largest Win" value={fmtUsd(stats.largest_win)} color={MINT} />
            <MiniStat label="Largest Loss" value={fmtUsd(stats.largest_loss)} color={RED} />
          </div>
        </Panel>

        <Panel title="Long / Short Split">
          {longShortData.length === 0 ? (
            <EmptyState text="No directional trade data yet." />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={longShortData} dataKey="value" nameKey="name" innerRadius="60%" outerRadius="85%" paddingAngle={2} stroke="none">
                  {longShortData.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v, n) => [`${v} trades`, n]} />
                <Legend verticalAlign="bottom" height={24} iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12, color: '#9096A0' }} />
              </PieChart>
            </ResponsiveContainer>
          )}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginTop: 12 }}>
            <MiniStat label="Long PnL" value={fmtUsd(stats.long_pnl)} color={pnlColor(stats.long_pnl)} />
            <MiniStat label="Short PnL" value={fmtUsd(stats.short_pnl)} color={pnlColor(stats.short_pnl)} />
            <MiniStat label="Long Win Rate" value={stats.long_win_rate_pct == null ? '—' : `${fmtNum(stats.long_win_rate_pct, 1)}%`} />
            <MiniStat label="Short Win Rate" value={stats.short_win_rate_pct == null ? '—' : `${fmtNum(stats.short_win_rate_pct, 1)}%`} />
          </div>
        </Panel>
      </div>

      {/* Streaks + holding time */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Max Win Streak" value={fmtNum(stats.max_win_streak, 0)} positive />
        <StatBox label="Max Loss Streak" value={fmtNum(stats.max_loss_streak, 0)} positive={false} />
        <StatBox
          label="Current Streak"
          value={stats.current_streak_len ? `${stats.current_streak_len} ${stats.current_streak_type}` : '—'}
          positive={stats.current_streak_type === 'win' ? true : stats.current_streak_type === 'loss' ? false : undefined}
        />
        <StatBox label="Avg Holding Time" value={fmtDuration(stats.avg_holding_time_sec)} />
        <StatBox label="Median Holding Time" value={fmtDuration(stats.median_holding_time_sec)} />
        <StatBox label="Trades / Day (avg)" value={fmtNum(stats.trades_per_day_avg)} />
      </div>

      {/* Activity charts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Trades by Hour of Day">
          {hourData.length === 0 ? (
            <EmptyState text="No trade timing data yet." />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={hourData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="hour" tick={axisStyle} axisLine={false} tickLine={false} interval={2} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip cursor={false} contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
                <Bar dataKey="count" fill={BLUE} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
          {stats.busiest_hour != null && (
            <div style={{ color: '#9096A0', fontSize: 12, marginTop: 10 }}>
              Busiest hour: <span style={{ color: '#F5F6F7', fontWeight: 600 }}>{stats.busiest_hour}:00</span>
            </div>
          )}
        </Panel>

        <Panel title="Trades by Day of Week">
          {dayData.length === 0 ? (
            <EmptyState text="No trade timing data yet." />
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={dayData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="day" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip cursor={false} contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
                <Bar dataKey="count" fill="#FF8A5C" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
          {stats.busiest_day_of_week && (
            <div style={{ color: '#9096A0', fontSize: 12, marginTop: 10 }}>
              Busiest day: <span style={{ color: '#F5F6F7', fontWeight: 600 }}>{stats.busiest_day_of_week}</span>
            </div>
          )}
        </Panel>
      </div>

      {/* Volume & fees */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Total Trades" value={fmtNum(stats.trade_row_count, 0)} />
        <StatBox label="Unique Symbols" value={fmtNum(stats.unique_symbols, 0)} />
        <StatBox label="Total Exec Value" value={fmtUsd(stats.total_exec_value)} />
        <StatBox label="Total Fees Paid" value={fmtUsd(stats.total_fees)} positive={false} />
        <StatBox label="Maker Ratio" value={stats.maker_ratio == null ? '—' : `${(stats.maker_ratio * 100).toFixed(1)}%`} />
        <StatBox label="Buy / Sell" value={`${fmtNum(stats.buy_count, 0)} / ${fmtNum(stats.sell_count, 0)}`} />
      </div>

      {/* Per-symbol breakdown */}
      {perSymbol.length > 0 && (
        <Panel title="Performance by Symbol" style={{ marginBottom: 20 }}>
          <Table
            size="small"
            pagination={false}
            rowKey="symbol"
            dataSource={perSymbol}
            locale={{ emptyText: 'No per-symbol data yet.' }}
            columns={[
              { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (v) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{v.toUpperCase()}</span> },
              { title: 'Trades', dataIndex: 'trade_count', key: 'trade_count', render: (v) => fmtNum(v, 0) },
              { title: 'Realized PnL', dataIndex: 'realized_pnl', key: 'realized_pnl', render: (v) => <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace' }}>{fmtUsd(v)}</span> },
              { title: 'Win Rate', dataIndex: 'win_rate_pct', key: 'win_rate_pct', render: (v) => (v == null ? '—' : `${fmtNum(v, 1)}%`) },
              { title: 'Total Qty', dataIndex: 'total_qty', key: 'total_qty', render: (v) => fmtNum(v, 4) },
              { title: 'Total Value', dataIndex: 'total_value', key: 'total_value', render: (v) => fmtUsd(v) },
              { title: 'Avg Price', dataIndex: 'avg_price', key: 'avg_price', render: (v) => fmtNum(v, 4) },
              { title: 'Fees', dataIndex: 'total_fees', key: 'total_fees', render: (v) => fmtUsd(v) },
            ]}
          />
        </Panel>
      )}

      {stats.first_trade_time && (
        <div style={{ color: '#6B7280', fontSize: 12, textAlign: 'right', marginTop: 4 }}>
          Trading since {new Date(stats.first_trade_time).toLocaleDateString()}
          {stats.last_trade_time && ` · Last trade ${new Date(stats.last_trade_time).toLocaleString()}`}
          {stats.updated_at && ` · Stats refreshed ${new Date(stats.updated_at).toLocaleString()}`}
        </div>
      )}
    </div>
  );
}

function EmptyState({ text }) {
  return (
    <div style={{ height: 220, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#6B7280', fontSize: 13 }}>
      {text}
    </div>
  );
}

function MiniStat({ label, value, color }) {
  return (
    <div style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: 12, padding: '10px 12px' }}>
      <div style={{ color: '#6B7280', fontSize: 11, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 700, color: color || '#F5F6F7', fontFamily: 'ui-monospace, monospace' }}>{value}</div>
    </div>
  );
}