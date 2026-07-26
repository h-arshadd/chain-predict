import { useState, useEffect, useCallback, useMemo } from 'react';
import { Table, Input, Select, Spin, Alert, Tooltip } from 'antd';
import { LineChart, Line, ResponsiveContainer, YAxis } from 'recharts';
import { useNavigate } from 'react-router-dom';
import {
  FundOutlined,
  ThunderboltOutlined,
  PlayCircleOutlined,
  RocketOutlined,
  WalletOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  InfoCircleOutlined,
} from '@ant-design/icons';
import { SearchOutlined } from '@ant-design/icons';
import { api } from '../lib/api';

const MINT = '#3DDC97';
const RED = '#F0466B';

const panel = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

const badgeAccents = ['#3DDC97', '#FF8A5C', '#5B9CF6', '#3DDC97', '#FF8A5C', '#5B9CF6', '#3DDC97', '#FF8A5C'];

const iconBadge = (color) => ({
  width: 36, height: 36, borderRadius: 10,
  background: `${color}20`,
  color,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontSize: 16, marginBottom: 14,
});

const fmtMoney = (v) => (v == null ? '—' : `${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
const fmtPct = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`);
const fmtCount = (v) => (v == null ? '—' : v);
const pnlColor = (v) => (v == null ? '#F5F6F7' : v > 0 ? MINT : v < 0 ? RED : '#F5F6F7');

// Same real pair_status values strategies_repo._pair_status() / the
// Strategies page use -- kept consistent across both pages.
const STATUS_META = {
  live: { label: 'Live', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  disabled: { label: 'Disabled', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  conflicted: { label: 'Conflicted', bg: 'rgba(240,70,107,0.14)', fg: RED },
};

function SectionHeader({ title, subtitle }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <h3 style={{ fontSize: 19, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{title}</h3>
      {subtitle && <p style={{ color: '#9096A0', fontSize: 13, margin: '4px 0 0' }}>{subtitle}</p>}
    </div>
  );
}

// Small per-row PnL sparkline for the strategy table -- series is
// [{t, v}], v = % return vs initial_balance (see
// strategies_repo._pnl_series_from_equity). Color follows the series'
// own net direction (last point vs first), not latest_return_pct, so a
// strategy that's currently up but was net negative over this window
// still reads as red -- matches what the line itself shows.
function PnlSparkline({ series }) {
  if (!series || series.length < 2) {
    return <span style={{ color: '#6B7280', fontSize: 12 }}>—</span>;
  }
  const trendUp = series[series.length - 1].v >= series[0].v;
  const color = trendUp ? MINT : RED;
  return (
    <div style={{ width: 100, height: 28 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={series} margin={{ top: 2, right: 2, left: 2, bottom: 2 }}>
          <YAxis hide domain={['auto', 'auto']} />
          <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.75} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const iconBtnStyle = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  width: 30, height: 30, borderRadius: 8, border: '1px solid rgba(255,255,255,0.08)',
  background: 'rgba(255,255,255,0.03)', color: '#9096A0', cursor: 'pointer',
};

export default function Dashboard() {
  const navigate = useNavigate();

  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState(null);

  const [strategies, setStrategies] = useState([]);
  const [tableLoading, setTableLoading] = useState(true);
  const [tableError, setTableError] = useState(null);

  const [search, setSearch] = useState('');
  const [coinFilter, setCoinFilter] = useState('All');

  const loadSummary = useCallback(() => {
    setSummaryLoading(true);
    setSummaryError(null);
    api.get('/api/dashboard')
      .then((res) => setSummary(res.data))
      .catch((err) => setSummaryError(err.message))
      .finally(() => setSummaryLoading(false));
  }, []);

  const loadStrategies = useCallback(() => {
    setTableLoading(true);
    setTableError(null);
    api.get('/api/dashboard/strategies?limit=500')
      .then((res) => setStrategies(res.data))
      .catch((err) => setTableError(err.message))
      .finally(() => setTableLoading(false));
  }, []);

  useEffect(() => {
    loadSummary();
    loadStrategies();
  }, [loadSummary, loadStrategies]);

  const coinOptions = useMemo(() => [
    { value: 'All', label: 'All coins' },
    ...[...new Set(strategies.map((s) => s.coin))].sort().map((c) => ({ value: c, label: c.toUpperCase() })),
  ], [strategies]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return strategies.filter((s) => {
      const matchesSearch = !q || s.strategy_name.toLowerCase().includes(q);
      const matchesCoin = coinFilter === 'All' || s.coin === coinFilter;
      return matchesSearch && matchesCoin;
    });
  }, [strategies, search, coinFilter]);

  // Exactly the 10 widgets the spec asks for. Every value is real
  // (execution or simulator data) -- None renders as "—" / "N/A", never
  // a fabricated number. total_backtests stays None until a real
  // Backtests module/DB exists.
  const statCards = summary ? [
    { title: 'Total Strategies', value: fmtCount(summary.total_strategies), icon: <FundOutlined /> },
    { title: 'Active Strategies', value: fmtCount(summary.active_strategies), icon: <ThunderboltOutlined /> },
    { title: 'Running Executions', value: fmtCount(summary.running_executions), icon: <RocketOutlined /> },
    { title: 'Running Simulations', value: fmtCount(summary.running_simulations), icon: <PlayCircleOutlined /> },
    { title: 'Connected Accounts', value: fmtCount(summary.connected_accounts), icon: <WalletOutlined /> },
    { title: 'Trained ML Models', value: fmtCount(summary.trained_ml_models), icon: <ExperimentOutlined /> },
    {
      title: `Total Backtests${summary.total_backtests == null ? ' (not available yet)' : ''}`,
      value: summary.total_backtests ?? 'N/A', icon: <BarChartOutlined />,
      valueColor: summary.total_backtests == null ? '#6B7280' : '#F5F6F7',
    },
    {
      title: "Today's PnL",
      value: fmtMoney(summary.today_pnl),
      valueColor: pnlColor(summary.today_pnl),
      icon: <BarChartOutlined />,
    },
    {
      title: 'Overall Portfolio Value',
      value: fmtMoney(summary.overall_portfolio_value),
      icon: <WalletOutlined />,
    },
    {
      title: 'Total Return',
      value: fmtPct(summary.total_return_pct),
      valueColor: pnlColor(summary.total_return_pct),
      icon: <BarChartOutlined />,
    },
  ] : [];

  const columns = [
    { title: 'Strategy Name', dataIndex: 'strategy_name', key: 'strategy_name', sorter: (a, b) => a.strategy_name.localeCompare(b.strategy_name), render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span> },
    { title: 'Symbol', dataIndex: 'coin', key: 'coin', render: (t) => <span style={{ color: '#9096A0' }}>{t.toUpperCase()}</span> },
    { title: 'Exchange', dataIndex: 'exchange', key: 'exchange', render: (t) => <span style={{ color: '#9096A0' }}>{t[0].toUpperCase() + t.slice(1)}</span> },
    { title: 'Timeframe', dataIndex: 'time_horizon', key: 'time_horizon', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    {
      title: 'Latest Return', dataIndex: 'latest_return_pct', key: 'latest_return_pct',
      sorter: (a, b) => (a.latest_return_pct ?? -Infinity) - (b.latest_return_pct ?? -Infinity),
      render: (v) => <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{fmtPct(v)}</span>,
    },
    {
      title: 'PnL',
      dataIndex: 'pnl_series',
      key: 'pnl_series',
      render: (series) => <PnlSparkline series={series} />,
    },
    {
      title: (
        <span>
          Sharpe Ratio{' '}
          <Tooltip title="Full Sharpe is computed on the Strategy Details page; not shown here to keep this list light.">
            <InfoCircleOutlined style={{ color: '#6B7280', fontSize: 11 }} />
          </Tooltip>
        </span>
      ),
      dataIndex: 'sharpe_ratio', key: 'sharpe_ratio',
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v == null ? '—' : v.toFixed(2)}</span>,
    },
    {
      title: 'Win Rate', dataIndex: 'win_rate_pct', key: 'win_rate_pct',
      sorter: (a, b) => (a.win_rate_pct ?? -Infinity) - (b.win_rate_pct ?? -Infinity),
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v == null ? '—' : `${v.toFixed(1)}%`}</span>,
    },
    {
      title: 'Current Status', dataIndex: 'pair_status', key: 'pair_status',
      filters: Object.keys(STATUS_META).map((k) => ({ text: STATUS_META[k].label, value: k })),
      onFilter: (value, record) => record.pair_status === value,
      render: (status) => {
        const meta = STATUS_META[status] || { label: status, bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' };
        return (
          <span style={{
            fontSize: 12, fontWeight: 600, padding: '3px 10px', borderRadius: 8,
            background: meta.bg, color: meta.fg,
          }}>
            {meta.label}
          </span>
        );
      },
    },
  ];

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Dashboard</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          System-wide overview of strategies, live execution, and simulation.
        </p>
      </div>

      {summaryError && (
        <Alert
          type="error"
          message="Couldn't load dashboard summary"
          description={summaryError}
          action={<button onClick={loadSummary} style={iconBtnStyle}>Retry</button>}
          style={{ marginBottom: 20 }}
          showIcon
        />
      )}

      {summaryLoading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '40px 0' }}>
          <Spin size="large" />
        </div>
      ) : summary ? (
        // One flat grid, exactly the 10 spec'd widgets.
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 28 }}>
          {statCards.map((stat, i) => (
            <div key={stat.title} style={{ ...panel, padding: 18 }}>
              <div style={iconBadge(badgeAccents[i % badgeAccents.length])}>{stat.icon}</div>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'ui-monospace, monospace', color: stat.valueColor || '#F5F6F7' }}>
                {stat.value}
              </div>
              <div style={{ color: '#9096A0', fontSize: 12.5, marginTop: 4 }}>
                {stat.title}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {/* Strategies table -- same real execution-first-fallback-to-
          simulator numbers as the Strategies page (see
          dashboard_repo.list_strategies), so this table and Strategies
          never disagree. */}
      <div style={{ ...panel, padding: 20 }}>
        <SectionHeader
          title="Strategies"
          subtitle="Every registered strategy, with real performance from execution (live) or simulator, whichever it has actually run in. Select one to view full details."
        />

        {tableError && (
          <Alert
            type="error"
            message="Couldn't load strategies"
            description={tableError}
            action={<button onClick={loadStrategies} style={iconBtnStyle}>Retry</button>}
            style={{ marginBottom: 16 }}
            showIcon
          />
        )}

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <Input
            placeholder="Search by strategy name"
            prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            style={{ maxWidth: 280, borderRadius: 999 }}
          />
          <Select value={coinFilter} onChange={setCoinFilter} options={coinOptions} style={{ width: 160 }} />
        </div>

        {tableLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '60px 0' }}>
            <Spin size="large" />
          </div>
        ) : (
          <Table
            columns={columns}
            dataSource={filtered.map((s) => ({ ...s, key: s.strategy_id }))}
            pagination={{ pageSize: 10 }}
            locale={{ emptyText: 'No strategies configured yet.' }}
            onRow={(row) => ({
              onClick: () => navigate(`/strategies/${row.strategy_id}`),
              style: { cursor: 'pointer' },
            })}
          />
        )}
      </div>
    </div>
  );
}