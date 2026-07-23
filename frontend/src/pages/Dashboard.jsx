import { Table, Tag } from 'antd';
import { useNavigate } from 'react-router-dom';
import {
  FundOutlined,
  ThunderboltOutlined,
  RocketOutlined,
  PlayCircleOutlined,
  WalletOutlined,
  ExperimentOutlined,
  BarChartOutlined,
  DollarOutlined,
  ArrowUpOutlined,
} from '@ant-design/icons';
import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area,
} from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';

// ---- placeholder data — replace with GET /api/dashboard once backend exists ----

// Hero: Overall Portfolio Value gets the big trend chart (like the reference layout's
// corner graph) instead of sitting flat in the stat grid.
const portfolioTrend = [
  { day: 'Jul 8', value: 7120 }, { day: 'Jul 10', value: 7340 },
  { day: 'Jul 12', value: 7180 }, { day: 'Jul 14', value: 7560 },
  { day: 'Jul 16', value: 7890 }, { day: 'Jul 18', value: 7710 },
  { day: 'Jul 20', value: 8040 }, { day: 'Jul 22', value: 8320 },
  { day: 'Jul 23', value: 8780 },
];

// Spec: Dashboard widgets — Total Strategies, Active Strategies, Running Executions,
// Running Simulation, Connected Accounts, Trained ML Models, Total Backtests,
// Today's PnL, Overall Portfolio Value, Total Return
// (Overall Portfolio Value is promoted to the hero chart above, not repeated here)
const statCards = [
  { title: "Today's PnL", value: '+$114.20', icon: <DollarOutlined />, highlight: true, positive: true },
  { title: 'Total Return', value: '+18.4%', icon: <ArrowUpOutlined />, highlight: true, positive: true },
  { title: 'Total Strategies', value: 24, icon: <FundOutlined /> },
  { title: 'Active Strategies', value: 9, icon: <ThunderboltOutlined /> },
  { title: 'Running Executions', value: 6, icon: <RocketOutlined /> },
  { title: 'Running Simulation', value: 3, icon: <PlayCircleOutlined /> },
  { title: 'Connected Accounts', value: 2, icon: <WalletOutlined /> },
  { title: 'Trained ML Models', value: 12, icon: <ExperimentOutlined /> },
  { title: 'Total Backtests', value: 87, icon: <BarChartOutlined /> },
];

const sparkline = (seed) =>
  Array.from({ length: 8 }, (_, i) => ({
    x: i,
    v: seed + Math.sin(i + seed) * 8 + i * (seed > 0 ? 1.5 : -1.2),
  }));

// Spec: strategies table — Strategy Name, Symbol, Exchange, Timeframe, Current Status,
// Latest Return, Sharpe Ratio, Win Rate. Selecting a strategy opens Strategy Details.
const strategyData = [
  { key: '1', id: 1, name: 'BTC Momentum', symbol: 'BTCUSDT', exchange: 'Bybit', timeframe: '4h', status: 'Active', return: 12.4, sharpe: 1.8, winRate: 61 },
  { key: '2', id: 2, name: 'ETH Mean Reversion', symbol: 'ETHUSDT', exchange: 'Binance', timeframe: '1h', status: 'Active', return: -3.2, sharpe: 0.6, winRate: 47 },
  { key: '3', id: 3, name: 'SOL Breakout', symbol: 'SOLUSDT', exchange: 'Bybit', timeframe: '15m', status: 'Paused', return: 8.1, sharpe: 1.3, winRate: 55 },
  { key: '4', id: 4, name: 'ADA Trend Follow', symbol: 'ADAUSDT', exchange: 'Binance', timeframe: '1d', status: 'Stopped', return: -1.1, sharpe: 0.2, winRate: 44 },
];

const panel = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

const heroPanel = {
  background: 'linear-gradient(155deg, rgba(61,220,151,0.16) 0%, rgba(19,23,27,0.85) 65%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(61,220,151,0.2)',
  borderRadius: 20,
  boxShadow: '0 0 40px -12px rgba(61,220,151,0.25)',
};

// Cycles warm/cool accents across the widget row so it reads like the reference's
// amber + mint + blue glow, not a single flat tone.
const badgeAccents = ['#3DDC97', '#FF8A5C', '#5B9CF6', '#3DDC97', '#FF8A5C', '#5B9CF6', '#3DDC97', '#FF8A5C', '#5B9CF6'];

const iconBadge = (color) => ({
  width: 36, height: 36, borderRadius: 10,
  background: `${color}20`,
  color,
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontSize: 16, marginBottom: 14,
});

function SectionHeader({ title }) {
  return (
    <div style={{ marginBottom: 18 }}>
      <h3 style={{ fontSize: 19, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{title}</h3>
    </div>
  );
}

const statusColors = {
  Active: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  Paused: { bg: 'rgba(255,138,92,0.14)', fg: '#FF8A5C' },
  Stopped: { bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
};

function buildColumns() {
  return [
    { title: 'Strategy Name', dataIndex: 'name', key: 'name', render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span> },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Exchange', dataIndex: 'exchange', key: 'exchange', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Timeframe', dataIndex: 'timeframe', key: 'timeframe', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    {
      title: 'Current Status', dataIndex: 'status', key: 'status',
      render: (status) => {
        const c = statusColors[status] || statusColors.Stopped;
        return (
          <Tag style={{ background: c.bg, color: c.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
            {status}
          </Tag>
        );
      },
    },
    {
      title: 'Latest Return', dataIndex: 'return', key: 'return',
      render: (val) => (
        <span style={{ color: val >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {val >= 0 ? '+' : ''}{val}%
        </span>
      ),
    },
    { title: 'Sharpe Ratio', dataIndex: 'sharpe', key: 'sharpe', render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v}</span> },
    { title: 'Win Rate', dataIndex: 'winRate', key: 'winRate', render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v}%</span> },
    {
      title: 'Trend', key: 'trend',
      render: (_, row) => (
        <ResponsiveContainer width={90} height={32}>
          <LineChart data={sparkline(row.return)}>
            <Line type="monotone" dataKey="v" stroke={row.return >= 0 ? MINT : RED} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      ),
    },
  ];
}

export default function Dashboard() {
  const navigate = useNavigate();

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Dashboard</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          System-wide overview of strategies, executions, and models.
        </p>
      </div>

      {/* Overall Portfolio Value — compact card with mini trend chart in the corner */}
      <div style={{ display: 'grid', gridTemplateColumns: '260px repeat(auto-fit, minmax(160px, 1fr))', gap: 16, marginBottom: 28 }}>
        <div style={{ ...heroPanel, padding: 18, display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ color: '#A8ADB8', fontSize: 12, fontWeight: 500, marginBottom: 8 }}>
              Overall Portfolio Value
            </div>
            <div style={{ fontSize: 22, fontWeight: 700, color: '#F5F6F7', fontFamily: 'ui-monospace, monospace' }}>
              $8,780.42
            </div>
            <span style={{
              color: MINT, fontSize: 11.5, fontWeight: 600, marginTop: 4,
              display: 'inline-flex', alignItems: 'center', gap: 4,
            }}>
              <ArrowUpOutlined style={{ fontSize: 9 }} /> +1.32% today
            </span>
          </div>

          <ResponsiveContainer width="100%" height={44}>
            <AreaChart data={portfolioTrend} margin={{ top: 4, right: 0, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="portfolioGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={MINT} stopOpacity={0.4} />
                  <stop offset="95%" stopColor={MINT} stopOpacity={0} />
                </linearGradient>
              </defs>
              <Area type="monotone" dataKey="value" stroke={MINT} strokeWidth={2} fill="url(#portfolioGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>


        {statCards.map((stat, i) => (
          <div key={stat.title} style={{ ...panel, padding: 18 }}>
            <div style={iconBadge(badgeAccents[i % badgeAccents.length])}>{stat.icon}</div>
            <div
              style={{
                fontSize: 22, fontWeight: 700, fontFamily: 'ui-monospace, monospace',
                color: stat.positive ? MINT : '#F5F6F7',
              }}
            >
              {stat.value}
            </div>
            <div style={{ color: '#9096A0', fontSize: 12.5, marginTop: 4 }}>
              {stat.title}
            </div>
          </div>
        ))}
      </div>

      {/* Strategies table — main section per spec, row click -> Strategy Details */}
      <div style={{ ...panel, padding: 20 }}>
        <SectionHeader title="Strategies" />
        <Table
          columns={buildColumns()}
          dataSource={strategyData}
          pagination={false}
          onRow={(row) => ({
            onClick: () => navigate(`/strategies/${row.id}`),
            style: { cursor: 'pointer' },
          })}
        />
      </div>
    </div>
  );
}