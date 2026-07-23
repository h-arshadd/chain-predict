import { Row, Col, Card, Table, Tag } from 'antd';
import {
  ArrowUpOutlined,
  ThunderboltOutlined,
  RocketOutlined,
  WalletOutlined,
  ExperimentOutlined,
  FundOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import {
  AreaChart, Area, ResponsiveContainer, Tooltip, XAxis,
  LineChart, Line,
} from 'recharts';

const portfolioTrend = [
  { day: 'Mon', value: 8100 }, { day: 'Tue', value: 8300 },
  { day: 'Wed', value: 8150 }, { day: 'Thu', value: 8420 },
  { day: 'Fri', value: 8390 }, { day: 'Sat', value: 8600 },
  { day: 'Sun', value: 8780 },
];

const sparkline = (seed) =>
  Array.from({ length: 8 }, (_, i) => ({
    x: i,
    v: seed + Math.sin(i + seed) * 8 + i * (seed > 0 ? 1.5 : -1.2),
  }));

const statCards = [
  { title: 'Total Strategies', value: 24, icon: <FundOutlined /> },
  { title: 'Active Strategies', value: 9, icon: <ThunderboltOutlined /> },
  { title: 'Running Executions', value: 6, icon: <RocketOutlined /> },
  { title: 'Running Simulations', value: 3, icon: <PlayCircleOutlined /> },
  { title: 'Connected Accounts', value: 2, icon: <WalletOutlined /> },
  { title: 'Trained ML Models', value: 12, icon: <ExperimentOutlined /> },
  { title: 'Total Backtests', value: 87, icon: <ExperimentOutlined /> },
  { title: 'Total Return', value: '+18.4%', icon: <ArrowUpOutlined /> },
];

const strategyColumns = [
  { title: 'Strategy Name', dataIndex: 'name', key: 'name', render: (t) => <span style={{ fontWeight: 600 }}>{t}</span> },
  { title: 'Symbol', dataIndex: 'symbol', key: 'symbol' },
  { title: 'Exchange', dataIndex: 'exchange', key: 'exchange' },
  { title: 'Timeframe', dataIndex: 'timeframe', key: 'timeframe' },
  {
    title: 'Status', dataIndex: 'status', key: 'status',
    render: (status) => <Tag color={status === 'Active' ? 'success' : 'default'}>{status}</Tag>,
  },
  {
    title: 'Latest Return', dataIndex: 'return', key: 'return',
    render: (val) => (
      <span style={{ color: val >= 0 ? '#12B76A' : '#F04438', fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
        {val >= 0 ? '+' : ''}{val}%
      </span>
    ),
  },
  { title: 'Sharpe Ratio', dataIndex: 'sharpe', key: 'sharpe', render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace' }}>{v}</span> },
  { title: 'Win Rate', dataIndex: 'winRate', key: 'winRate', render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace' }}>{v}%</span> },
  {
    title: 'Trend', key: 'trend',
    render: (_, row) => (
      <ResponsiveContainer width={90} height={32}>
        <LineChart data={sparkline(row.return)}>
          <Line
            type="monotone" dataKey="v" stroke={row.return >= 0 ? '#12B76A' : '#F04438'}
            strokeWidth={2} dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    ),
  },
];

const strategyData = [
  { key: 1, name: 'BTC Momentum', symbol: 'BTCUSDT', exchange: 'Bybit', timeframe: '4h', status: 'Active', return: 12.4, sharpe: 1.8, winRate: 61 },
  { key: 2, name: 'ETH Mean Reversion', symbol: 'ETHUSDT', exchange: 'Binance', timeframe: '1h', status: 'Active', return: -3.2, sharpe: 0.6, winRate: 47 },
  { key: 3, name: 'SOL Breakout', symbol: 'SOLUSDT', exchange: 'Bybit', timeframe: '15m', status: 'Paused', return: 8.1, sharpe: 1.3, winRate: 55 },
];

const cardStyle = {
  borderRadius: 18,
  border: '1px solid #ECEBEF',
  boxShadow: '0 2px 8px rgba(108, 92, 231, 0.06)',
};

const iconBadge = {
  width: 36, height: 36, borderRadius: 10,
  background: 'rgba(108, 92, 231, 0.1)', color: '#6C5CE7',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  fontSize: 16, marginBottom: 14,
};

export default function Dashboard() {
  return (
    <div>
      <h2 style={{ marginBottom: 24, fontWeight: 700, fontSize: 24, color: '#1D2129' }}>
        Dashboard
      </h2>

      {/* Hero cards */}
      <Row gutter={20} style={{ marginBottom: 20 }}>
        <Col xs={24} md={8}>
          <Card
            style={{
              ...cardStyle,
              background: 'linear-gradient(135deg, rgba(108,92,231,0.06) 0%, rgba(108,92,231,0.01) 100%)',
            }}
            bodyStyle={{ padding: 24 }}
          >
            <div style={{ color: '#6B7280', fontSize: 13, marginBottom: 10, fontWeight: 500 }}>
              Overall Portfolio Value
            </div>
            <div style={{ fontSize: 34, fontWeight: 700, fontFamily: 'ui-monospace, monospace', letterSpacing: -0.5 }}>
              $8,780.42
            </div>
            <div style={{
              color: '#12B76A', fontSize: 13, marginTop: 8, fontWeight: 600,
              background: 'rgba(18, 183, 106, 0.1)', display: 'inline-block',
              padding: '3px 10px', borderRadius: 20,
            }}>
              <ArrowUpOutlined /> 1.32% today
            </div>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card style={cardStyle} bodyStyle={{ padding: 24 }}>
            <div style={{ color: '#6B7280', fontSize: 13, marginBottom: 10, fontWeight: 500 }}>
              Today's PnL
            </div>
            <div style={{ fontSize: 34, fontWeight: 700, fontFamily: 'ui-monospace, monospace', color: '#12B76A', letterSpacing: -0.5 }}>
              +$114.20
            </div>
            <div style={{ color: '#6B7280', fontSize: 13, marginTop: 10 }}>
              Across 6 live executions
            </div>
          </Card>
        </Col>
        <Col xs={24} md={8}>
          <Card style={cardStyle} bodyStyle={{ padding: '18px 16px 8px' }}>
            <div style={{ color: '#6B7280', fontSize: 13, marginBottom: 4, paddingLeft: 8, fontWeight: 500 }}>
              Portfolio Trend (7D)
            </div>
            <ResponsiveContainer width="100%" height={104}>
              <AreaChart data={portfolioTrend} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#6C5CE7" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#6C5CE7" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" hide />
                <Tooltip
                  contentStyle={{ borderRadius: 10, border: '1px solid #ECEBEF', fontSize: 12 }}
                />
                <Area
                  type="monotone" dataKey="value" stroke="#6C5CE7"
                  strokeWidth={2.5} fill="url(#colorValue)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Compact stat grid */}
      <Row gutter={[16, 16]} style={{ marginBottom: 20 }}>
        {statCards.map((stat) => (
          <Col xs={12} sm={8} md={6} lg={6} xl={3} key={stat.title}>
            <Card style={cardStyle} bodyStyle={{ padding: 18 }}>
              <div style={iconBadge}>{stat.icon}</div>
              <div style={{ fontSize: 22, fontWeight: 700, fontFamily: 'ui-monospace, monospace', color: '#1D2129' }}>
                {stat.value}
              </div>
              <div style={{ color: '#6B7280', fontSize: 12.5, marginTop: 4 }}>
                {stat.title}
              </div>
            </Card>
          </Col>
        ))}
      </Row>

      {/* Strategies table */}
      <Card style={cardStyle} bodyStyle={{ padding: '8px 8px 16px' }} title="Strategies">
        <Table
          columns={strategyColumns}
          dataSource={strategyData}
          pagination={false}
        />
      </Card>
    </div>
  );
}