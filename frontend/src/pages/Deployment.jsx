import { useState, useMemo } from 'react';
import { Table, Tag, Input, Select } from 'antd';
import { useNavigate } from 'react-router-dom';
import { SearchOutlined } from '@ant-design/icons';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const COINS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'LTC', 'MINA', 'SUI'];

// ---- placeholder data — replace with GET /api/executions once backend exists ----
const deploymentData = [
  { key: '1', id: 1, strategy: 'BTC Momentum', symbol: 'BTCUSDT', exchange: 'Bybit', wallet: 'Main Trading', status: 'Running', position: 'Long 0.42 BTC', pnl: 562.8, dailyReturn: 1.9, lastSignal: 'Buy @ 61,240', lastExecution: '2m ago' },
  { key: '2', id: 2, strategy: 'SOL Breakout', symbol: 'SOLUSDT', exchange: 'Bybit', wallet: 'Main Trading', status: 'Running', position: 'Short 18 SOL', pnl: 37.8, dailyReturn: 0.4, lastSignal: 'Sell @ 148.20', lastExecution: '11m ago' },
  { key: '3', id: 3, strategy: 'ADA Trend Follow', symbol: 'ADAUSDT', exchange: 'Bybit', wallet: 'Altcoin Sub-Account', status: 'Running', position: 'Long 3200 ADA', pnl: -41.6, dailyReturn: -0.5, lastSignal: 'Hold', lastExecution: '38m ago' },
  { key: '4', id: 4, strategy: 'DOGE Volatility Break', symbol: 'DOGEUSDT', exchange: 'Bybit', wallet: 'Altcoin Sub-Account', status: 'Paused', position: 'Flat', pnl: 0, dailyReturn: 0, lastSignal: 'Wait', lastExecution: '3h ago' },
  { key: '5', id: 5, strategy: 'MINA Grid', symbol: 'MINAUSDT', exchange: 'Bybit', wallet: 'MINA/SUI Desk', status: 'Running', position: 'Long 1200 MINA', pnl: 30.0, dailyReturn: 0.6, lastSignal: 'Buy @ 0.712', lastExecution: '5m ago' },
  { key: '6', id: 6, strategy: 'SUI Scalper', symbol: 'SUIUSDT', exchange: 'Bybit', wallet: 'MINA/SUI Desk', status: 'Running', position: 'Long 340 SUI', pnl: 23.8, dailyReturn: 1.1, lastSignal: 'Buy @ 3.82', lastExecution: '1m ago' },
  { key: '7', id: 7, strategy: 'LTC Swing', symbol: 'LTCUSDT', exchange: 'Bybit', wallet: 'Main Trading', status: 'Stopped', position: 'Flat', pnl: -12.4, dailyReturn: 0, lastSignal: 'Stop-loss hit', lastExecution: '1d ago' },
  { key: '8', id: 8, strategy: 'ETH Mean Reversion', symbol: 'ETHUSDT', exchange: 'Bybit', wallet: 'Main Trading', status: 'Error', position: 'Flat', pnl: 0, dailyReturn: 0, lastSignal: 'API error', lastExecution: '6h ago' },
];

const panel = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

const statusColors = {
  Running: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  Paused: { bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  Stopped: { bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  Error: { bg: 'rgba(240,70,107,0.14)', fg: RED },
};

const statusFilterOptions = [
  { value: 'All', label: 'All statuses' },
  { value: 'Running', label: 'Running' },
  { value: 'Paused', label: 'Paused' },
  { value: 'Stopped', label: 'Stopped' },
  { value: 'Error', label: 'Error' },
];

const coinFilterOptions = [
  { value: 'All', label: 'All coins' },
  ...COINS.map((c) => ({ value: c, label: c })),
];

function buildColumns() {
  return [
    {
      title: 'Strategy', dataIndex: 'strategy', key: 'strategy',
      sorter: (a, b) => a.strategy.localeCompare(b.strategy),
      render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span>,
    },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Exchange', dataIndex: 'exchange', key: 'exchange', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Wallet', dataIndex: 'wallet', key: 'wallet', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      filters: Object.keys(statusColors).map((s) => ({ text: s, value: s })),
      onFilter: (value, record) => record.status === value,
      render: (status) => {
        const c = statusColors[status] || statusColors.Stopped;
        return (
          <Tag style={{ background: c.bg, color: c.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
            {status}
          </Tag>
        );
      },
    },
    { title: 'Position', dataIndex: 'position', key: 'position', render: (t) => <span style={{ color: '#F5F6F7', fontSize: 13 }}>{t}</span> },
    {
      title: 'Current PnL', dataIndex: 'pnl', key: 'pnl',
      sorter: (a, b) => a.pnl - b.pnl,
      render: (v) => (
        <span style={{ color: v > 0 ? MINT : v < 0 ? RED : '#9096A0', fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {v > 0 ? '+' : ''}{v.toFixed(2)}
        </span>
      ),
    },
    {
      title: 'Daily Return', dataIndex: 'dailyReturn', key: 'dailyReturn',
      sorter: (a, b) => a.dailyReturn - b.dailyReturn,
      render: (v) => (
        <span style={{ color: v > 0 ? MINT : v < 0 ? RED : '#9096A0', fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {v > 0 ? '+' : ''}{v}%
        </span>
      ),
    },
    { title: 'Last Signal', dataIndex: 'lastSignal', key: 'lastSignal', render: (t) => <span style={{ color: '#9096A0', fontSize: 13 }}>{t}</span> },
    { title: 'Last Execution', dataIndex: 'lastExecution', key: 'lastExecution', render: (t) => <span style={{ color: '#6B7280', fontSize: 13 }}>{t}</span> },
  ];
}

export default function Deployment() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [coinFilter, setCoinFilter] = useState('All');

  const filtered = useMemo(() => {
    return deploymentData.filter((d) => {
      const matchesSearch =
        d.strategy.toLowerCase().includes(search.toLowerCase()) ||
        d.symbol.toLowerCase().includes(search.toLowerCase()) ||
        d.wallet.toLowerCase().includes(search.toLowerCase());
      const matchesStatus = statusFilter === 'All' || d.status === statusFilter;
      const matchesCoin = coinFilter === 'All' || d.symbol.startsWith(coinFilter);
      return matchesSearch && matchesStatus && matchesCoin;
    });
  }, [search, statusFilter, coinFilter]);

  const runningCount = deploymentData.filter((d) => d.status === 'Running').length;
  const totalPnl = deploymentData.reduce((s, d) => s + d.pnl, 0);
  const avgDailyReturn = deploymentData.reduce((s, d) => s + d.dailyReturn, 0) / deploymentData.length;

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Strategy Deployment</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          Live executions across all connected Bybit wallets. Select a row for full execution details.
        </p>
      </div>

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
        <SummaryCard label="Running Deployments" value={`${runningCount} / ${deploymentData.length}`} />
        <SummaryCard label="Combined Current PnL" value={`${totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}`} color={totalPnl >= 0 ? MINT : RED} />
        <SummaryCard label="Avg Daily Return" value={`${avgDailyReturn >= 0 ? '+' : ''}${avgDailyReturn.toFixed(2)}%`} color={avgDailyReturn >= 0 ? MINT : RED} />
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
        <Input
          placeholder="Search by strategy, symbol, or wallet"
          prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            maxWidth: 300, borderRadius: 999,
            background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)',
            padding: '8px 16px',
          }}
        />
        <Select value={statusFilter} onChange={setStatusFilter} options={statusFilterOptions} style={{ width: 160 }} />
        <Select value={coinFilter} onChange={setCoinFilter} options={coinFilterOptions} style={{ width: 160 }} />
      </div>

      {/* Table */}
      <div style={{ ...panel, padding: 20 }}>
        <Table
          columns={buildColumns()}
          dataSource={filtered}
          pagination={{ pageSize: 8 }}
          locale={{ emptyText: 'No deployments match your filters.' }}
          onRow={(row) => ({
            onClick: () => navigate(`/deployment/${row.id}`),
            style: { cursor: 'pointer' },
          })}
        />
      </div>
    </div>
  );
}

function SummaryCard({ label, value, color }) {
  return (
    <div style={{ ...panel, padding: '18px 20px' }}>
      <div style={{ fontSize: 12, color: '#6B7280', fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4 }}>
        {label}
      </div>
      <div style={{ fontSize: 22, fontWeight: 700, color: color || '#F5F6F7', marginTop: 6, fontFamily: 'ui-monospace, monospace' }}>
        {value}
      </div>
    </div>
  );
}