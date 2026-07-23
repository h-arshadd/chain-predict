import { useState, useMemo } from 'react';
import { Table, Tag, Input, Select } from 'antd';
import { useNavigate } from 'react-router-dom';
import { SearchOutlined, PlusOutlined } from '@ant-design/icons';
import { ResponsiveContainer, LineChart, Line } from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';

// ---- placeholder data — replace with GET /api/strategies once backend exists ----
const strategyData = [
  { key: '1', id: 1, name: 'BTC Momentum', symbol: 'BTCUSDT', exchange: 'Bybit', timeframe: '4h', status: 'Active', return: 12.4, sharpe: 1.8, winRate: 61 },
  { key: '2', id: 2, name: 'ETH Mean Reversion', symbol: 'ETHUSDT', exchange: 'Binance', timeframe: '1h', status: 'Active', return: -3.2, sharpe: 0.6, winRate: 47 },
  { key: '3', id: 3, name: 'SOL Breakout', symbol: 'SOLUSDT', exchange: 'Bybit', timeframe: '15m', status: 'Paused', return: 8.1, sharpe: 1.3, winRate: 55 },
  { key: '4', id: 4, name: 'ADA Trend Follow', symbol: 'ADAUSDT', exchange: 'Binance', timeframe: '1d', status: 'Stopped', return: -1.1, sharpe: 0.2, winRate: 44 },
  { key: '5', id: 5, name: 'XRP Scalper', symbol: 'XRPUSDT', exchange: 'Bybit', timeframe: '5m', status: 'Active', return: 5.7, sharpe: 1.1, winRate: 58 },
  { key: '6', id: 6, name: 'DOGE Volatility Break', symbol: 'DOGEUSDT', exchange: 'Binance', timeframe: '1h', status: 'Paused', return: -6.4, sharpe: -0.2, winRate: 39 },
  { key: '7', id: 7, name: 'MATIC Grid', symbol: 'MATICUSDT', exchange: 'Bybit', timeframe: '4h', status: 'Active', return: 3.3, sharpe: 0.9, winRate: 52 },
  { key: '8', id: 8, name: 'LINK Trend Rider', symbol: 'LINKUSDT', exchange: 'Binance', timeframe: '1d', status: 'Stopped', return: 1.8, sharpe: 0.4, winRate: 49 },
];

const sparkline = (seed) =>
  Array.from({ length: 8 }, (_, i) => ({
    x: i,
    v: seed + Math.sin(i + seed) * 8 + i * (seed > 0 ? 1.5 : -1.2),
  }));

const panel = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

const statusColors = {
  Active: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  Paused: { bg: 'rgba(255,138,92,0.14)', fg: '#FF8A5C' },
  Stopped: { bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
};

const statusFilterOptions = [
  { value: 'All', label: 'All statuses' },
  { value: 'Active', label: 'Active' },
  { value: 'Paused', label: 'Paused' },
  { value: 'Stopped', label: 'Stopped' },
];

const exchangeFilterOptions = [
  { value: 'All', label: 'All exchanges' },
  { value: 'Bybit', label: 'Bybit' },
  { value: 'Binance', label: 'Binance' },
];

function buildColumns() {
  return [
    {
      title: 'Strategy Name', dataIndex: 'name', key: 'name',
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span>,
    },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Exchange', dataIndex: 'exchange', key: 'exchange', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Timeframe', dataIndex: 'timeframe', key: 'timeframe', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    {
      title: 'Current Status', dataIndex: 'status', key: 'status',
      filters: [
        { text: 'Active', value: 'Active' },
        { text: 'Paused', value: 'Paused' },
        { text: 'Stopped', value: 'Stopped' },
      ],
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
    {
      title: 'Latest Return', dataIndex: 'return', key: 'return',
      sorter: (a, b) => a.return - b.return,
      render: (val) => (
        <span style={{ color: val >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {val >= 0 ? '+' : ''}{val}%
        </span>
      ),
    },
    {
      title: 'Sharpe Ratio', dataIndex: 'sharpe', key: 'sharpe',
      sorter: (a, b) => a.sharpe - b.sharpe,
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v}</span>,
    },
    {
      title: 'Win Rate', dataIndex: 'winRate', key: 'winRate',
      sorter: (a, b) => a.winRate - b.winRate,
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v}%</span>,
    },
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

export default function Strategies() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [exchangeFilter, setExchangeFilter] = useState('All');

  const filtered = useMemo(() => {
    return strategyData.filter((s) => {
      const matchesSearch =
        s.name.toLowerCase().includes(search.toLowerCase()) ||
        s.symbol.toLowerCase().includes(search.toLowerCase());
      const matchesStatus = statusFilter === 'All' || s.status === statusFilter;
      const matchesExchange = exchangeFilter === 'All' || s.exchange === exchangeFilter;
      return matchesSearch && matchesStatus && matchesExchange;
    });
  }, [search, statusFilter, exchangeFilter]);

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Strategies</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            All configured strategies across exchanges. Select one to view full details.
          </p>
        </div>
        <button
          style={{
            display: 'flex', alignItems: 'center', gap: 8,
            background: MINT, color: '#0B0E11', border: 'none',
            fontSize: 14, fontWeight: 700, padding: '10px 18px',
            borderRadius: 999, cursor: 'pointer',
          }}
        >
          <PlusOutlined /> New Strategy
        </button>
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
        <Input
          placeholder="Search by name or symbol"
          prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            maxWidth: 280, borderRadius: 999,
            background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)',
            padding: '8px 16px',
          }}
        />
        <Select
          value={statusFilter}
          onChange={setStatusFilter}
          options={statusFilterOptions}
          style={{ width: 160 }}
        />
        <Select
          value={exchangeFilter}
          onChange={setExchangeFilter}
          options={exchangeFilterOptions}
          style={{ width: 160 }}
        />
      </div>

      {/* Table */}
      <div style={{ ...panel, padding: 20 }}>
        <Table
          columns={buildColumns()}
          dataSource={filtered}
          pagination={{ pageSize: 8 }}
          locale={{ emptyText: 'No strategies match your filters.' }}
          onRow={(row) => ({
            onClick: () => navigate(`/strategies/${row.id}`),
            style: { cursor: 'pointer' },
          })}
        />
      </div>
    </div>
  );
}