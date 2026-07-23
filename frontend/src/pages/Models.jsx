import { useState, useMemo } from 'react';
import { Table, Tag, Input, Select } from 'antd';
import { useNavigate } from 'react-router-dom';
import { SearchOutlined, ExperimentOutlined } from '@ant-design/icons';
import { ResponsiveContainer, LineChart, Line } from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const COINS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'LTC', 'MINA', 'SUI'];

// ---- placeholder data — replace with GET /api/models once backend exists ----
const modelData = [
  { key: '1', id: 1, name: 'BTC-4h-LSTM-v3', type: 'LSTM', symbol: 'BTCUSDT', timeframe: '4h', trainingDate: '2026-07-18', primaryMetric: 'Accuracy', score: 68.4, status: 'Deployed' },
  { key: '2', id: 2, name: 'ETH-1h-XGB-v5', type: 'XGBoost', symbol: 'ETHUSDT', timeframe: '1h', trainingDate: '2026-07-15', primaryMetric: 'F1 Score', score: 0.71, status: 'Deployed' },
  { key: '3', id: 3, name: 'SOL-15m-RF-v2', type: 'Random Forest', symbol: 'SOLUSDT', timeframe: '15m', trainingDate: '2026-07-10', primaryMetric: 'Accuracy', score: 61.2, status: 'Archived' },
  { key: '4', id: 4, name: 'ADA-1d-Transformer-v1', type: 'Transformer', symbol: 'ADAUSDT', timeframe: '1d', trainingDate: '2026-07-20', primaryMetric: 'Sharpe (sim)', score: 1.4, status: 'Training' },
  { key: '5', id: 5, name: 'DOGE-1h-LSTM-v2', type: 'LSTM', symbol: 'DOGEUSDT', timeframe: '1h', trainingDate: '2026-07-05', primaryMetric: 'Accuracy', score: 57.9, status: 'Deployed' },
  { key: '6', id: 6, name: 'LTC-4h-XGB-v1', type: 'XGBoost', symbol: 'LTCUSDT', timeframe: '4h', trainingDate: '2026-06-28', primaryMetric: 'F1 Score', score: 0.63, status: 'Archived' },
  { key: '7', id: 7, name: 'MINA-4h-RF-v1', type: 'Random Forest', symbol: 'MINAUSDT', timeframe: '4h', trainingDate: '2026-07-21', primaryMetric: 'Accuracy', score: 64.7, status: 'Deployed' },
  { key: '8', id: 8, name: 'SUI-5m-LSTM-v1', type: 'LSTM', symbol: 'SUIUSDT', timeframe: '5m', trainingDate: '2026-07-22', primaryMetric: 'Accuracy', score: 59.3, status: 'Training' },
];

const sparkline = (seed) =>
  Array.from({ length: 8 }, (_, i) => ({
    x: i,
    v: seed + Math.sin(i + seed) * 4 + i * 0.8,
  }));

const panel = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

const statusColors = {
  Deployed: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  Training: { bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  Archived: { bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
};

const typeOptions = [
  { value: 'All', label: 'All model types' },
  { value: 'LSTM', label: 'LSTM' },
  { value: 'XGBoost', label: 'XGBoost' },
  { value: 'Random Forest', label: 'Random Forest' },
  { value: 'Transformer', label: 'Transformer' },
];

const coinFilterOptions = [
  { value: 'All', label: 'All coins' },
  ...COINS.map((c) => ({ value: c, label: c })),
];

function buildColumns() {
  return [
    {
      title: 'Model Name', dataIndex: 'name', key: 'name',
      sorter: (a, b) => a.name.localeCompare(b.name),
      render: (t) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(61,220,151,0.12)', color: MINT, flexShrink: 0,
          }}>
            <ExperimentOutlined style={{ fontSize: 14 }} />
          </div>
          <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span>
        </div>
      ),
    },
    { title: 'Model Type', dataIndex: 'type', key: 'type', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Timeframe', dataIndex: 'timeframe', key: 'timeframe', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    {
      title: 'Training Date', dataIndex: 'trainingDate', key: 'trainingDate',
      sorter: (a, b) => a.trainingDate.localeCompare(b.trainingDate),
      render: (t) => <span style={{ color: '#6B7280', fontSize: 13 }}>{t}</span>,
    },
    { title: 'Primary Metric', dataIndex: 'primaryMetric', key: 'primaryMetric', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    {
      title: 'Score', dataIndex: 'score', key: 'score',
      sorter: (a, b) => a.score - b.score,
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7', fontWeight: 600 }}>{v}</span>,
    },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      filters: Object.keys(statusColors).map((s) => ({ text: s, value: s })),
      onFilter: (value, record) => record.status === value,
      render: (status) => {
        const c = statusColors[status] || statusColors.Archived;
        return (
          <Tag style={{ background: c.bg, color: c.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
            {status}
          </Tag>
        );
      },
    },
    {
      title: 'Trend', key: 'trend',
      render: (_, row) => (
        <ResponsiveContainer width={90} height={32}>
          <LineChart data={sparkline(row.score)}>
            <Line type="monotone" dataKey="v" stroke={MINT} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      ),
    },
  ];
}

export default function Models() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [typeFilter, setTypeFilter] = useState('All');
  const [coinFilter, setCoinFilter] = useState('All');

  const filtered = useMemo(() => {
    return modelData.filter((m) => {
      const matchesSearch = m.name.toLowerCase().includes(search.toLowerCase());
      const matchesType = typeFilter === 'All' || m.type === typeFilter;
      const matchesCoin = coinFilter === 'All' || m.symbol.startsWith(coinFilter);
      return matchesSearch && matchesType && matchesCoin;
    });
  }, [search, typeFilter, coinFilter]);

  const deployedCount = modelData.filter((m) => m.status === 'Deployed').length;
  const trainingCount = modelData.filter((m) => m.status === 'Training').length;

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Machine Learning</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          All trained predictive models across coins. Select one to inspect dataset, training, and evaluation details.
        </p>
      </div>

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
        <SummaryCard label="Total Models" value={modelData.length} />
        <SummaryCard label="Deployed" value={deployedCount} color={MINT} />
        <SummaryCard label="Training" value={trainingCount} color={AMBER} />
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
        <Input
          placeholder="Search by model name"
          prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{
            maxWidth: 280, borderRadius: 999,
            background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.07)',
            padding: '8px 16px',
          }}
        />
        <Select value={typeFilter} onChange={setTypeFilter} options={typeOptions} style={{ width: 180 }} />
        <Select value={coinFilter} onChange={setCoinFilter} options={coinFilterOptions} style={{ width: 160 }} />
      </div>

      {/* Table */}
      <div style={{ ...panel, padding: 20 }}>
        <Table
          columns={buildColumns()}
          dataSource={filtered}
          pagination={{ pageSize: 8 }}
          locale={{ emptyText: 'No models match your filters.' }}
          onRow={(row) => ({
            onClick: () => navigate(`/models/${row.id}`),
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