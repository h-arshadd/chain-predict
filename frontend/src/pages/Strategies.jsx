import { useState, useEffect, useCallback } from 'react';
import { Table, Tag, Input, Select, Spin, Alert, Tooltip } from 'antd';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { SearchOutlined } from '@ant-design/icons';
import { api } from '../lib/api';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const panel = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

// Real pair_status values from the backend (strategies_repo._pair_status):
//   "live"        - this strategy is THE one execution runs for its pair
//   "disabled"    - execution_enabled is off for this row (paused)
//   "conflicted"  - this row + a sibling are both enabled for the same
//                   pair, so execution/main.py treats the pair as
//                   misconfigured and skips it entirely (nobody's live)
const STATUS_META = {
  live: { label: 'Live', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  disabled: { label: 'Disabled', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  conflicted: { label: 'Conflicted', bg: 'rgba(240,70,107,0.14)', fg: RED },
};

const fmtPct = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`);
const pnlColor = (v) => (v == null ? '#6B7280' : v > 0 ? MINT : v < 0 ? RED : '#9096A0');

export default function Strategies() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [strategies, setStrategies] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Deep-link support: Deployment/ExecutionDetails' "no strategy enabled
  // for this pair" link lands here with ?coin= pre-filled, so the person
  // sees exactly the pair that needs a strategy enabled instead of the
  // full unfiltered list.
  const [search, setSearch] = useState('');
  const [coinFilter, setCoinFilter] = useState(searchParams.get('coin') || 'All');

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.get('/api/strategies?limit=500')
      .then((res) => setStrategies(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const coinOptions = [
    { value: 'All', label: 'All coins' },
    ...[...new Set(strategies.map((s) => s.coin))].sort().map((c) => ({ value: c, label: c.toUpperCase() })),
  ];

  const filtered = strategies.filter((s) => {
    const q = search.toLowerCase();
    const matchesSearch = s.strategy_name.toLowerCase().includes(q);
    const matchesCoin = coinFilter === 'All' || s.coin === coinFilter;
    return matchesSearch && matchesCoin;
  });

  const columns = [
    {
      title: 'Strategy Name', dataIndex: 'strategy_name', key: 'strategy_name',
      sorter: (a, b) => a.strategy_name.localeCompare(b.strategy_name),
      render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span>,
    },
    { title: 'Symbol', dataIndex: 'coin', key: 'coin', render: (t) => <span style={{ color: '#9096A0' }}>{t.toUpperCase()}</span> },
    { title: 'Timeframe', dataIndex: 'time_horizon', key: 'time_horizon', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    {
      title: 'Current Status', dataIndex: 'pair_status', key: 'pair_status',
      filters: [
        { text: 'Live', value: 'live' },
        { text: 'Disabled', value: 'disabled' },
        { text: 'Conflicted', value: 'conflicted' },
      ],
      onFilter: (value, record) => record.pair_status === value,
      render: (status) => {
        const c = STATUS_META[status] || STATUS_META.disabled;
        const tag = (
          <Tag style={{ background: c.bg, color: c.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
            {c.label}
          </Tag>
        );
        return status === 'conflicted' ? (
          <Tooltip title="Another strategy on this pair is also enabled — execution skips this pair entirely until only one is enabled.">
            {tag}
          </Tooltip>
        ) : tag;
      },
    },
    {
      title: 'Latest Return', dataIndex: 'latest_return_pct', key: 'latest_return_pct',
      sorter: (a, b) => (a.latest_return_pct ?? -Infinity) - (b.latest_return_pct ?? -Infinity),
      render: (v) => (
        <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {fmtPct(v)}
        </span>
      ),
    },
    {
      title: 'Sharpe Ratio', dataIndex: 'sharpe_ratio', key: 'sharpe_ratio',
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v == null ? '—' : v.toFixed(2)}</span>,
    },
    {
      title: 'Win Rate', dataIndex: 'win_rate_pct', key: 'win_rate_pct',
      sorter: (a, b) => (a.win_rate_pct ?? -Infinity) - (b.win_rate_pct ?? -Infinity),
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{v == null ? '—' : `${v.toFixed(1)}%`}</span>,
    },
  ];

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Strategies</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            All configured strategies. Only one strategy can be live for execution per coin — use the switch to change which one runs.
          </p>
        </div>
      </div>

      {error && (
        <Alert
          type="error"
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load strategies</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={iconBtnStyle}>Retry</button>}
          style={{ marginBottom: 20, background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
          showIcon
        />
      )}

      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
        <Input
          placeholder="Search by strategy name"
          prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 280, borderRadius: 999 }}
        />
        <Select value={coinFilter} onChange={setCoinFilter} options={coinOptions} style={{ width: 160 }} />
      </div>

      <div style={{ ...panel, padding: 20 }}>
        {loading ? (
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

const iconBtnStyle = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  width: 30, height: 30, borderRadius: 8, border: '1px solid rgba(255,255,255,0.08)',
  background: 'rgba(255,255,255,0.03)', color: '#9096A0', cursor: 'pointer',
};