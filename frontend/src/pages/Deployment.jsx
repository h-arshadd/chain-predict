import { useState, useEffect, useCallback, useMemo } from 'react';
import { Table, Tag, Input, Select, Spin, Alert, Tooltip } from 'antd';
import { useNavigate } from 'react-router-dom';
import { SearchOutlined, WarningFilled } from '@ant-design/icons';
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

// Backend statuses: "running" | "paused" | "unassigned" | "never_run"
const STATUS_META = {
  running: { label: 'Running', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  paused: { label: 'Paused', bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  unassigned: { label: 'Unassigned', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  never_run: { label: 'Never Run', bg: 'rgba(255,255,255,0.06)', fg: '#6B7280' },
};

const statusFilterOptions = [
  { value: 'All', label: 'All statuses' },
  ...Object.entries(STATUS_META).map(([value, m]) => ({ value, label: m.label })),
];

const fmtUsd = (v) =>
  v == null ? '—' : `${v >= 0 ? '' : '-'}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const pnlColor = (v) => (v == null ? '#6B7280' : v > 0 ? MINT : v < 0 ? RED : '#9096A0');

function buildColumns(navigate) {
  return [
    {
      title: 'Strategy', dataIndex: 'strategy_name', key: 'strategy_name',
      sorter: (a, b) => a.strategy_name.localeCompare(b.strategy_name),
      render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span>,
    },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (t) => <span style={{ color: '#9096A0', textTransform: 'uppercase' }}>{t}</span> },
    { title: 'Exchange', dataIndex: 'exchange', key: 'exchange', render: (t) => <span style={{ color: '#9096A0', textTransform: 'capitalize' }}>{t}</span> },
    {
      title: 'Wallet', dataIndex: 'account_name', key: 'account_name',
      render: (t, row) =>
        t ? (
          <span style={{ color: '#9096A0', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
            {t}
            {row.wallet_enabled === false && (
              <Tooltip title="Wallet disabled — this execution is paused">
                <WarningFilled style={{ color: AMBER, fontSize: 12 }} />
              </Tooltip>
            )}
          </span>
        ) : (
          <span style={{ color: '#6B7280' }}>No wallet assigned</span>
        ),
    },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      filters: Object.entries(STATUS_META).map(([value, m]) => ({ text: m.label, value })),
      onFilter: (value, record) => record.status === value,
      render: (status, record) => {
        const m = STATUS_META[status] || STATUS_META.never_run;
        if (status === 'unassigned') {
          return (
            <span
              onClick={(e) => {
                e.stopPropagation();
                navigate(`/strategies?coin=${record.symbol}`);
              }}
              style={{ cursor: 'pointer' }}
              title="No strategy is enabled for this pair yet -- click to go enable one"
            >
              <Tag style={{ background: m.bg, color: m.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
                {m.label} &rarr; enable a strategy
              </Tag>
            </span>
          );
        }
        return <Tag style={{ background: m.bg, color: m.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>{m.label}</Tag>;
      },
    },
    {
      title: 'Position', dataIndex: 'position', key: 'position',
      render: (position) =>
        position ? (
          <span style={{ color: position.direction === 'long' ? MINT : RED, fontWeight: 600, fontSize: 13, textTransform: 'capitalize' }}>
            {position.direction} {position.quantity ?? ''}
          </span>
        ) : (
          <span style={{ color: '#6B7280', fontSize: 13 }}>Flat</span>
        ),
    },
    {
      title: 'Current PnL', dataIndex: 'cumulative_pnl', key: 'cumulative_pnl',
      sorter: (a, b) => (a.cumulative_pnl ?? -Infinity) - (b.cumulative_pnl ?? -Infinity),
      render: (v) => (
        <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {v == null ? '—' : `${v >= 0 ? '+' : ''}${fmtUsd(v)}`}
        </span>
      ),
    },
    {
      title: 'Daily Return', dataIndex: 'daily_return_pct', key: 'daily_return_pct',
      sorter: (a, b) => (a.daily_return_pct ?? -Infinity) - (b.daily_return_pct ?? -Infinity),
      render: (v) => (
        <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`}
        </span>
      ),
    },
    { title: 'Last Signal', dataIndex: 'last_signal', key: 'last_signal', render: (t) => <span style={{ color: '#9096A0', fontSize: 13 }}>{t || '—'}</span> },
    {
      title: 'Last Execution', dataIndex: 'last_processed', key: 'last_processed',
      render: (t) => <span style={{ color: '#6B7280', fontSize: 13 }}>{t ? new Date(t).toLocaleString() : '—'}</span>,
    },
  ];
}

export default function Deployment() {
  const navigate = useNavigate();
  const [rows, setRows] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('All');
  const [coinFilter, setCoinFilter] = useState('All');

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.get('/api/executions?limit=200')
      .then((res) => setRows(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const coinOptions = useMemo(() => {
    const symbols = [...new Set(rows.map((r) => r.symbol))].sort();
    return [{ value: 'All', label: 'All coins' }, ...symbols.map((s) => ({ value: s, label: s.toUpperCase() }))];
  }, [rows]);

  const filtered = useMemo(() => {
    return rows.filter((r) => {
      const matchesSearch =
        r.strategy_name.toLowerCase().includes(search.toLowerCase()) ||
        r.symbol.toLowerCase().includes(search.toLowerCase()) ||
        (r.account_name || '').toLowerCase().includes(search.toLowerCase());
      const matchesStatus = statusFilter === 'All' || r.status === statusFilter;
      const matchesCoin = coinFilter === 'All' || r.symbol === coinFilter;
      return matchesSearch && matchesStatus && matchesCoin;
    });
  }, [rows, search, statusFilter, coinFilter]);

  const runningCount = rows.filter((r) => r.status === 'running').length;
  const totalPnl = rows.reduce((s, r) => s + (r.cumulative_pnl ?? 0), 0);
  const returnsWithValue = rows.filter((r) => r.daily_return_pct != null);
  const avgDailyReturn = returnsWithValue.length
    ? returnsWithValue.reduce((s, r) => s + r.daily_return_pct, 0) / returnsWithValue.length
    : null;

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Strategy Deployment</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          Live executions across all connected wallets. Select a row for full execution details.
        </p>
      </div>

      {error && (
        <Alert
          type="error"
          message="Couldn't load deployments"
          description={error}
          action={<button onClick={load} style={iconBtnStyle}>Retry</button>}
          style={{ marginBottom: 20 }}
          showIcon
        />
      )}

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
        <SummaryCard label="Running Deployments" value={`${runningCount} / ${rows.length}`} />
        <SummaryCard label="Combined Current PnL" value={`${totalPnl >= 0 ? '+' : ''}${fmtUsd(totalPnl)}`} color={pnlColor(totalPnl)} />
        <SummaryCard label="Avg Daily Return" value={avgDailyReturn == null ? '—' : `${avgDailyReturn >= 0 ? '+' : ''}${avgDailyReturn.toFixed(2)}%`} color={avgDailyReturn == null ? undefined : pnlColor(avgDailyReturn)} />
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 20 }}>
        <Input
          placeholder="Search by strategy, symbol, or wallet"
          prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 300, borderRadius: 999 }}
        />
        <Select value={statusFilter} onChange={setStatusFilter} options={statusFilterOptions} style={{ width: 160 }} />
        <Select value={coinFilter} onChange={setCoinFilter} options={coinOptions} style={{ width: 160 }} />
      </div>

      {/* Table */}
      <div style={{ ...panel, padding: 20 }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '60px 0' }}>
            <Spin size="large" />
          </div>
        ) : (
          <Table
            rowKey={(row) => `${row.exchange}-${row.symbol}`}
            columns={buildColumns(navigate)}
            dataSource={filtered}
            pagination={{ pageSize: 10 }}
            locale={{ emptyText: 'No deployments configured yet. Add a pair to execution.config to see it here.' }}
            onRow={(row) => ({
              onClick: () => navigate(`/deployment/${row.exchange}/${row.symbol}`),
              style: { cursor: 'pointer' },
            })}
          />
        )}
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

const iconBtnStyle = {
  display: 'flex', alignItems: 'center', justifyContent: 'center',
  width: 30, height: 30, borderRadius: 8, border: '1px solid rgba(255,255,255,0.08)',
  background: 'rgba(255,255,255,0.03)', color: '#9096A0', cursor: 'pointer',
};