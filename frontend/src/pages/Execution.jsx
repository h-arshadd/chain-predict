import { useState, useEffect, useCallback, useMemo } from 'react';
import { Table, Tag, Input, Select, Spin, Alert, Tooltip } from 'antd';
import { useNavigate } from 'react-router-dom';
import { SearchOutlined, WarningFilled } from '@ant-design/icons';
import { api } from '../lib/api';
import { fmtUsd, pnlColor } from '../lib/format';
import { panelGradient as panel } from '../components/Panel';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';



// Backend statuses: "running" | "flat" | "paused" | "unassigned" | "never_run"
const STATUS_META = {
  running: { label: 'Running', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  flat: { label: 'Flat', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  paused: { label: 'Paused', bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  unassigned: { label: 'Unassigned', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  never_run: { label: 'Never Run', bg: 'rgba(255,255,255,0.06)', fg: '#6B7280' },
};

const statusFilterOptions = [
  { value: 'All', label: 'All statuses' },
  ...Object.entries(STATUS_META).map(([value, m]) => ({ value, label: m.label })),
];

function buildColumns() {
  return [
    {
      title: 'Strategy', dataIndex: 'strategy_name', key: 'strategy_name',
      sorter: (a, b) => a.strategy_name.localeCompare(b.strategy_name),
      width: 220,
      render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7', whiteSpace: 'normal', wordBreak: 'break-word' }}>{t}</span>,
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
      title: 'Current PnL', dataIndex: 'unrealized_pnl', key: 'unrealized_pnl',
      sorter: (a, b) => (a.unrealized_pnl ?? -Infinity) - (b.unrealized_pnl ?? -Infinity),
      render: (v) => (
        <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {v == null ? '—' : `${v >= 0 ? '+' : ''}${fmtUsd(v)}`}
        </span>
      ),
    },
    {
      title: 'Cumulative PnL', dataIndex: 'cumulative_pnl', key: 'cumulative_pnl',
      sorter: (a, b) => (a.cumulative_pnl ?? -Infinity) - (b.cumulative_pnl ?? -Infinity),
      render: (v) => (
        <span style={{ color: pnlColor(v), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
          {v == null ? '—' : `${v >= 0 ? '+' : ''}${fmtUsd(v)}`}
        </span>
      ),
    },
    {
      title: 'Last Signal', dataIndex: 'position', key: 'last_signal',
      render: (position) =>
        position ? (
          <span style={{ color: position.direction === 'long' ? MINT : RED, fontWeight: 600, fontSize: 13, textTransform: 'capitalize' }}>
            {position.direction}
          </span>
        ) : (
          <span style={{ color: '#9096A0', fontSize: 13 }}>—</span>
        ),
    },
    {
      title: 'Last Execution', dataIndex: 'last_processed', key: 'last_processed',
      render: (t) => <span style={{ color: '#6B7280', fontSize: 13 }}>{t ? new Date(t).toLocaleString() : '—'}</span>,
    },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      render: (status) => {
        const m = STATUS_META[status] || STATUS_META.never_run;
        return <Tag style={{ background: m.bg, color: m.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>{m.label}</Tag>;
      },
    },
    {
      title: 'Trade Status', dataIndex: 'position', key: 'trade_status',
      render: (position) =>
        position ? (
          <Tag style={{ background: 'rgba(61,220,151,0.12)', color: MINT, border: 'none', borderRadius: 8, fontWeight: 600 }}>
            Open
          </Tag>
        ) : (
          <Tag style={{ background: 'rgba(255,255,255,0.06)', color: '#9096A0', border: 'none', borderRadius: 8, fontWeight: 600 }}>
            Closed
          </Tag>
        ),
    },
  ];
}

export default function Execution() {
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
  const totalUnrealizedPnl = rows.reduce((s, r) => s + (r.unrealized_pnl ?? 0), 0);

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Execution</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          Live executions across all connected wallets. Select a row for full execution details.
        </p>
      </div>

      {error && (
        <Alert
          type="error"
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load executions</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={iconBtnStyle}>Retry</button>}
          style={{ marginBottom: 20, background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
          showIcon
        />
      )}

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 24 }}>
        <SummaryCard label="Running Executions" value={`${runningCount} / ${rows.length}`} />
        <SummaryCard label="Combined Cumulative PnL" value={`${totalPnl >= 0 ? '+' : ''}${fmtUsd(totalPnl)}`} color={pnlColor(totalPnl)} />
        <SummaryCard label="Combined Current PnL" value={`${totalUnrealizedPnl >= 0 ? '+' : ''}${fmtUsd(totalUnrealizedPnl)}`} color={pnlColor(totalUnrealizedPnl)} />
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
            columns={buildColumns()}
            dataSource={filtered}
            pagination={{ pageSize: 10 }}
            locale={{ emptyText: 'No executions configured yet. Add a pair to execution.config to see it here.' }}
            onRow={(row) => ({
              onClick: () => navigate(`/execution/${row.exchange}/${row.symbol}`),
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