import { useState, useEffect, useCallback, useMemo } from 'react';
import { Table, Input, Select, Spin, Alert } from 'antd';
import { useNavigate } from 'react-router-dom';
import { SearchOutlined, ExperimentOutlined } from '@ant-design/icons';
import { api } from '../lib/api';
import { panelGradient as panel } from '../components/Panel';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';



// Real model_type values from the backend (ml_repo.py's _SUPPORTED_KINDS
// filtering) -- "timeseries" runs are excluded entirely, never sent by
// the API, so there is no third option here.
const MODEL_TYPE_OPTIONS = [
  { value: 'All', label: 'All model types' },
  { value: 'regression', label: 'Regression' },
  { value: 'classification', label: 'Classification' },
];

const fmtMetric = (v, digits = 2) => (v == null ? '—' : v.toFixed(digits));
const fmtPct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

function algorithmLabel(algo) {
  if (!algo) return '—';
  return algo
    .split('_')
    .map((w) => (w.toLowerCase() === 'mlp' || w.toLowerCase() === 'gru' || w.toLowerCase() === 'knn' || w.toLowerCase() === 'svm' || w.toLowerCase() === 'svr' ? w.toUpperCase() : w.toLowerCase() === 'lstm' ? 'LSTM' : w[0].toUpperCase() + w.slice(1)))
    .join(' ');
}

export default function Models() {
  const navigate = useNavigate();
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const [search, setSearch] = useState('');
  const [modelType, setModelType] = useState('All');
  const [symbolFilter, setSymbolFilter] = useState('All');

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    // include_deep_learning always requested as 'true' -- the Models
    // page shows every trained run by default (regression +
    // classification, traditional + deep learning), with no separate
    // toggle to narrow that down.
    const params = new URLSearchParams({ limit: '500', include_deep_learning: 'true' });
    if (modelType !== 'All') params.set('model_type', modelType);

    api.get(`/api/ml-models?${params.toString()}`)
      .then((res) => setRuns(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [modelType]);

  useEffect(() => {
    load();
  }, [load]);

  const symbolOptions = useMemo(() => [
    { value: 'All', label: 'All symbols' },
    ...[...new Set(runs.map((r) => r.symbol).filter(Boolean))].sort().map((s) => ({ value: s, label: s.toUpperCase() })),
  ], [runs]);

  const filtered = useMemo(() => {
    const q = search.toLowerCase();
    return runs.filter((r) => {
      const matchesSearch = !q || r.run_id.toLowerCase().includes(q) || (r.algorithm || '').toLowerCase().includes(q);
      const matchesSymbol = symbolFilter === 'All' || r.symbol === symbolFilter;
      return matchesSearch && matchesSymbol;
    });
  }, [runs, search, symbolFilter]);

  const regressionCount = runs.filter((r) => r.model_type === 'regression').length;
  const classificationCount = runs.filter((r) => r.model_type === 'classification').length;
  const dlCount = runs.filter((r) => r.is_deep_learning).length;

  const columns = [
    {
      title: 'Run', dataIndex: 'run_id', key: 'run_id',
      sorter: (a, b) => a.run_id.localeCompare(b.run_id),
      render: (t, row) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
          <div style={{
            width: 30, height: 30, borderRadius: 9, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(61,220,151,0.12)', color: MINT, flexShrink: 0,
          }}>
            <ExperimentOutlined style={{ fontSize: 14 }} />
          </div>
          <div style={{ minWidth: 0 }}>
            {/* Friendly label only -- run_id itself isn't shown, per
                request. Exchange + horizon are folded in alongside
                symbol/model_type/algorithm specifically so two runs never
                render identically: e.g. once Binance- and Bybit-trained
                BTC classification xgboost runs both exist side by side
                (see ml_repo.py -- exchange is baked into run_id/folder
                naming), this line is still what tells them apart. The
                raw run_id is still the row's React key and still what
                the click navigates on -- just not rendered.
            */}
            <div style={{ fontWeight: 600, color: '#F5F6F7', fontSize: 13.5, whiteSpace: 'nowrap' }}>
              {row.symbol ? row.symbol.toUpperCase() : '—'}
              {' · '}
              {row.model_type ? row.model_type[0].toUpperCase() + row.model_type.slice(1) : '—'}
              {' · '}
              {algorithmLabel(row.algorithm)}
            </div>
            <div style={{ color: '#6B7280', fontSize: 11.5, whiteSpace: 'nowrap' }}>
              {row.exchange ? row.exchange[0].toUpperCase() + row.exchange.slice(1) : '—'}
              {row.horizon != null && <> · horizon {row.horizon}</>}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'Algorithm', dataIndex: 'algorithm', key: 'algorithm',
      sorter: (a, b) => (a.algorithm || '').localeCompare(b.algorithm || ''),
      render: (t) => <span style={{ color: '#9096A0' }}>{algorithmLabel(t)}</span>,
    },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (t) => <span style={{ color: '#9096A0' }}>{t ? t.toUpperCase() : '—'}</span> },
    {
      title: 'Sharpe', dataIndex: 'sharpe', key: 'sharpe',
      sorter: (a, b) => (a.sharpe ?? -Infinity) - (b.sharpe ?? -Infinity),
      render: (v) => (
        <span style={{ fontFamily: 'ui-monospace, monospace', fontWeight: 600, color: v == null ? '#6B7280' : v >= 0 ? MINT : RED }}>
          {fmtMetric(v)}
        </span>
      ),
    },
    {
      title: 'Win Rate', dataIndex: 'win_rate', key: 'win_rate',
      sorter: (a, b) => (a.win_rate ?? -Infinity) - (b.win_rate ?? -Infinity),
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7' }}>{fmtPct(v)}</span>,
    },
  ];

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Machine Learning</h2>
        <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
          Every trained run currently stored. Select one to inspect its dataset, training, and evaluation record.
        </p>
      </div>

      {error && (
        <Alert
          type="error"
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load models</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={iconBtnStyle}>Retry</button>}
          style={{ marginBottom: 20, background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
          showIcon
        />
      )}

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <SummaryCard label="Total Runs" value={runs.length} />
        <SummaryCard label="Regression" value={regressionCount} />
        <SummaryCard label="Classification" value={classificationCount} />
        <SummaryCard label="Deep Learning" value={dlCount} color={AMBER} />
      </div>

      {/* Filters */}
      <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <Input
          placeholder="Search by run ID or algorithm"
          prefix={<SearchOutlined style={{ color: '#6B7280', marginRight: 4 }} />}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ maxWidth: 280, borderRadius: 999 }}
        />
        <Select value={modelType} onChange={setModelType} options={MODEL_TYPE_OPTIONS} style={{ width: 190 }} />
        <Select value={symbolFilter} onChange={setSymbolFilter} options={symbolOptions} style={{ width: 160 }} />
      </div>

      {/* Table -- overflow: hidden keeps the table's own corners/scrollbar
          inside the panel's rounded border instead of the table bleeding
          past it; scroll.x lets the (now wider, two-line) Run column and
          the rest of the columns scroll horizontally within the panel on
          narrow viewports rather than overflowing it. */}
      <div style={{ ...panel, padding: 20, overflow: 'hidden' }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '60px 0' }}>
            <Spin size="large" />
          </div>
        ) : (
          <Table
            columns={columns}
            dataSource={filtered.map((r) => ({ ...r, key: r.run_id }))}
            pagination={{ pageSize: 10 }}
            scroll={{ x: 'max-content' }}
            locale={{ emptyText: 'No trained runs match your filters.' }}
            onRow={(row) => ({
              onClick: () => navigate(`/models/${row.run_id}`),
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