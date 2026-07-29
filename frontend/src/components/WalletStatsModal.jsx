import { useState, useEffect } from 'react';
import { Modal, Spin, Table } from 'antd';
import { api } from '../lib/api';
import Panel from './Panel';
import KeyValue from './KeyValue';

const RED = '#F0466B';

// Columns from accounts.stats that are dict-shaped (per_symbol,
// trades_by_hour_of_day, trades_by_day_of_week, trades_by_date) --
// stored as JSON text (see accounts_utils.refresh_account_stats).
// Rendered as their own tables below the scalar KeyValue list instead
// of jammed into it.
const DICT_COLUMNS = new Set([
  'per_symbol',
  'trades_by_hour_of_day',
  'trades_by_day_of_week',
  'trades_by_date',
]);

// Columns not worth showing as their own KeyValue row.
const HIDDEN_COLUMNS = new Set(['account_name', 'updated_at']);

function parseIfJson(v) {
  if (typeof v !== 'string') return v;
  try {
    return JSON.parse(v);
  } catch {
    return v;
  }
}

function fmtLabel(key) {
  return key
    .split('_')
    .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}

function fmtValue(v) {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return Number.isInteger(v) ? v.toLocaleString('en-US') : v.toLocaleString('en-US', { maximumFractionDigits: 8 });
  return String(v);
}

/**
 * Renders every column in one accounts.stats row (GET
 * /api/wallets/{account_name}/stats) as-is -- no curation, no dropped
 * fields. Scalar columns become KeyValue rows in the order the DB
 * returns them; the handful of JSON/dict-valued columns
 * (per_symbol, trades_by_hour_of_day, trades_by_day_of_week,
 * trades_by_date) each get their own table underneath.
 */
export default function WalletStatsModal({ accountName, open, onClose }) {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!open || !accountName) return;
    setLoading(true);
    setError(null);
    setStats(null);
    api.get(`/api/wallets/${accountName}/stats`)
      .then((res) => setStats(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [open, accountName]);

  const scalarEntries = stats
    ? Object.entries(stats).filter(([k]) => !DICT_COLUMNS.has(k) && !HIDDEN_COLUMNS.has(k))
    : [];

  const dictSections = stats
    ? Object.keys(stats)
        .filter((k) => DICT_COLUMNS.has(k) && stats[k] != null)
        .map((k) => ({ key: k, value: parseIfJson(stats[k]) }))
    : [];

  return (
    <Modal
      title={accountName ? `Wallet Stats — ${accountName}` : 'Wallet Stats'}
      open={open}
      onCancel={onClose}
      footer={null}
      width={780}
      destroyOnClose
    >
      {loading ? (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '60px 0' }}>
          <Spin size="large" />
        </div>
      ) : error ? (
        <div style={{ color: RED, padding: '16px 2px', fontSize: 13 }}>{error}</div>
      ) : !stats ? (
        <div style={{ color: '#6B7280', fontSize: 13, padding: '24px 2px' }}>
          No row in accounts.stats yet for this wallet.
        </div>
      ) : (
        <div>
          <Panel title="accounts.stats">
            {scalarEntries.map(([key, value]) => (
              <KeyValue key={key} label={fmtLabel(key)} value={fmtValue(value)} mono />
            ))}
          </Panel>

          {dictSections.map(({ key, value }) => (
            <div key={key} style={{ marginTop: 20 }}>
              <Panel title={fmtLabel(key)}>
                <Table
                  size="small"
                  pagination={false}
                  rowKey="k"
                  dataSource={Object.entries(value || {}).map(([k, v]) => ({ k, v }))}
                  locale={{ emptyText: 'Empty.' }}
                  columns={
                    key === 'per_symbol'
                      ? [
                          { title: 'Symbol', dataIndex: 'k', key: 'k' },
                          {
                            title: 'Value', dataIndex: 'v', key: 'v',
                            render: (v) => (
                              <pre style={{ margin: 0, fontSize: 12, color: '#F5F6F7', whiteSpace: 'pre-wrap' }}>
                                {JSON.stringify(v, null, 2)}
                              </pre>
                            ),
                          },
                        ]
                      : [
                          { title: 'Key', dataIndex: 'k', key: 'k' },
                          { title: 'Value', dataIndex: 'v', key: 'v', render: (v) => fmtValue(v) },
                        ]
                  }
                />
              </Panel>
            </div>
          ))}

          {stats.updated_at && (
            <div style={{ color: '#6B7280', fontSize: 12, marginTop: 16, textAlign: 'right' }}>
              Last refreshed {new Date(stats.updated_at).toLocaleString()}
            </div>
          )}
        </div>
      )}
    </Modal>
  );
}