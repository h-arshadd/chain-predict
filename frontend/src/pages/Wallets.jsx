import { useState, useEffect, useCallback } from 'react';
import { Table, Tag, Switch, Modal, Form, Input, Select, message, Tooltip, Spin, Alert } from 'antd';
import {
  PlusOutlined, EditOutlined, DeleteOutlined, EyeInvisibleOutlined, EyeOutlined,
  WalletOutlined, WarningFilled,
} from '@ant-design/icons';
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

const subPanel = {
  background: 'rgba(255,255,255,0.02)',
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: 14,
};

const fmtUsd = (v) =>
  v == null
    ? '—'
    : v.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });

const pnlColor = (v) => (v == null ? '#6B7280' : v > 0 ? MINT : v < 0 ? RED : '#9096A0');

function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 0.6, color: '#6B7280', textTransform: 'uppercase', marginBottom: 10 }}>
      {children}
    </div>
  );
}

function MiniTable({ columns, data, emptyText }) {
  if (!data.length) {
    return <div style={{ color: '#6B7280', fontSize: 13, padding: '10px 2px' }}>{emptyText}</div>;
  }
  return (
    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
      <thead>
        <tr>
          {columns.map((c) => (
            <th
              key={c.key}
              style={{
                textAlign: c.align || 'left', color: '#6B7280', fontWeight: 600,
                fontSize: 11.5, textTransform: 'uppercase', letterSpacing: 0.3,
                padding: '0 10px 8px', borderBottom: '1px solid rgba(255,255,255,0.06)',
              }}
            >
              {c.title}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, i) => (
          <tr key={i} style={{ borderBottom: i < data.length - 1 ? '1px solid rgba(255,255,255,0.04)' : 'none' }}>
            {columns.map((c) => (
              <td key={c.key} style={{ padding: '9px 10px', textAlign: c.align || 'left', color: '#F5F6F7' }}>
                {c.render ? c.render(row) : row[c.key]}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// Strategies/positions/open orders/executions are always [] for now --
// the wallets API stubs these until the Strategy Deployment module is
// wired up to join against them. The expandable row still renders
// correctly with its own empty states in the meantime.
function WalletExpandedRow({ wallet }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, padding: '4px 8px 16px' }}>
      <div style={{ ...subPanel, padding: 16 }}>
        <SectionLabel>Strategies Assigned</SectionLabel>
        <MiniTable
          emptyText="No strategies assigned to this wallet."
          columns={[
            { key: 'name', title: 'Strategy' },
            { key: 'symbol', title: 'Symbol' },
            {
              key: 'status', title: 'Status',
              render: (r) => (
                <Tag style={{
                  background: r.status === 'Active' ? 'rgba(61,220,151,0.12)' : 'rgba(255,138,92,0.14)',
                  color: r.status === 'Active' ? MINT : AMBER,
                  border: 'none', borderRadius: 8, fontWeight: 600,
                }}>
                  {r.status}
                </Tag>
              ),
            },
          ]}
          data={wallet.strategies}
        />
      </div>

      <div style={{ ...subPanel, padding: 16 }}>
        <SectionLabel>Active Positions</SectionLabel>
        <MiniTable
          emptyText="No open positions."
          columns={[
            { key: 'symbol', title: 'Symbol' },
            {
              key: 'side', title: 'Side',
              render: (r) => <span style={{ color: r.side === 'Long' ? MINT : RED, fontWeight: 600 }}>{r.side}</span>,
            },
            { key: 'size', title: 'Size', align: 'right' },
            {
              key: 'pnl', title: 'PnL', align: 'right',
              render: (r) => <span style={{ color: pnlColor(r.pnl), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
                {r.pnl >= 0 ? '+' : ''}{r.pnl.toFixed(2)}
              </span>,
            },
          ]}
          data={wallet.positions}
        />
      </div>

      <div style={{ ...subPanel, padding: 16 }}>
        <SectionLabel>Open Orders</SectionLabel>
        <MiniTable
          emptyText="No open orders."
          columns={[
            { key: 'symbol', title: 'Symbol' },
            {
              key: 'side', title: 'Side',
              render: (r) => <span style={{ color: r.side === 'Buy' ? MINT : RED, fontWeight: 600 }}>{r.side}</span>,
            },
            { key: 'type', title: 'Type' },
            { key: 'price', title: 'Price', align: 'right' },
            { key: 'qty', title: 'Qty', align: 'right' },
          ]}
          data={wallet.open_orders}
        />
      </div>

      <div style={{ ...subPanel, padding: 16 }}>
        <SectionLabel>Running Executions</SectionLabel>
        <MiniTable
          emptyText="No running executions."
          columns={[
            { key: 'strategy', title: 'Strategy' },
            { key: 'symbol', title: 'Symbol' },
            {
              key: 'status', title: 'Status',
              render: (r) => (
                <Tag style={{
                  background: r.status === 'Running' ? 'rgba(61,220,151,0.12)' : 'rgba(255,255,255,0.06)',
                  color: r.status === 'Running' ? MINT : '#9096A0',
                  border: 'none', borderRadius: 8, fontWeight: 600,
                }}>
                  {r.status}
                </Tag>
              ),
            },
            { key: 'uptime', title: 'Uptime', align: 'right' },
          ]}
          data={wallet.executions}
        />
      </div>
    </div>
  );
}

export default function Wallets() {
  const [wallets, setWallets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedRows, setExpandedRows] = useState({}); // account_name -> detail (fetched on expand)
  const [modalOpen, setModalOpen] = useState(false);
  const [editingWallet, setEditingWallet] = useState(null); // null = add mode
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();

  const loadWallets = useCallback(() => {
    setLoading(true);
    setError(null);
    api.get('/api/wallets')
      .then((res) => setWallets(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadWallets();
  }, [loadWallets]);

  const toggleEnabled = (accountName, nextEnabled) => {
    // Optimistic update so the switch feels instant; rolled back on failure.
    setWallets((prev) => prev.map((w) => (w.account_name === accountName ? { ...w, enabled: nextEnabled } : w)));
    api.patch(`/api/wallets/${accountName}/enabled`, { enabled: nextEnabled })
      .then(() => message.success(nextEnabled ? 'Wallet enabled' : 'Wallet disabled — no new executions will open'))
      .catch((err) => {
        setWallets((prev) => prev.map((w) => (w.account_name === accountName ? { ...w, enabled: !nextEnabled } : w)));
        message.error(err.message);
      });
  };

  const removeWallet = (accountName) => {
    Modal.confirm({
      title: 'Remove this wallet?',
      content: 'This will delete the stored API credentials for this account. This cannot be undone.',
      okText: 'Remove',
      okButtonProps: { danger: true },
      onOk: () =>
        api.delete(`/api/wallets/${accountName}`)
          .then(() => {
            message.success('Wallet removed');
            loadWallets();
          })
          .catch((err) => message.error(err.message)),
    });
  };

  const openAddModal = () => {
    setEditingWallet(null);
    form.resetFields();
    setModalOpen(true);
  };

  const openEditModal = (wallet) => {
    setEditingWallet(wallet);
    form.setFieldsValue({
      account_name: wallet.account_name,
      exchange: wallet.exchange,
      demo: wallet.demo,
      api_key: '',
      api_secret: '',
    });
    setModalOpen(true);
  };

  const handleSubmit = () => {
    form.validateFields().then((values) => {
      setSubmitting(true);
      const request = editingWallet
        ? api.put(`/api/wallets/${editingWallet.account_name}`, {
            exchange: values.exchange,
            demo: values.demo,
            // blank = keep current key/secret, matches placeholder copy below
            api_key: values.api_key || null,
            api_secret: values.api_secret || null,
          })
        : api.post('/api/wallets', {
            account_name: values.account_name,
            exchange: values.exchange,
            demo: values.demo,
            api_key: values.api_key,
            api_secret: values.api_secret,
          });

      request
        .then(() => {
          message.success(editingWallet ? 'Wallet updated' : 'Wallet added');
          setModalOpen(false);
          loadWallets();
        })
        .catch((err) => message.error(err.message))
        .finally(() => setSubmitting(false));
    });
  };

  const handleExpand = (expanded, row) => {
    if (!expanded || expandedRows[row.account_name]) return;
    api.get(`/api/wallets/${row.account_name}`)
      .then((res) => setExpandedRows((prev) => ({ ...prev, [row.account_name]: res.data })))
      .catch((err) => message.error(err.message));
  };

  const columns = [
    {
      title: 'Wallet', key: 'account_name',
      render: (_, row) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(61,220,151,0.12)', color: MINT, flexShrink: 0,
          }}>
            <WalletOutlined style={{ fontSize: 16 }} />
          </div>
          <div>
            <div style={{ fontWeight: 600, color: '#F5F6F7' }}>{row.account_name}</div>
            <div style={{ fontSize: 12, color: '#6B7280' }}>
              {row.exchange} &middot; {row.demo ? 'Demo' : 'Production'}
            </div>
          </div>
        </div>
      ),
    },
    {
      title: 'API Key', dataIndex: 'api_key_masked', key: 'api_key_masked',
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 13, color: '#9096A0' }}>{v}</span>,
    },
    {
      title: 'Account Balance', dataIndex: 'balance', key: 'balance',
      sorter: (a, b) => (a.balance ?? -Infinity) - (b.balance ?? -Infinity),
      render: (v, row) =>
        row.balance_error ? (
          <Tooltip title={row.balance_error}>
            <span style={{ color: RED, display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: 13 }}>
              <WarningFilled /> Unavailable
            </span>
          </Tooltip>
        ) : (
          <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7', fontWeight: 600 }}>{fmtUsd(v)}</span>
        ),
    },
    {
      title: 'Unrealized PnL', dataIndex: 'unrealized_pnl', key: 'unrealized_pnl',
      sorter: (a, b) => (a.unrealized_pnl ?? -Infinity) - (b.unrealized_pnl ?? -Infinity),
      render: (v) => (
        <span style={{ fontFamily: 'ui-monospace, monospace', color: pnlColor(v), fontWeight: 600 }}>
          {v == null ? '—' : `${v >= 0 ? '+' : ''}${fmtUsd(v)}`}
        </span>
      ),
    },
    {
      title: 'Total PnL', dataIndex: 'total_pnl', key: 'total_pnl',
      sorter: (a, b) => (a.total_pnl ?? -Infinity) - (b.total_pnl ?? -Infinity),
      render: (v) => (
        <span style={{ fontFamily: 'ui-monospace, monospace', color: pnlColor(v), fontWeight: 600 }}>
          {v == null ? '—' : `${v >= 0 ? '+' : ''}${fmtUsd(v)}`}
        </span>
      ),
    },
    {
      title: 'Enabled', key: 'enabled',
      render: (_, row) => (
        <Switch
          checked={row.enabled}
          onChange={(checked) => toggleEnabled(row.account_name, checked)}
          onClick={(_, e) => e.stopPropagation()}
        />
      ),
    },
    {
      title: '', key: 'actions',
      render: (_, row) => (
        <div style={{ display: 'flex', gap: 6 }} onClick={(e) => e.stopPropagation()}>
          <Tooltip title="Edit API keys">
            <button onClick={() => openEditModal(row)} style={iconBtnStyle}>
              <EditOutlined />
            </button>
          </Tooltip>
          <Tooltip title="Remove wallet">
            <button onClick={() => removeWallet(row.account_name)} style={{ ...iconBtnStyle, color: RED }}>
              <DeleteOutlined />
            </button>
          </Tooltip>
        </div>
      ),
    },
  ];

  const totalBalance = wallets.reduce((s, w) => s + (w.balance ?? 0), 0);
  const totalUnrealized = wallets.reduce((s, w) => s + (w.unrealized_pnl ?? 0), 0);
  const totalPnl = wallets.reduce((s, w) => s + (w.total_pnl ?? 0), 0);
  const enabledCount = wallets.filter((w) => w.enabled).length;

  const tableData = wallets.map((w) => ({
    ...w,
    key: w.account_name,
    ...(expandedRows[w.account_name] || {}),
  }));

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Wallets</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            Manage your exchange accounts, API connections, and per-wallet activity.
          </p>
        </div>
        <button onClick={openAddModal} style={primaryBtnStyle}>
          <PlusOutlined /> Add Wallet
        </button>
      </div>

      {error && (
        <Alert
          type="error"
          message="Couldn't load wallets"
          description={error}
          action={<button onClick={loadWallets} style={iconBtnStyle}>Retry</button>}
          style={{ marginBottom: 20 }}
          showIcon
        />
      )}

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <SummaryCard label="Total Balance" value={fmtUsd(totalBalance)} />
        <SummaryCard label="Unrealized PnL" value={`${totalUnrealized >= 0 ? '+' : ''}${fmtUsd(totalUnrealized)}`} color={pnlColor(totalUnrealized)} />
        <SummaryCard label="Total PnL" value={`${totalPnl >= 0 ? '+' : ''}${fmtUsd(totalPnl)}`} color={pnlColor(totalPnl)} />
        <SummaryCard label="Enabled Wallets" value={`${enabledCount} / ${wallets.length}`} />
      </div>

      <div style={{ ...panel, padding: 20 }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '60px 0' }}>
            <Spin size="large" />
          </div>
        ) : (
          <Table
            columns={columns}
            dataSource={tableData}
            pagination={false}
            expandable={{
              expandedRowRender: (row) => <WalletExpandedRow wallet={row} />,
              onExpand: handleExpand,
            }}
            locale={{ emptyText: 'No wallets connected yet. Click "Add Wallet" to connect your first exchange account.' }}
          />
        )}
      </div>

      <Modal
        title={editingWallet ? 'Edit Wallet' : 'Add Wallet'}
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={handleSubmit}
        confirmLoading={submitting}
        okText={editingWallet ? 'Save Changes' : 'Add Wallet'}
        destroyOnClose
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item
            name="account_name"
            label="Account Name"
            rules={[{ required: true, message: 'Please enter a unique account name' }]}
          >
            <Input placeholder="e.g. main_trading" disabled={!!editingWallet} />
          </Form.Item>
          <Form.Item name="exchange" label="Exchange" rules={[{ required: true }]} initialValue="bybit">
            <Select
              options={[
                { value: 'bybit', label: 'Bybit' },
                { value: 'binance', label: 'Binance' },
              ]}
            />
          </Form.Item>
          <Form.Item name="demo" label="Environment" rules={[{ required: true }]} initialValue={true}>
            <Select
              options={[
                { value: true, label: 'Demo Trading' },
                { value: false, label: 'Production (real funds)' },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="api_key"
            label="API Key"
            rules={editingWallet ? [] : [{ required: true, message: 'API key is required' }]}
          >
            <Input.Password placeholder={editingWallet ? 'Leave blank to keep current key' : 'Enter API key'} />
          </Form.Item>
          <Form.Item
            name="api_secret"
            label="API Secret"
            rules={editingWallet ? [] : [{ required: true, message: 'API secret is required' }]}
          >
            <Input.Password placeholder={editingWallet ? 'Leave blank to keep current secret' : 'Enter API secret'} />
          </Form.Item>
        </Form>
      </Modal>
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

const primaryBtnStyle = {
  display: 'flex', alignItems: 'center', gap: 8,
  background: MINT, color: '#0B0E11', border: 'none',
  fontSize: 14, fontWeight: 700, padding: '10px 18px',
  borderRadius: 999, cursor: 'pointer',
};