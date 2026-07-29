import { useState, useEffect, useCallback } from 'react';
import { Table, Switch, Modal, Form, Input, Select, message, Tooltip, Spin, Alert } from 'antd';
import {
  PlusOutlined, EditOutlined, DeleteOutlined,
  WalletOutlined, WarningFilled, ControlOutlined,
} from '@ant-design/icons';
import { api } from '../lib/api';
import { fmtUsd, pnlColor } from '../lib/format';
import { panelGradient as panel } from '../components/Panel';
import WalletStatsModal from '../components/WalletStatsModal';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const subPanel = {
  background: 'rgba(255,255,255,0.02)',
  border: '1px solid rgba(255,255,255,0.06)',
  borderRadius: 14,
};

export default function Wallets() {
  const [wallets, setWallets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [statsWallet, setStatsWallet] = useState(null); // account_name currently shown in the stats modal
  const [modalOpen, setModalOpen] = useState(false);
  const [editingWallet, setEditingWallet] = useState(null); // null = add mode
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();

  const [assignModalOpen, setAssignModalOpen] = useState(false);
  const [assignWallet, setAssignWallet] = useState(null); // wallet row this modal is for
  const [assignCoins, setAssignCoins] = useState([]);
  const [assignLoading, setAssignLoading] = useState(false);
  const [assignSavingKey, setAssignSavingKey] = useState(null); // `${exchange}-${symbol}` currently saving

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

  const openStatsModal = (row) => setStatsWallet(row.account_name);
  const closeStatsModal = () => setStatsWallet(null);

  const openAssignModal = (wallet) => {
    setAssignWallet(wallet);
    setAssignModalOpen(true);
    setAssignLoading(true);
    api.get(`/api/wallets/${wallet.account_name}/assignments`)
      .then((res) => setAssignCoins(res.data))
      .catch((err) => message.error(err.message))
      .finally(() => setAssignLoading(false));
  };

  const closeAssignModal = () => {
    setAssignModalOpen(false);
    setAssignWallet(null);
    setAssignCoins([]);
  };

  const handleAssignStrategy = (coin, strategyId) => {
    const key = `${coin.exchange}-${coin.symbol}`;
    setAssignSavingKey(key);

    const request = strategyId == null
      ? api.delete(`/api/wallets/${assignWallet.account_name}/assignments?exchange=${coin.exchange}&symbol=${coin.symbol}`)
      : api.patch(`/api/wallets/${assignWallet.account_name}/assignments`, {
          exchange: coin.exchange,
          symbol: coin.symbol,
          strategy_id: strategyId,
        });

    request
      .then(() => {
        message.success(
          strategyId == null
            ? `Execution turned off for ${coin.symbol.toUpperCase()}`
            : `${coin.symbol.toUpperCase()} now running on ${assignWallet.account_name}`
        );
        return api.get(`/api/wallets/${assignWallet.account_name}/assignments`).then((res) => setAssignCoins(res.data));
      })
      .catch((err) => message.error(err.message))
      .finally(() => setAssignSavingKey(null));
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
          <Tooltip title="Manage strategies">
            <button onClick={() => openAssignModal(row)} style={iconBtnStyle}>
              <ControlOutlined />
            </button>
          </Tooltip>
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

  const tableData = wallets.map((w) => ({ ...w, key: w.account_name }));

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
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load wallets</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={loadWallets} style={iconBtnStyle}>Retry</button>}
          style={{ marginBottom: 20, background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
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
            onRow={(row) => ({
              onClick: () => openStatsModal(row),
              style: { cursor: 'pointer' },
            })}
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

      <Modal
        title={assignWallet ? `Manage Strategies — ${assignWallet.account_name}` : 'Manage Strategies'}
        open={assignModalOpen}
        onCancel={closeAssignModal}
        footer={null}
        destroyOnClose
        width={640}
      >
        <p style={{ color: '#9096A0', fontSize: 13, marginTop: -4, marginBottom: 16 }}>
          Pick one strategy per coin to run live on this wallet. Choosing a strategy here disables
          any other strategy currently live for that coin — only one strategy can run per coin at a time.
        </p>

        {assignLoading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '40px 0' }}>
            <Spin size="large" />
          </div>
        ) : assignCoins.length === 0 ? (
          <div style={{ color: '#6B7280', fontSize: 13, padding: '16px 2px' }}>
            No coins are set up for execution yet. Configure a pair in Execution first.
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {assignCoins.map((coin) => {
              const key = `${coin.exchange}-${coin.symbol}`;
              const liveStrategy = coin.strategies.find((s) => s.execution_enabled);
              const isOtherWallet = coin.assigned_account && coin.assigned_account !== assignWallet?.account_name;
              return (
                <div
                  key={key}
                  style={{
                    ...subPanel, padding: 14,
                    display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap',
                  }}
                >
                  <div>
                    <div style={{ fontWeight: 600, color: '#F5F6F7' }}>
                      {coin.symbol.toUpperCase()} <span style={{ color: '#6B7280', fontWeight: 400 }}>· {coin.exchange}</span>
                    </div>
                    {isOtherWallet && (
                      <div style={{ fontSize: 12, color: AMBER, marginTop: 2 }}>
                        Currently assigned to {coin.assigned_account} — picking a strategy moves it here.
                      </div>
                    )}
                    {coin.strategies.length === 0 && (
                      <div style={{ fontSize: 12, color: '#6B7280', marginTop: 2 }}>No strategies loaded for this coin yet.</div>
                    )}
                  </div>
                  <Select
                    allowClear
                    placeholder="No strategy running"
                    value={liveStrategy ? liveStrategy.strategy_id : undefined}
                    loading={assignSavingKey === key}
                    disabled={coin.strategies.length === 0}
                    onChange={(value) => handleAssignStrategy(coin, value ?? null)}
                    style={{ width: 260 }}
                    options={coin.strategies.map((s) => ({
                      value: s.strategy_id,
                      label: `${s.strategy_name} (${s.time_horizon || '—'})`,
                    }))}
                  />
                </div>
              );
            })}
          </div>
        )}
      </Modal>

      <WalletStatsModal
        accountName={statsWallet}
        open={!!statsWallet}
        onClose={closeStatsModal}
      />
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