import { useState } from 'react';
import { Table, Tag, Switch, Modal, Form, Input, Select, message, Tooltip } from 'antd';
import {
  PlusOutlined, EditOutlined, DeleteOutlined, EyeInvisibleOutlined, EyeOutlined,
  CheckCircleFilled, CloseCircleFilled, WalletOutlined,
} from '@ant-design/icons';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const COINS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'LTC', 'MINA', 'SUI'];

// ---- placeholder data — replace with GET /api/wallets once backend exists ----
const initialWallets = [
  {
    key: '1', id: 1, label: 'Main Trading', accountType: 'Unified Trading (UTA)',
    apiKey: 'kQ7f...A93x', apiStatus: 'Connected', enabled: true,
    balance: 24812.44, unrealizedPnl: 312.8, totalPnl: 4218.6,
    strategies: [
      { name: 'BTC Momentum', symbol: 'BTCUSDT', status: 'Active' },
      { name: 'SOL Breakout', symbol: 'SOLUSDT', status: 'Active' },
    ],
    positions: [
      { symbol: 'BTCUSDT', side: 'Long', size: 0.42, entry: 61240, mark: 62580, pnl: 562.8 },
      { symbol: 'SOLUSDT', side: 'Short', size: 18, entry: 148.2, mark: 146.1, pnl: 37.8 },
    ],
    openOrders: [
      { symbol: 'BTCUSDT', side: 'Sell', type: 'Limit', price: 64200, qty: 0.42 },
      { symbol: 'ADAUSDT', side: 'Buy', type: 'Limit', price: 0.38, qty: 4200 },
    ],
    executions: [
      { strategy: 'BTC Momentum', symbol: 'BTCUSDT', status: 'Running', uptime: '2d 4h' },
      { strategy: 'SOL Breakout', symbol: 'SOLUSDT', status: 'Running', uptime: '14h 10m' },
    ],
  },
  {
    key: '2', id: 2, label: 'Altcoin Sub-Account', accountType: 'Spot',
    apiKey: 'pT2m...C81z', apiStatus: 'Connected', enabled: true,
    balance: 8734.12, unrealizedPnl: -84.3, totalPnl: 612.4,
    strategies: [
      { name: 'ADA Trend Follow', symbol: 'ADAUSDT', status: 'Active' },
      { name: 'DOGE Volatility Break', symbol: 'DOGEUSDT', status: 'Paused' },
    ],
    positions: [
      { symbol: 'ADAUSDT', side: 'Long', size: 3200, entry: 0.402, mark: 0.389, pnl: -41.6 },
    ],
    openOrders: [
      { symbol: 'DOGEUSDT', side: 'Buy', type: 'Limit', price: 0.112, qty: 5000 },
    ],
    executions: [
      { strategy: 'ADA Trend Follow', symbol: 'ADAUSDT', status: 'Running', uptime: '6d 2h' },
      { strategy: 'DOGE Volatility Break', symbol: 'DOGEUSDT', status: 'Paused', uptime: '—' },
    ],
  },
  {
    key: '3', id: 3, label: 'Testnet Sandbox', accountType: 'Unified Trading (UTA)',
    apiKey: 'zR4k...E22q', apiStatus: 'Disconnected', enabled: false,
    balance: 1000.0, unrealizedPnl: 0, totalPnl: -18.2,
    strategies: [],
    positions: [],
    openOrders: [],
    executions: [],
  },
  {
    key: '4', id: 4, label: 'MINA/SUI Desk', accountType: 'Unified Trading (UTA)',
    apiKey: 'jH9v...F04w', apiStatus: 'Connected', enabled: true,
    balance: 5340.9, unrealizedPnl: 55.2, totalPnl: 201.9,
    strategies: [
      { name: 'MINA Grid', symbol: 'MINAUSDT', status: 'Active' },
      { name: 'SUI Scalper', symbol: 'SUIUSDT', status: 'Active' },
    ],
    positions: [
      { symbol: 'MINAUSDT', side: 'Long', size: 1200, entry: 0.71, mark: 0.735, pnl: 30.0 },
      { symbol: 'SUIUSDT', side: 'Long', size: 340, entry: 3.82, mark: 3.89, pnl: 23.8 },
    ],
    openOrders: [
      { symbol: 'SUIUSDT', side: 'Sell', type: 'Limit', price: 4.05, qty: 340 },
    ],
    executions: [
      { strategy: 'MINA Grid', symbol: 'MINAUSDT', status: 'Running', uptime: '3d 9h' },
      { strategy: 'SUI Scalper', symbol: 'SUIUSDT', status: 'Running', uptime: '11h 40m' },
    ],
  },
];

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
  v.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 });

const pnlColor = (v) => (v > 0 ? MINT : v < 0 ? RED : '#9096A0');

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
          data={wallet.openOrders}
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
  const [wallets, setWallets] = useState(initialWallets);
  const [revealedKeys, setRevealedKeys] = useState({});
  const [modalOpen, setModalOpen] = useState(false);
  const [editingWallet, setEditingWallet] = useState(null); // null = add mode
  const [form] = Form.useForm();

  const toggleReveal = (id) => setRevealedKeys((prev) => ({ ...prev, [id]: !prev[id] }));

  const toggleEnabled = (id) => {
    setWallets((prev) => prev.map((w) => (w.id === id ? { ...w, enabled: !w.enabled } : w)));
    message.success('Wallet status updated');
  };

  const removeWallet = (id) => {
    Modal.confirm({
      title: 'Remove this wallet?',
      content: 'This will unlink the Bybit account and stop any running executions tied to it.',
      okText: 'Remove',
      okButtonProps: { danger: true },
      onOk: () => {
        setWallets((prev) => prev.filter((w) => w.id !== id));
        message.success('Wallet removed');
      },
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
      label: wallet.label,
      accountType: wallet.accountType,
      apiKey: '',
      apiSecret: '',
      coinFocus: wallet.coinFocus || [],
    });
    setModalOpen(true);
  };

  const handleSubmit = () => {
    form.validateFields().then((values) => {
      if (editingWallet) {
        setWallets((prev) =>
          prev.map((w) =>
            w.id === editingWallet.id
              ? { ...w, label: values.label, accountType: values.accountType, coinFocus: values.coinFocus }
              : w
          )
        );
        message.success('Wallet updated');
      } else {
        const newWallet = {
          key: String(Date.now()), id: Date.now(),
          label: values.label, accountType: values.accountType,
          apiKey: values.apiKey.slice(0, 4) + '...' + values.apiKey.slice(-4),
          apiStatus: 'Connected', enabled: true,
          balance: 0, unrealizedPnl: 0, totalPnl: 0,
          coinFocus: values.coinFocus,
          strategies: [], positions: [], openOrders: [], executions: [],
        };
        setWallets((prev) => [...prev, newWallet]);
        message.success('Bybit wallet added');
      }
      setModalOpen(false);
    });
  };

  const columns = [
    {
      title: 'Wallet', key: 'label',
      render: (_, row) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{
            width: 34, height: 34, borderRadius: 10, display: 'flex', alignItems: 'center', justifyContent: 'center',
            background: 'rgba(61,220,151,0.12)', color: MINT, flexShrink: 0,
          }}>
            <WalletOutlined style={{ fontSize: 16 }} />
          </div>
          <div>
            <div style={{ fontWeight: 600, color: '#F5F6F7' }}>{row.label}</div>
            <div style={{ fontSize: 12, color: '#6B7280' }}>Bybit &middot; {row.accountType}</div>
          </div>
        </div>
      ),
    },
    {
      title: 'API Key', key: 'apiKey',
      render: (_, row) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'ui-monospace, monospace', fontSize: 13, color: '#9096A0' }}>
          {revealedKeys[row.id] ? row.apiKey.replace('...', '••••••••') : row.apiKey}
          <span
            onClick={(e) => { e.stopPropagation(); toggleReveal(row.id); }}
            style={{ cursor: 'pointer', color: '#6B7280', display: 'flex' }}
          >
            {revealedKeys[row.id] ? <EyeInvisibleOutlined /> : <EyeOutlined />}
          </span>
        </div>
      ),
    },
    {
      title: 'API Status', dataIndex: 'apiStatus', key: 'apiStatus',
      render: (status) => (
        <span style={{ display: 'flex', alignItems: 'center', gap: 6, color: status === 'Connected' ? MINT : RED, fontWeight: 600 }}>
          {status === 'Connected' ? <CheckCircleFilled /> : <CloseCircleFilled />}
          {status}
        </span>
      ),
    },
    {
      title: 'Account Balance', dataIndex: 'balance', key: 'balance',
      sorter: (a, b) => a.balance - b.balance,
      render: (v) => <span style={{ fontFamily: 'ui-monospace, monospace', color: '#F5F6F7', fontWeight: 600 }}>{fmtUsd(v)}</span>,
    },
    {
      title: 'Unrealized PnL', dataIndex: 'unrealizedPnl', key: 'unrealizedPnl',
      sorter: (a, b) => a.unrealizedPnl - b.unrealizedPnl,
      render: (v) => (
        <span style={{ fontFamily: 'ui-monospace, monospace', color: pnlColor(v), fontWeight: 600 }}>
          {v >= 0 ? '+' : ''}{fmtUsd(v)}
        </span>
      ),
    },
    {
      title: 'Total PnL', dataIndex: 'totalPnl', key: 'totalPnl',
      sorter: (a, b) => a.totalPnl - b.totalPnl,
      render: (v) => (
        <span style={{ fontFamily: 'ui-monospace, monospace', color: pnlColor(v), fontWeight: 600 }}>
          {v >= 0 ? '+' : ''}{fmtUsd(v)}
        </span>
      ),
    },
    {
      title: 'Enabled', key: 'enabled',
      render: (_, row) => (
        <Switch
          checked={row.enabled}
          onChange={() => toggleEnabled(row.id)}
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
            <button onClick={() => removeWallet(row.id)} style={{ ...iconBtnStyle, color: RED }}>
              <DeleteOutlined />
            </button>
          </Tooltip>
        </div>
      ),
    },
  ];

  const totalBalance = wallets.reduce((s, w) => s + w.balance, 0);
  const totalUnrealized = wallets.reduce((s, w) => s + w.unrealizedPnl, 0);
  const totalPnl = wallets.reduce((s, w) => s + w.totalPnl, 0);
  const connectedCount = wallets.filter((w) => w.apiStatus === 'Connected').length;

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Wallets</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            Manage your Bybit accounts, API connections, and per-wallet activity.
          </p>
        </div>
        <button onClick={openAddModal} style={primaryBtnStyle}>
          <PlusOutlined /> Add Wallet
        </button>
      </div>

      {/* Summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <SummaryCard label="Total Balance" value={fmtUsd(totalBalance)} />
        <SummaryCard label="Unrealized PnL" value={`${totalUnrealized >= 0 ? '+' : ''}${fmtUsd(totalUnrealized)}`} color={pnlColor(totalUnrealized)} />
        <SummaryCard label="Total PnL" value={`${totalPnl >= 0 ? '+' : ''}${fmtUsd(totalPnl)}`} color={pnlColor(totalPnl)} />
        <SummaryCard label="Connected Accounts" value={`${connectedCount} / ${wallets.length}`} />
      </div>

      <div style={{ ...panel, padding: 20 }}>
        <Table
          columns={columns}
          dataSource={wallets}
          pagination={false}
          expandable={{
            expandedRowRender: (row) => <WalletExpandedRow wallet={row} />,
            expandRowByClick: true,
          }}
          locale={{ emptyText: 'No wallets connected yet.' }}
        />
      </div>

      <Modal
        title={editingWallet ? 'Edit Wallet' : 'Add Bybit Wallet'}
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={handleSubmit}
        okText={editingWallet ? 'Save Changes' : 'Add Wallet'}
        destroyOnClose
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <Form.Item name="label" label="Wallet Label" rules={[{ required: true, message: 'Please enter a label' }]}>
            <Input placeholder="e.g. Main Trading" />
          </Form.Item>
          <Form.Item name="accountType" label="Account Type" rules={[{ required: true }]} initialValue="Unified Trading (UTA)">
            <Select
              options={[
                { value: 'Unified Trading (UTA)', label: 'Unified Trading (UTA)' },
                { value: 'Spot', label: 'Spot' },
                { value: 'Derivatives', label: 'Derivatives (USDT Perpetual)' },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="apiKey"
            label="Bybit API Key"
            rules={editingWallet ? [] : [{ required: true, message: 'API key is required' }]}
          >
            <Input.Password placeholder={editingWallet ? 'Leave blank to keep current key' : 'Enter API key'} />
          </Form.Item>
          <Form.Item
            name="apiSecret"
            label="Bybit API Secret"
            rules={editingWallet ? [] : [{ required: true, message: 'API secret is required' }]}
          >
            <Input.Password placeholder={editingWallet ? 'Leave blank to keep current secret' : 'Enter API secret'} />
          </Form.Item>
          <Form.Item name="coinFocus" label="Coin Focus (optional)">
            <Select
              mode="multiple"
              placeholder="Select coins traded on this wallet"
              options={COINS.map((c) => ({ value: c, label: c }))}
            />
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