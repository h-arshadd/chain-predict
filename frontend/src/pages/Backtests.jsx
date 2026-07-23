import { useState } from 'react';
import { Table, Tag, Modal, Form, Input, Select, DatePicker, InputNumber, message, Progress, Tabs } from 'antd';
import { useNavigate } from 'react-router-dom';
import { PlusOutlined, ExperimentOutlined } from '@ant-design/icons';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const COINS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'LTC', 'MINA', 'SUI'];
const TIMEFRAMES = ['1m', '5m', '15m', '1h', '4h', '1d'];
const STRATEGIES = ['BTC Momentum', 'ETH Mean Reversion', 'SOL Breakout', 'ADA Trend Follow', 'DOGE Volatility Break', 'MINA Grid', 'SUI Scalper', 'LTC Swing'];

// ---- placeholder data — replace with GET /api/backtests once backend exists ----
const initialRequests = [
  { key: '1', id: 1, strategy: 'BTC Momentum', symbol: 'BTCUSDT', timeframe: '4h', dateRange: '2025-01-01 → 2026-01-01', status: 'Completed', progress: 100, requestedAt: '2026-07-20 10:12', finalReturn: 34.2 },
  { key: '2', id: 2, strategy: 'ETH Mean Reversion', symbol: 'ETHUSDT', timeframe: '1h', dateRange: '2025-06-01 → 2026-06-01', status: 'Completed', progress: 100, requestedAt: '2026-07-19 08:40', finalReturn: -6.1 },
  { key: '3', id: 3, strategy: 'SOL Breakout', symbol: 'SOLUSDT', timeframe: '15m', dateRange: '2026-01-01 → 2026-07-01', status: 'Running', progress: 62, requestedAt: '2026-07-22 22:05', finalReturn: null },
  { key: '4', id: 4, strategy: 'ADA Trend Follow', symbol: 'ADAUSDT', timeframe: '1d', dateRange: '2024-01-01 → 2026-01-01', status: 'Pending', progress: 0, requestedAt: '2026-07-23 07:15', finalReturn: null },
  { key: '5', id: 5, strategy: 'DOGE Volatility Break', symbol: 'DOGEUSDT', timeframe: '1h', dateRange: '2025-03-01 → 2026-03-01', status: 'Failed', progress: 0, requestedAt: '2026-07-18 14:22', finalReturn: null, error: 'Insufficient historical data for range' },
  { key: '6', id: 6, strategy: 'MINA Grid', symbol: 'MINAUSDT', timeframe: '4h', dateRange: '2025-01-01 → 2026-01-01', status: 'Completed', progress: 100, requestedAt: '2026-07-15 09:00', finalReturn: 18.7 },
  { key: '7', id: 7, strategy: 'SUI Scalper', symbol: 'SUIUSDT', timeframe: '5m', dateRange: '2026-02-01 → 2026-07-01', status: 'Completed', progress: 100, requestedAt: '2026-07-12 16:30', finalReturn: 9.4 },
  { key: '8', id: 8, strategy: 'LTC Swing', symbol: 'LTCUSDT', timeframe: '1d', dateRange: '2024-06-01 → 2026-06-01', status: 'Running', progress: 21, requestedAt: '2026-07-23 06:50', finalReturn: null },
];

const panel = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

const statusColors = {
  Pending: { bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  Running: { bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  Completed: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  Failed: { bg: 'rgba(240,70,107,0.14)', fg: RED },
};

function StatusTag({ status }) {
  const c = statusColors[status] || statusColors.Pending;
  return (
    <Tag style={{ background: c.bg, color: c.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
      {status}
    </Tag>
  );
}

function buildColumns(navigate) {
  return [
    {
      title: 'Strategy', dataIndex: 'strategy', key: 'strategy',
      render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span>,
    },
    { title: 'Symbol', dataIndex: 'symbol', key: 'symbol', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Timeframe', dataIndex: 'timeframe', key: 'timeframe', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
    { title: 'Date Range', dataIndex: 'dateRange', key: 'dateRange', render: (t) => <span style={{ color: '#9096A0', fontSize: 13 }}>{t}</span> },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      render: (status, row) => (
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <StatusTag status={status} />
          {status === 'Running' && (
            <Progress percent={row.progress} size="small" style={{ width: 80 }} strokeColor={AMBER} showInfo={false} />
          )}
        </div>
      ),
    },
    {
      title: 'Result', key: 'result',
      render: (_, row) => {
        if (row.status === 'Completed') {
          return (
            <span style={{ color: row.finalReturn >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>
              {row.finalReturn >= 0 ? '+' : ''}{row.finalReturn}%
            </span>
          );
        }
        if (row.status === 'Failed') {
          return <span style={{ color: RED, fontSize: 12.5 }}>{row.error}</span>;
        }
        return <span style={{ color: '#6B7280' }}>—</span>;
      },
    },
    { title: 'Requested', dataIndex: 'requestedAt', key: 'requestedAt', render: (t) => <span style={{ color: '#6B7280', fontSize: 13 }}>{t}</span> },
  ];
}

function RequestsTable({ data, navigate, emptyText }) {
  return (
    <div style={{ ...panel, padding: 20 }}>
      <Table
        columns={buildColumns(navigate)}
        dataSource={data}
        pagination={{ pageSize: 8 }}
        locale={{ emptyText }}
        onRow={(row) => ({
          onClick: () => {
            if (row.status === 'Completed') navigate(`/backtests/${row.id}`);
          },
          style: { cursor: row.status === 'Completed' ? 'pointer' : 'default' },
        })}
      />
    </div>
  );
}

export default function Backtests() {
  const navigate = useNavigate();
  const [requests, setRequests] = useState(initialRequests);
  const [modalOpen, setModalOpen] = useState(false);
  const [form] = Form.useForm();

  const byStatus = (status) => requests.filter((r) => r.status === status);

  const handleSubmit = () => {
    form.validateFields().then((values) => {
      const newRequest = {
        key: String(Date.now()), id: Date.now(),
        strategy: values.strategy,
        symbol: `${values.coin}USDT`,
        timeframe: values.timeframe,
        dateRange: `${values.dateRange[0].format('YYYY-MM-DD')} → ${values.dateRange[1].format('YYYY-MM-DD')}`,
        status: 'Pending',
        progress: 0,
        requestedAt: new Date().toISOString().slice(0, 16).replace('T', ' '),
        finalReturn: null,
      };
      setRequests((prev) => [newRequest, ...prev]);
      message.success('Backtest request queued');
      setModalOpen(false);
      form.resetFields();
    });
  };

  const tabItems = [
    { key: 'pending', label: `Pending (${byStatus('Pending').length})`, children: <RequestsTable data={byStatus('Pending')} navigate={navigate} emptyText="No pending requests." /> },
    { key: 'running', label: `Running (${byStatus('Running').length})`, children: <RequestsTable data={byStatus('Running')} navigate={navigate} emptyText="No running requests." /> },
    { key: 'completed', label: `Completed (${byStatus('Completed').length})`, children: <RequestsTable data={byStatus('Completed')} navigate={navigate} emptyText="No completed requests." /> },
    { key: 'failed', label: `Failed (${byStatus('Failed').length})`, children: <RequestsTable data={byStatus('Failed')} navigate={navigate} emptyText="No failed requests." /> },
    { key: 'all', label: `All (${requests.length})`, children: <RequestsTable data={requests} navigate={navigate} emptyText="No backtest requests yet." /> },
  ];

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Backtest Requests</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            Configure and queue new backtests. Select a completed run to view full results.
          </p>
        </div>
        <button onClick={() => setModalOpen(true)} style={primaryBtnStyle}>
          <PlusOutlined /> New Backtest
        </button>
      </div>

      <Tabs items={tabItems} />

      <Modal
        title={
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ExperimentOutlined style={{ color: MINT }} /> New Backtest Request
          </div>
        }
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={handleSubmit}
        okText="Queue Backtest"
        width={640}
        destroyOnClose
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Form.Item name="strategy" label="Strategy" rules={[{ required: true }]}>
              <Select placeholder="Select strategy" options={STRATEGIES.map((s) => ({ value: s, label: s }))} />
            </Form.Item>
            <Form.Item name="coin" label="Symbol" rules={[{ required: true }]}>
              <Select placeholder="Select coin" options={COINS.map((c) => ({ value: c, label: `${c}USDT` }))} />
            </Form.Item>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Form.Item name="exchange" label="Exchange" initialValue="Bybit" rules={[{ required: true }]}>
              <Select disabled options={[{ value: 'Bybit', label: 'Bybit' }]} />
            </Form.Item>
            <Form.Item name="timeframe" label="Timeframe" rules={[{ required: true }]}>
              <Select placeholder="Select timeframe" options={TIMEFRAMES.map((t) => ({ value: t, label: t }))} />
            </Form.Item>
          </div>
          <Form.Item name="dateRange" label="Date Range" rules={[{ required: true, message: 'Please select a date range' }]}>
            <DatePicker.RangePicker style={{ width: '100%' }} />
          </Form.Item>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
            <Form.Item name="initialCapital" label="Initial Capital ($)" rules={[{ required: true }]} initialValue={10000}>
              <InputNumber style={{ width: '100%' }} min={100} step={500} />
            </Form.Item>
            <Form.Item name="commission" label="Commission (%)" rules={[{ required: true }]} initialValue={0.075}>
              <InputNumber style={{ width: '100%' }} min={0} step={0.005} />
            </Form.Item>
            <Form.Item name="slippage" label="Slippage (%)" rules={[{ required: true }]} initialValue={0.05}>
              <InputNumber style={{ width: '100%' }} min={0} step={0.01} />
            </Form.Item>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Form.Item name="riskPerTrade" label="Risk Per Trade (%)" initialValue={2}>
              <InputNumber style={{ width: '100%' }} min={0} step={0.5} />
            </Form.Item>
            <Form.Item name="maxDrawdownLimit" label="Max Drawdown Limit (%)" initialValue={15}>
              <InputNumber style={{ width: '100%' }} min={0} step={1} />
            </Form.Item>
          </div>
        </Form>
      </Modal>
    </div>
  );
}

const primaryBtnStyle = {
  display: 'flex', alignItems: 'center', gap: 8,
  background: MINT, color: '#0B0E11', border: 'none',
  fontSize: 14, fontWeight: 700, padding: '10px 18px',
  borderRadius: 999, cursor: 'pointer',
};