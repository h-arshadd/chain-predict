import { useState, useEffect, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Table, Select, Spin, Alert, Modal, Form, InputNumber, DatePicker, Switch, Tag, message } from 'antd';
import { PlusOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import { api } from '../lib/api';
import { fmtUsd, pnlColor } from '../lib/format';
import { panelGradient as panel } from '../components/Panel';

// Matches start_date in crypto_pipeline/data/binance/config_binance.yml
// and config_bybit.yml -- there's no candle data in the DB before this,
// for any coin/exchange. Used to block picking a start date the backend
// would just fail on later (see backtests_repo.run_backtest_job's data-
// coverage check).
const EARLIEST_DATA_DATE = dayjs('2024-01-01');

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

// Real status values from metadata.backtest.status -- see
// metadata_utils.create_backtest_table's self-heal / start_backtest /
// complete_backtest / fail_backtest. Pending/Running/Completed/Failed
// tabs, exactly the PDF's four buckets.
const STATUS_META = {
  pending: { label: 'Pending', bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  running: { label: 'Running', bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  completed: { label: 'Completed', bg: 'rgba(61,220,151,0.12)', fg: MINT },
  failed: { label: 'Failed', bg: 'rgba(240,70,107,0.14)', fg: RED },
};



const primaryBtnStyle = {
  display: 'flex', alignItems: 'center', gap: 8, padding: '10px 18px', borderRadius: 10,
  border: 'none', background: MINT, color: '#0B0F12', fontWeight: 700, fontSize: 13.5, cursor: 'pointer',
};



const TABS = [
  { key: 'all', label: 'All' },
  { key: 'pending', label: 'Pending' },
  { key: 'running', label: 'Running' },
  { key: 'completed', label: 'Completed' },
  { key: 'failed', label: 'Failed' },
];

export default function Backtests() {
  const navigate = useNavigate();

  const [backtests, setBacktests] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [tab, setTab] = useState('all');

  const [strategies, setStrategies] = useState([]);
  const [strategiesLoading, setStrategiesLoading] = useState(false);

  const [modalOpen, setModalOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [form] = Form.useForm();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.get('/api/backtests?limit=200')
      .then((res) => setBacktests(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Poll while anything is pending/running, so newly submitted requests
  // move through the list without a manual refresh -- same reasoning as
  // BacktestDetails.jsx's own polling.
  useEffect(() => {
    const hasInFlight = backtests.some((b) => b.status === 'pending' || b.status === 'running');
    if (!hasInFlight) return undefined;
    const interval = setInterval(load, 4000);
    return () => clearInterval(interval);
  }, [backtests, load]);

  const openModal = () => {
    setModalOpen(true);
    if (strategies.length === 0) {
      setStrategiesLoading(true);
      api.get('/api/strategies?limit=500')
        .then((res) => setStrategies(res.data))
        .catch((err) => message.error(`Couldn't load strategies: ${err.message}`))
        .finally(() => setStrategiesLoading(false));
    }
  };

  // Coin picked first, then only that coin's strategies are shown --
  // avoids one giant flat list of 100+ "name — coin · exchange · tf"
  // options that's hard to scan when there are many strategies per coin.
  const coinOptions = useMemo(() => {
    const seen = new Set();
    const list = [];
    for (const s of strategies) {
      if (seen.has(s.coin)) continue;
      seen.add(s.coin);
      list.push(s.coin);
    }
    return list
      .sort((a, b) => a.localeCompare(b))
      .map((coin) => ({ value: coin, label: coin.toUpperCase() }));
  }, [strategies]);

  const selectedCoin = Form.useWatch('coin_key', form);

  // Most coins only trade on one exchange, so the strategy label just
  // shows name + timeframe. If a coin is listed on more than one
  // exchange, the exchange is appended so those strategies stay
  // distinguishable without cluttering the common case.
  const strategiesForCoin = useMemo(() => {
    if (!selectedCoin) return [];
    const matches = strategies.filter((s) => s.coin === selectedCoin);
    const exchangeCount = new Set(matches.map((s) => s.exchange)).size;
    return matches.map((s) => ({
      value: s.strategy_id,
      label: exchangeCount > 1
        ? `${s.strategy_name} · ${s.exchange} · ${s.time_horizon}`
        : `${s.strategy_name} · ${s.time_horizon}`,
    }));
  }, [strategies, selectedCoin]);

  const submitRequest = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);

      const payload = {
        strategy_id: values.strategy_id,
        start_date: values.date_range?.[0] ? values.date_range[0].format('YYYY-MM-DD') : undefined,
        end_date: values.date_range?.[1] ? values.date_range[1].format('YYYY-MM-DD') : undefined,
        initial_balance: values.initial_balance,
        commission: values.commission,
        slippage: values.slippage,
        allow_long: values.allow_long,
        allow_short: values.allow_short,
        max_open_positions: values.max_open_positions,
        take_profit: values.take_profit_value != null ? { type: 'percentage', value: values.take_profit_value } : undefined,
        stop_loss: values.stop_loss_value != null ? { type: 'percentage', value: values.stop_loss_value } : undefined,
      };

      const res = await api.post('/api/backtests', payload);
      message.success('Backtest submitted — running in the background.');
      setModalOpen(false);
      form.resetFields();
      load();
      navigate(`/backtests/${res.data.backtest_id}`);
    } catch (err) {
      if (err?.errorFields) return; // antd validation error, already shown inline
      message.error(err.message || 'Failed to submit backtest');
    } finally {
      setSubmitting(false);
    }
  };

  const filtered = tab === 'all' ? backtests : backtests.filter((b) => b.status === tab);

  const counts = useMemo(() => {
    const c = { all: backtests.length, pending: 0, running: 0, completed: 0, failed: 0 };
    for (const b of backtests) c[b.status] = (c[b.status] || 0) + 1;
    return c;
  }, [backtests]);

  const columns = [
    { title: 'ID', dataIndex: 'backtest_id', key: 'backtest_id', width: 70, render: (v) => <span style={{ color: '#9096A0', fontFamily: 'ui-monospace, monospace' }}>#{v}</span> },
    { title: 'Strategy', dataIndex: 'strategy_name', key: 'strategy_name', render: (t) => <span style={{ fontWeight: 600, color: '#F5F6F7' }}>{t}</span> },
    {
      title: 'Status', dataIndex: 'status', key: 'status',
      render: (status) => {
        const meta = STATUS_META[status] || { label: status, bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' };
        return (
          <Tag style={{ background: meta.bg, color: meta.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
            {meta.label}
          </Tag>
        );
      },
    },
    {
      title: 'Date Range', key: 'date_range',
      render: (_, r) => <span style={{ color: '#9096A0', fontSize: 12.5 }}>{r.backtest_config?.start_date} → {r.backtest_config?.end_date}</span>,
    },
    {
      title: 'Net Profit', key: 'total_net_profit',
      render: (_, r) => {
        const v = r.result_summary?.total_net_profit;
        return <span style={{ color: pnlColor(v, '#F5F6F7', '#F5F6F7'), fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{v == null ? '—' : `${v >= 0 ? '+' : ''}${fmtUsd(v)}`}</span>;
      },
    },
    {
      title: 'Trades', key: 'total_trades',
      render: (_, r) => r.result_summary?.total_trades ?? '—',
    },
    {
      title: 'Submitted', dataIndex: 'created_at', key: 'created_at',
      render: (v) => <span style={{ color: '#9096A0', fontSize: 12.5 }}>{new Date(v).toLocaleString()}</span>,
    },
  ];

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Backtest Requests</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            Request new backtests and track pending, running, completed, and failed runs.
          </p>
        </div>
        <button style={primaryBtnStyle} onClick={openModal}>
          <PlusOutlined /> New Backtest
        </button>
      </div>

      {error && (
        <Alert
          type="error"
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load backtests</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={{ background: 'none', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8, color: '#F5F6F7', padding: '4px 10px', cursor: 'pointer' }}>Retry</button>}
          style={{ marginBottom: 20, background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
          showIcon
        />
      )}

      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        {TABS.map((t) => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            style={{
              padding: '7px 16px', borderRadius: 999, fontSize: 13, fontWeight: 600, cursor: 'pointer',
              border: tab === t.key ? `1px solid ${MINT}` : '1px solid rgba(255,255,255,0.08)',
              background: tab === t.key ? 'rgba(61,220,151,0.12)' : 'rgba(255,255,255,0.03)',
              color: tab === t.key ? MINT : '#9096A0',
            }}
          >
            {t.label} {counts[t.key] ? <span style={{ opacity: 0.7 }}>({counts[t.key]})</span> : ''}
          </button>
        ))}
      </div>

      <div style={{ ...panel, padding: 20 }}>
        {loading ? (
          <div style={{ display: 'flex', justifyContent: 'center', padding: '60px 0' }}>
            <Spin size="large" />
          </div>
        ) : (
          <Table
            columns={columns}
            dataSource={filtered.map((b) => ({ ...b, key: b.backtest_id }))}
            pagination={{ pageSize: 10 }}
            locale={{ emptyText: 'No backtests yet — request one to get started.' }}
            onRow={(row) => ({
              onClick: () => navigate(`/backtests/${row.backtest_id}`),
              style: { cursor: 'pointer' },
            })}
          />
        )}
      </div>

      <Modal
        title="New Backtest"
        open={modalOpen}
        onCancel={() => setModalOpen(false)}
        onOk={submitRequest}
        confirmLoading={submitting}
        okText="Run Backtest"
        width={560}
      >
        <Form form={form} layout="vertical" style={{ marginTop: 16 }} initialValues={{
          initial_balance: 10000,
          commission: 0.05,
          slippage: 0.02,
          allow_long: true,
          allow_short: true,
          max_open_positions: 1,
          take_profit_value: 2.0,
          stop_loss_value: 1.0,
        }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <Form.Item name="coin_key" label="Coin" rules={[{ required: true, message: 'Select a coin' }]}>
              <Select
                showSearch
                loading={strategiesLoading}
                placeholder="Select a coin"
                options={coinOptions}
                filterOption={(input, option) => option.label.toLowerCase().includes(input.toLowerCase())}
                onChange={() => form.setFieldValue('strategy_id', undefined)}
              />
            </Form.Item>

            <Form.Item name="strategy_id" label="Strategy" rules={[{ required: true, message: 'Select a strategy' }]}>
              <Select
                showSearch
                disabled={!selectedCoin}
                placeholder={selectedCoin ? 'Select a strategy' : 'Pick a coin first'}
                options={strategiesForCoin}
                filterOption={(input, option) => option.label.toLowerCase().includes(input.toLowerCase())}
                notFoundContent="No strategies for this coin."
              />
            </Form.Item>
          </div>

          <Form.Item name="date_range" label="Date Range" rules={[{ required: true, message: 'Select a date range' }]}>
            <DatePicker.RangePicker
              style={{ width: '100%' }}
              defaultValue={[dayjs().subtract(3, 'month'), dayjs()]}
              disabledDate={(current) => !!current && (current < EARLIEST_DATA_DATE.startOf('day') || current > dayjs().endOf('day'))}
            />
          </Form.Item>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <Form.Item name="initial_balance" label="Initial Capital ($)">
              <InputNumber style={{ width: '100%' }} min={0} />
            </Form.Item>
            <Form.Item name="max_open_positions" label="Max Open Positions">
              <InputNumber style={{ width: '100%' }} min={1} />
            </Form.Item>
            <Form.Item name="commission" label="Commission (%)">
              <InputNumber style={{ width: '100%' }} min={0} step={0.01} />
            </Form.Item>
            <Form.Item name="slippage" label="Slippage (%)">
              <InputNumber style={{ width: '100%' }} min={0} step={0.01} />
            </Form.Item>
            <Form.Item name="take_profit_value" label="Take Profit (%)">
              <InputNumber style={{ width: '100%' }} min={0} step={0.1} />
            </Form.Item>
            <Form.Item name="stop_loss_value" label="Stop Loss (%)">
              <InputNumber style={{ width: '100%' }} min={0} step={0.1} />
            </Form.Item>
          </div>

          <div style={{ display: 'flex', gap: 24, marginTop: 4 }}>
            <Form.Item name="allow_long" label="Allow Long" valuePropName="checked">
              <Switch />
            </Form.Item>
            <Form.Item name="allow_short" label="Allow Short" valuePropName="checked">
              <Switch />
            </Form.Item>
          </div>
        </Form>
      </Modal>
    </div>
  );
}