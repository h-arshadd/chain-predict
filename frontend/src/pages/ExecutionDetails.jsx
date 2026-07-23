import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

// ---- placeholder data — replace with GET /api/executions/{id} once backend exists ----
const execution = {
  strategy: { name: 'BTC Momentum', symbol: 'BTCUSDT', timeframe: '4h', entryLogic: 'EMA(20) crosses above EMA(50) with RSI(14) > 55', exitLogic: 'EMA(20) crosses below EMA(50) or stop-loss hit' },
  wallet: { label: 'Main Trading', exchange: 'Bybit', accountType: 'Unified Trading (UTA)', apiStatus: 'Connected' },
  status: 'Running',
  currentPosition: { side: 'Long', size: 0.42, entryPrice: 61240, markPrice: 62580, unrealizedPnl: 562.8, leverage: '3x' },
  risk: { stopLoss: 59800, takeProfit: 64200, maxDrawdownLimit: '15%', riskPerTrade: '2%', currentDrawdown: '-3.1%' },
};

const positionHistory = [
  { id: 1, side: 'Long', size: 0.38, entry: 58300, exit: 60110, pnl: 688.0, opened: '2026-07-16 09:20', closed: '2026-07-18 14:05' },
  { id: 2, side: 'Long', size: 0.30, entry: 57200, exit: 56800, pnl: -120.0, opened: '2026-07-14 03:10', closed: '2026-07-14 22:40' },
  { id: 3, side: 'Long', size: 0.35, entry: 55900, exit: 57650, pnl: 612.5, opened: '2026-07-11 11:00', closed: '2026-07-13 08:15' },
];

const filledOrders = [
  { id: 1, type: 'Market', side: 'Buy', price: 61240, qty: 0.42, fee: 4.29, time: '2026-07-21 09:40' },
  { id: 2, type: 'Limit', side: 'Buy', price: 58300, qty: 0.38, fee: 3.88, time: '2026-07-16 09:20' },
  { id: 3, type: 'Limit', side: 'Sell', price: 60110, qty: 0.38, fee: 3.96, time: '2026-07-18 14:05' },
];

const openOrders = [
  { id: 1, type: 'Limit', side: 'Sell', price: 64200, qty: 0.42, status: 'Working' },
  { id: 2, type: 'Stop', side: 'Sell', price: 59800, qty: 0.42, status: 'Working' },
];

const signalHistory = [
  { id: 1, signal: 'Buy', reason: 'EMA(20) crossed above EMA(50), RSI 58.2', time: '2026-07-21 09:40' },
  { id: 2, signal: 'Hold', reason: 'No crossover, RSI neutral', time: '2026-07-20 05:40' },
  { id: 3, signal: 'Sell', reason: 'EMA(20) crossed below EMA(50)', time: '2026-07-18 14:05' },
  { id: 4, signal: 'Buy', reason: 'EMA(20) crossed above EMA(50), RSI 61.4', time: '2026-07-16 09:20' },
];

const equityCurve = Array.from({ length: 20 }, (_, i) => ({
  day: `D${i + 1}`,
  equity: 24000 + i * 45 + Math.sin(i / 2) * 220,
}));

const positionSize = Array.from({ length: 20 }, (_, i) => ({
  day: `D${i + 1}`,
  size: Math.max(0, 0.2 + Math.sin(i / 3) * 0.15 + i * 0.01),
}));

const dailyReturns = Array.from({ length: 14 }, (_, i) => ({
  day: `D${i + 1}`,
  ret: Math.sin(i * 0.7) * 2.4,
}));

const panel = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

const statusColors = {
  Running: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  Paused: { bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  Stopped: { bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
  Error: { bg: 'rgba(240,70,107,0.14)', fg: RED },
};

function Panel({ title, children, style }) {
  return (
    <div style={{ ...panel, padding: 22, ...style }}>
      {title && <h3 style={{ fontSize: 15.5, fontWeight: 700, color: '#F5F6F7', margin: '0 0 16px' }}>{title}</h3>}
      {children}
    </div>
  );
}

function KeyValue({ label, value, mono, color }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <span style={{ color: '#9096A0', fontSize: 13 }}>{label}</span>
      <span style={{ color: color || '#F5F6F7', fontSize: 13, fontWeight: 600, fontFamily: mono ? 'ui-monospace, monospace' : undefined }}>
        {value}
      </span>
    </div>
  );
}

function StatBox({ label, value, positive }) {
  return (
    <div style={{ ...panel, padding: 16 }}>
      <div style={{ color: '#9096A0', fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 19, fontWeight: 700, color: positive === undefined ? '#F5F6F7' : positive ? MINT : RED, fontFamily: 'ui-monospace, monospace' }}>
        {value}
      </div>
    </div>
  );
}

const tooltipStyle = { background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 };
const axisStyle = { fill: '#6B7280', fontSize: 11 };

export default function ExecutionDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const statusStyle = statusColors[execution.status] || statusColors.Stopped;

  return (
    <div style={{ paddingTop: 8 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
        <button
          onClick={() => navigate(-1)}
          style={{
            width: 36, height: 36, borderRadius: 10, border: '1px solid rgba(255,255,255,0.1)',
            background: 'rgba(255,255,255,0.04)', color: '#F5F6F7', cursor: 'pointer',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
          }}
        >
          <ArrowLeftOutlined />
        </button>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <h2 style={{ fontSize: 22, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{execution.strategy.name}</h2>
            <Tag style={{ background: statusStyle.bg, color: statusStyle.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
              {execution.status}
            </Tag>
          </div>
          <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2 }}>
            {execution.strategy.symbol} &middot; {execution.wallet.exchange} &middot; {execution.wallet.label} &middot; Execution ID {id}
          </div>
        </div>
      </div>

      {/* Current position summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Position" value={`${execution.currentPosition.side} ${execution.currentPosition.size} BTC`} />
        <StatBox label="Entry Price" value={execution.currentPosition.entryPrice.toLocaleString()} />
        <StatBox label="Mark Price" value={execution.currentPosition.markPrice.toLocaleString()} />
        <StatBox label="Unrealized PnL" value={`+${execution.currentPosition.unrealizedPnl.toFixed(2)}`} positive />
        <StatBox label="Leverage" value={execution.currentPosition.leverage} />
        <StatBox label="Current Drawdown" value={execution.risk.currentDrawdown} positive={false} />
      </div>

      {/* Strategy info / Wallet info / Risk statistics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginBottom: 20 }}>
        <Panel title="Strategy Information">
          <KeyValue label="Symbol" value={execution.strategy.symbol} />
          <KeyValue label="Timeframe" value={execution.strategy.timeframe} />
          <KeyValue label="Entry Logic" value={execution.strategy.entryLogic} />
          <KeyValue label="Exit Logic" value={execution.strategy.exitLogic} />
        </Panel>
        <Panel title="Wallet Information">
          <KeyValue label="Wallet" value={execution.wallet.label} />
          <KeyValue label="Exchange" value={execution.wallet.exchange} />
          <KeyValue label="Account Type" value={execution.wallet.accountType} />
          <KeyValue
            label="API Status"
            value={execution.wallet.apiStatus}
            color={execution.wallet.apiStatus === 'Connected' ? MINT : RED}
          />
        </Panel>
        <Panel title="Risk Statistics">
          <KeyValue label="Stop Loss" value={execution.risk.stopLoss.toLocaleString()} mono />
          <KeyValue label="Take Profit" value={execution.risk.takeProfit.toLocaleString()} mono />
          <KeyValue label="Max Drawdown Limit" value={execution.risk.maxDrawdownLimit} />
          <KeyValue label="Risk Per Trade" value={execution.risk.riskPerTrade} />
        </Panel>
      </div>

      {/* Equity Curve + Position Size */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Equity Curve">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={equityCurve} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="eqGrad2" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={MINT} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={MINT} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={axisStyle} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="equity" stroke={MINT} strokeWidth={2.5} fill="url(#eqGrad2)" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Position Size">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={positionSize} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="posGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={AMBER} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={AMBER} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={axisStyle} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="size" stroke={AMBER} strokeWidth={2.5} fill="url(#posGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Daily Returns + Trade History (as chart) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Daily Returns">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={dailyReturns} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={axisStyle} axisLine={false} tickLine={false} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="ret" radius={[6, 6, 6, 6]}>
                {dailyReturns.map((entry, i) => (
                  <Cell key={i} fill={entry.ret >= 0 ? MINT : RED} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Trade History">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={positionHistory.map((t) => ({ id: `#${t.id}`, pnl: t.pnl }))} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="id" tick={axisStyle} axisLine={false} tickLine={false} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="pnl" stroke={MINT} strokeWidth={2.5} dot={{ r: 4, fill: MINT }} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Position History */}
      <div style={{ marginBottom: 20 }}>
        <Panel title="Position History">
          <Table
            size="small"
            pagination={false}
            dataSource={positionHistory.map((r) => ({ ...r, key: r.id }))}
            columns={[
              { title: 'Side', dataIndex: 'side', key: 'side', render: (v) => <span style={{ color: v === 'Long' ? MINT : RED, fontWeight: 600 }}>{v}</span> },
              { title: 'Size (BTC)', dataIndex: 'size', key: 'size' },
              { title: 'Entry', dataIndex: 'entry', key: 'entry', render: (v) => v.toLocaleString() },
              { title: 'Exit', dataIndex: 'exit', key: 'exit', render: (v) => v.toLocaleString() },
              { title: 'PnL', dataIndex: 'pnl', key: 'pnl', render: (v) => <span style={{ color: v >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}</span> },
              { title: 'Opened', dataIndex: 'opened', key: 'opened' },
              { title: 'Closed', dataIndex: 'closed', key: 'closed' },
            ]}
          />
        </Panel>
      </div>

      {/* Filled Orders + Open Orders */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Filled Orders">
          <Table
            size="small"
            pagination={false}
            dataSource={filledOrders.map((r) => ({ ...r, key: r.id }))}
            columns={[
              { title: 'Type', dataIndex: 'type', key: 'type' },
              { title: 'Side', dataIndex: 'side', key: 'side', render: (v) => <span style={{ color: v === 'Buy' ? MINT : RED, fontWeight: 600 }}>{v}</span> },
              { title: 'Price', dataIndex: 'price', key: 'price', render: (v) => v.toLocaleString() },
              { title: 'Qty', dataIndex: 'qty', key: 'qty' },
              { title: 'Fee', dataIndex: 'fee', key: 'fee' },
              { title: 'Time', dataIndex: 'time', key: 'time' },
            ]}
          />
        </Panel>
        <Panel title="Open Orders">
          <Table
            size="small"
            pagination={false}
            dataSource={openOrders.map((r) => ({ ...r, key: r.id }))}
            columns={[
              { title: 'Type', dataIndex: 'type', key: 'type' },
              { title: 'Side', dataIndex: 'side', key: 'side', render: (v) => <span style={{ color: v === 'Buy' ? MINT : RED, fontWeight: 600 }}>{v}</span> },
              { title: 'Price', dataIndex: 'price', key: 'price', render: (v) => v.toLocaleString() },
              { title: 'Qty', dataIndex: 'qty', key: 'qty' },
              {
                title: 'Status', dataIndex: 'status', key: 'status',
                render: (v) => (
                  <Tag style={{ background: 'rgba(61,220,151,0.12)', color: MINT, border: 'none', borderRadius: 8, fontWeight: 600 }}>
                    {v}
                  </Tag>
                ),
              },
            ]}
          />
        </Panel>
      </div>

      {/* Signal History */}
      <div style={{ marginBottom: 8 }}>
        <Panel title="Signal History">
          <Table
            size="small"
            pagination={false}
            dataSource={signalHistory.map((r) => ({ ...r, key: r.id }))}
            columns={[
              {
                title: 'Signal', dataIndex: 'signal', key: 'signal',
                render: (v) => {
                  const c = v === 'Buy' ? MINT : v === 'Sell' ? RED : '#9096A0';
                  return <Tag style={{ background: 'rgba(255,255,255,0.06)', color: c, border: 'none', borderRadius: 8, fontWeight: 600 }}>{v}</Tag>;
                },
              },
              { title: 'Reason', dataIndex: 'reason', key: 'reason', render: (v) => <span style={{ color: '#9096A0' }}>{v}</span> },
              { title: 'Time', dataIndex: 'time', key: 'time' },
            ]}
          />
        </Panel>
      </div>
    </div>
  );
}