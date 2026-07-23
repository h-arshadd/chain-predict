import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table, message } from 'antd';
import { ArrowLeftOutlined, DownloadOutlined, FileTextOutlined } from '@ant-design/icons';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

// ---- placeholder data — replace with GET /api/backtests/{id} once backend exists ----
const backtest = {
  strategy: 'BTC Momentum',
  symbol: 'BTCUSDT',
  exchange: 'Bybit',
  timeframe: '4h',
  dateRange: '2025-01-01 → 2026-01-01',
  requestedAt: '2026-07-20 10:12',
  completedAt: '2026-07-20 10:19',
  config: {
    initialCapital: 10000,
    commission: 0.075,
    slippage: 0.05,
    riskPerTrade: '2%',
    maxDrawdownLimit: '15%',
    entryLogic: 'EMA(20) crosses above EMA(50) with RSI(14) > 55',
    exitLogic: 'EMA(20) crosses below EMA(50) or stop-loss hit',
  },
  stats: {
    finalReturn: 34.2,
    finalCapital: 13420,
    sharpe: 1.9,
    sortino: 2.4,
    winRate: 63,
    profitFactor: 2.1,
    maxDrawdown: -9.4,
    totalTrades: 186,
    avgWin: 3.4,
    avgLoss: -1.6,
    avgHoldTime: '11h 05m',
    bestTrade: 8.9,
    worstTrade: -4.2,
  },
};

const tradeList = [
  { id: 1, side: 'Long', entry: 61240, exit: 62580, size: 0.42, pnl: 562.8, opened: '2026-07-21 09:40', closed: '2026-07-22 03:10' },
  { id: 2, side: 'Long', entry: 58300, exit: 60110, size: 0.38, pnl: 688.0, opened: '2026-07-16 09:20', closed: '2026-07-18 14:05' },
  { id: 3, side: 'Long', entry: 57200, exit: 56800, size: 0.30, pnl: -120.0, opened: '2026-07-14 03:10', closed: '2026-07-14 22:40' },
  { id: 4, side: 'Long', entry: 55900, exit: 57650, size: 0.35, pnl: 612.5, opened: '2026-07-11 11:00', closed: '2026-07-13 08:15' },
  { id: 5, side: 'Long', entry: 54200, exit: 53780, size: 0.28, pnl: -117.6, opened: '2026-07-08 15:30', closed: '2026-07-09 04:20' },
];

const orders = [
  { id: 1, type: 'Market', side: 'Buy', symbol: 'BTCUSDT', price: 61240, qty: 0.42, fee: 4.29, time: '2026-07-21 09:40' },
  { id: 2, type: 'Limit', side: 'Sell', symbol: 'BTCUSDT', price: 62580, qty: 0.42, fee: 4.38, time: '2026-07-22 03:10' },
  { id: 3, type: 'Limit', side: 'Buy', symbol: 'BTCUSDT', price: 58300, qty: 0.38, fee: 3.88, time: '2026-07-16 09:20' },
  { id: 4, type: 'Limit', side: 'Sell', symbol: 'BTCUSDT', price: 60110, qty: 0.38, fee: 3.96, time: '2026-07-18 14:05' },
];

const equityCurve = Array.from({ length: 24 }, (_, i) => ({
  month: `M${i + 1}`,
  equity: 10000 + i * 145 + Math.sin(i / 2.5) * 260,
}));

const drawdown = Array.from({ length: 24 }, (_, i) => ({
  month: `M${i + 1}`,
  dd: -(Math.abs(Math.sin(i / 3)) * 9.4).toFixed(2),
}));

const monthlyReturns = [
  { month: 'Jan', ret: 4.2 }, { month: 'Feb', ret: -1.8 }, { month: 'Mar', ret: 6.1 },
  { month: 'Apr', ret: 2.4 }, { month: 'May', ret: -3.0 }, { month: 'Jun', ret: 5.7 },
  { month: 'Jul', ret: 3.1 }, { month: 'Aug', ret: -0.9 }, { month: 'Sep', ret: 4.8 },
  { month: 'Oct', ret: 2.2 }, { month: 'Nov', ret: -2.1 }, { month: 'Dec', ret: 5.3 },
];

const rollingMetrics = Array.from({ length: 24 }, (_, i) => ({
  month: `M${i + 1}`,
  sharpe: 1.2 + Math.sin(i / 4) * 0.6 + i * 0.02,
}));

const panel = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

const tooltipStyle = { background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 };
const axisStyle = { fill: '#6B7280', fontSize: 11 };

function Panel({ title, children, style, action }) {
  return (
    <div style={{ ...panel, padding: 22, ...style }}>
      {(title || action) && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          {title && <h3 style={{ fontSize: 15.5, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{title}</h3>}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

function KeyValue({ label, value, mono }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <span style={{ color: '#9096A0', fontSize: 13 }}>{label}</span>
      <span style={{ color: '#F5F6F7', fontSize: 13, fontWeight: 600, fontFamily: mono ? 'ui-monospace, monospace' : undefined }}>
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

function exportReport(format) {
  // Placeholder — wire to GET /api/backtests/{id}/export?format= once backend exists
  message.success(`Exporting report as ${format.toUpperCase()}...`);
}

export default function BacktestDetails() {
  const { id } = useParams();
  const navigate = useNavigate();

  return (
    <div style={{ paddingTop: 8 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
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
              <h2 style={{ fontSize: 22, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{backtest.strategy}</h2>
              <Tag style={{ background: 'rgba(61,220,151,0.12)', color: MINT, border: 'none', borderRadius: 8, fontWeight: 600 }}>
                Completed
              </Tag>
            </div>
            <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2 }}>
              {backtest.symbol} &middot; {backtest.exchange} &middot; {backtest.timeframe} &middot; {backtest.dateRange} &middot; Backtest ID {id}
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button onClick={() => exportReport('pdf')} style={secondaryBtnStyle}>
            <FileTextOutlined /> Export PDF
          </button>
          <button onClick={() => exportReport('csv')} style={secondaryBtnStyle}>
            <DownloadOutlined /> Export CSV
          </button>
        </div>
      </div>

      {/* Performance summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Final Return" value={`+${backtest.stats.finalReturn}%`} positive />
        <StatBox label="Final Capital" value={`$${backtest.stats.finalCapital.toLocaleString()}`} />
        <StatBox label="Sharpe Ratio" value={backtest.stats.sharpe} />
        <StatBox label="Sortino Ratio" value={backtest.stats.sortino} />
        <StatBox label="Win Rate" value={`${backtest.stats.winRate}%`} />
        <StatBox label="Profit Factor" value={backtest.stats.profitFactor} />
        <StatBox label="Max Drawdown" value={`${backtest.stats.maxDrawdown}%`} positive={false} />
      </div>

      {/* Complete statistics + Strategy configuration */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Complete Statistics">
          <KeyValue label="Total Trades" value={backtest.stats.totalTrades} />
          <KeyValue label="Average Win" value={`+${backtest.stats.avgWin}%`} />
          <KeyValue label="Average Loss" value={`${backtest.stats.avgLoss}%`} />
          <KeyValue label="Average Hold Time" value={backtest.stats.avgHoldTime} />
          <KeyValue label="Best Trade" value={`+${backtest.stats.bestTrade}%`} />
          <KeyValue label="Worst Trade" value={`${backtest.stats.worstTrade}%`} />
        </Panel>
        <Panel title="Strategy Configuration">
          <KeyValue label="Initial Capital" value={`$${backtest.config.initialCapital.toLocaleString()}`} mono />
          <KeyValue label="Commission" value={`${backtest.config.commission}%`} />
          <KeyValue label="Slippage" value={`${backtest.config.slippage}%`} />
          <KeyValue label="Risk Per Trade" value={backtest.config.riskPerTrade} />
          <KeyValue label="Max Drawdown Limit" value={backtest.config.maxDrawdownLimit} />
          <KeyValue label="Entry Logic" value={backtest.config.entryLogic} />
        </Panel>
      </div>

      {/* Equity Curve + Drawdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Equity Curve">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={equityCurve} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="btEqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={MINT} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={MINT} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="month" tick={axisStyle} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="equity" stroke={MINT} strokeWidth={2.5} fill="url(#btEqGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Drawdown">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={drawdown} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="btDdGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={RED} stopOpacity={0} />
                  <stop offset="95%" stopColor={RED} stopOpacity={0.35} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="month" tick={axisStyle} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="dd" stroke={RED} strokeWidth={2.5} fill="url(#btDdGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Monthly Returns + Rolling Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Monthly Returns">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={monthlyReturns} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="month" tick={axisStyle} axisLine={false} tickLine={false} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="ret" radius={[6, 6, 6, 6]}>
                {monthlyReturns.map((entry, i) => (
                  <Cell key={i} fill={entry.ret >= 0 ? MINT : RED} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Rolling Metrics (Sharpe)">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={rollingMetrics} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="month" tick={axisStyle} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="sharpe" stroke={AMBER} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Trade List */}
      <div style={{ marginBottom: 20 }}>
        <Panel title="Trade List">
          <Table
            size="small"
            pagination={{ pageSize: 5 }}
            dataSource={tradeList.map((r) => ({ ...r, key: r.id }))}
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

      {/* Orders */}
      <div style={{ marginBottom: 8 }}>
        <Panel title="Orders">
          <Table
            size="small"
            pagination={{ pageSize: 5 }}
            dataSource={orders.map((r) => ({ ...r, key: r.id }))}
            columns={[
              { title: 'Type', dataIndex: 'type', key: 'type' },
              { title: 'Side', dataIndex: 'side', key: 'side', render: (v) => <span style={{ color: v === 'Buy' ? MINT : RED, fontWeight: 600 }}>{v}</span> },
              { title: 'Symbol', dataIndex: 'symbol', key: 'symbol' },
              { title: 'Price', dataIndex: 'price', key: 'price', render: (v) => v.toLocaleString() },
              { title: 'Qty', dataIndex: 'qty', key: 'qty' },
              { title: 'Fee', dataIndex: 'fee', key: 'fee' },
              { title: 'Time', dataIndex: 'time', key: 'time' },
            ]}
          />
        </Panel>
      </div>
    </div>
  );
}

const secondaryBtnStyle = {
  display: 'flex', alignItems: 'center', gap: 8,
  background: 'rgba(255,255,255,0.04)', color: '#F5F6F7', border: '1px solid rgba(255,255,255,0.1)',
  fontSize: 13.5, fontWeight: 600, padding: '9px 16px',
  borderRadius: 999, cursor: 'pointer',
};