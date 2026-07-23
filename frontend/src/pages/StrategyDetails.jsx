import { useParams, useNavigate } from 'react-router-dom';
import { Tag } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

// ---- placeholder data — replace with GET /api/strategies/{id} once backend exists ----
const strategy = {
  name: 'BTC Momentum',
  symbol: 'BTCUSDT',
  exchange: 'Bybit',
  timeframe: '4h',
  status: 'Active',
  config: {
    entryLogic: 'EMA(20) crosses above EMA(50) with RSI(14) > 55',
    exitLogic: 'EMA(20) crosses below EMA(50) or stop-loss hit',
    positionSizing: '2% risk per trade',
    maxConcurrentPositions: 1,
  },
  indicators: ['EMA(20)', 'EMA(50)', 'RSI(14)', 'ATR(14)'],
  risk: {
    stopLoss: '1.5x ATR',
    takeProfit: '3x ATR',
    maxDrawdownLimit: '15%',
    maxDailyLoss: '4%',
  },
  performance: {
    latestReturn: 12.4,
    sharpe: 1.8,
    sortino: 2.3,
    winRate: 61,
    profitFactor: 1.9,
    maxDrawdown: -8.2,
  },
  tradeStats: {
    totalTrades: 142,
    wins: 87,
    losses: 55,
    avgWin: 3.1,
    avgLoss: -1.4,
    avgHoldTime: '9h 20m',
  },
};

const recentTrades = [
  { id: 1, side: 'Long', entry: 61240, exit: 62580, pnl: 2.19, date: '2026-07-21' },
  { id: 2, side: 'Long', entry: 59870, exit: 59120, pnl: -1.25, date: '2026-07-19' },
  { id: 3, side: 'Long', entry: 58300, exit: 60110, pnl: 3.10, date: '2026-07-16' },
  { id: 4, side: 'Long', entry: 57200, exit: 56800, pnl: -0.70, date: '2026-07-14' },
  { id: 5, side: 'Long', entry: 55900, exit: 57650, pnl: 3.13, date: '2026-07-11' },
];

const equityCurve = Array.from({ length: 20 }, (_, i) => ({
  day: `D${i + 1}`,
  equity: 10000 + i * 180 + Math.sin(i / 2) * 300,
}));

const drawdown = Array.from({ length: 20 }, (_, i) => ({
  day: `D${i + 1}`,
  dd: -(Math.abs(Math.sin(i / 3)) * 8).toFixed(2),
}));

const monthlyReturns = [
  { month: 'Feb', ret: 4.2 }, { month: 'Mar', ret: -1.8 }, { month: 'Apr', ret: 6.1 },
  { month: 'May', ret: 2.4 }, { month: 'Jun', ret: -3.0 }, { month: 'Jul', ret: 5.7 },
];

const tradeDistribution = [
  { name: 'Wins', value: 87 },
  { name: 'Losses', value: 55 },
];

const panel = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

function Panel({ title, children, style }) {
  return (
    <div style={{ ...panel, padding: 22, ...style }}>
      {title && <h3 style={{ fontSize: 15.5, fontWeight: 700, color: '#F5F6F7', margin: '0 0 16px' }}>{title}</h3>}
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

export default function StrategyDetails() {
  const { id } = useParams();
  const navigate = useNavigate();

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
            <h2 style={{ fontSize: 22, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{strategy.name}</h2>
            <Tag style={{ background: 'rgba(61,220,151,0.12)', color: MINT, border: 'none', borderRadius: 8, fontWeight: 600 }}>
              {strategy.status}
            </Tag>
          </div>
          <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2 }}>
            {strategy.symbol} · {strategy.exchange} · {strategy.timeframe} · Strategy ID {id}
          </div>
        </div>
      </div>

      {/* Performance summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Latest Return" value={`+${strategy.performance.latestReturn}%`} positive />
        <StatBox label="Sharpe Ratio" value={strategy.performance.sharpe} />
        <StatBox label="Sortino Ratio" value={strategy.performance.sortino} />
        <StatBox label="Win Rate" value={`${strategy.performance.winRate}%`} />
        <StatBox label="Profit Factor" value={strategy.performance.profitFactor} />
        <StatBox label="Max Drawdown" value={`${strategy.performance.maxDrawdown}%`} positive={false} />
      </div>

      {/* Config / Indicators / Risk */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 20, marginBottom: 20 }}>
        <Panel title="Strategy Configuration">
          <KeyValue label="Entry Logic" value={strategy.config.entryLogic} />
          <KeyValue label="Exit Logic" value={strategy.config.exitLogic} />
          <KeyValue label="Position Sizing" value={strategy.config.positionSizing} />
          <KeyValue label="Max Concurrent Positions" value={strategy.config.maxConcurrentPositions} />
        </Panel>
        <Panel title="Indicators Used">
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {strategy.indicators.map((ind) => (
              <span
                key={ind}
                style={{
                  background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)',
                  color: '#F5F6F7', fontSize: 12.5, fontWeight: 600, padding: '6px 12px', borderRadius: 999,
                }}
              >
                {ind}
              </span>
            ))}
          </div>
        </Panel>
        <Panel title="Risk Management">
          <KeyValue label="Stop Loss" value={strategy.risk.stopLoss} />
          <KeyValue label="Take Profit" value={strategy.risk.takeProfit} />
          <KeyValue label="Max Drawdown Limit" value={strategy.risk.maxDrawdownLimit} />
          <KeyValue label="Max Daily Loss" value={strategy.risk.maxDailyLoss} />
        </Panel>
      </div>

      {/* Equity Curve + Drawdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Equity Curve">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={equityCurve} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="eqGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={MINT} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={MINT} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 }} />
              <Area type="monotone" dataKey="equity" stroke={MINT} strokeWidth={2.5} fill="url(#eqGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Drawdown">
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={drawdown} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={RED} stopOpacity={0} />
                  <stop offset="95%" stopColor={RED} stopOpacity={0.35} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 }} />
              <Area type="monotone" dataKey="dd" stroke={RED} strokeWidth={2.5} fill="url(#ddGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Monthly Returns + Trade Distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Monthly Returns">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={monthlyReturns} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="month" tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={{ background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 }} />
              <Bar dataKey="ret" radius={[6, 6, 6, 6]}>
                {monthlyReturns.map((entry, i) => (
                  <Cell key={i} fill={entry.ret >= 0 ? MINT : RED} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Trade Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={tradeDistribution} dataKey="value" nameKey="name"
                cx="50%" cy="50%" innerRadius={55} outerRadius={80} paddingAngle={3}
              >
                <Cell fill={MINT} />
                <Cell fill={RED} />
              </Pie>
              <Legend
                verticalAlign="middle" align="right" layout="vertical"
                iconType="circle" wrapperStyle={{ fontSize: 12.5, color: '#9096A0' }}
              />
              <Tooltip contentStyle={{ background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 }} />
            </PieChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Trade stats + Recent trades */}
      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 20 }}>
        <Panel title="Trade Statistics">
          <KeyValue label="Total Trades" value={strategy.tradeStats.totalTrades} mono />
          <KeyValue label="Wins" value={strategy.tradeStats.wins} mono />
          <KeyValue label="Losses" value={strategy.tradeStats.losses} mono />
          <KeyValue label="Avg Win" value={`+${strategy.tradeStats.avgWin}%`} mono />
          <KeyValue label="Avg Loss" value={`${strategy.tradeStats.avgLoss}%`} mono />
          <KeyValue label="Avg Hold Time" value={strategy.tradeStats.avgHoldTime} mono />
        </Panel>
        <Panel title="Recent Trades">
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '80px 1fr 1fr 1fr 100px', padding: '0 4px 10px', color: '#6B7280', fontSize: 11.5, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 0.4 }}>
              <span>Side</span><span>Entry</span><span>Exit</span><span>Date</span><span style={{ textAlign: 'right' }}>PnL</span>
            </div>
            {recentTrades.map((t, i) => (
              <div
                key={t.id}
                style={{
                  display: 'grid', gridTemplateColumns: '80px 1fr 1fr 1fr 100px', padding: '12px 4px',
                  borderTop: '1px solid rgba(255,255,255,0.05)', alignItems: 'center',
                }}
              >
                <Tag style={{ background: 'rgba(61,220,151,0.1)', color: MINT, border: 'none', width: 'fit-content', borderRadius: 6 }}>{t.side}</Tag>
                <span style={{ color: '#F5F6F7', fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>${t.entry.toLocaleString()}</span>
                <span style={{ color: '#F5F6F7', fontFamily: 'ui-monospace, monospace', fontSize: 13 }}>${t.exit.toLocaleString()}</span>
                <span style={{ color: '#9096A0', fontSize: 13 }}>{t.date}</span>
                <span style={{ color: t.pnl >= 0 ? MINT : RED, fontWeight: 600, fontFamily: 'ui-monospace, monospace', fontSize: 13, textAlign: 'right' }}>
                  {t.pnl >= 0 ? '+' : ''}{t.pnl}%
                </span>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}