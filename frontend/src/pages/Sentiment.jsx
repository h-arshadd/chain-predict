import { useState } from 'react';
import { Tag, Select, Table } from 'antd';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const COINS = ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA', 'LTC', 'MINA', 'SUI'];

// ---- placeholder data — replace with GET /api/sentiment once backend exists ----
const fearGreed = {
  score: 62,
  label: 'Greed',
  yesterday: 58,
  lastWeek: 47,
  lastMonth: 39,
};

const overallSentiment = {
  score: 0.34, // -1 (very negative) to +1 (very positive)
  label: 'Bullish',
  bullishPct: 58,
  neutralPct: 24,
  bearishPct: 18,
};

const newsSentiment = [
  { id: 1, headline: 'Bybit reports record derivatives volume amid ETF optimism', source: 'CoinDesk', sentiment: 'Positive', score: 0.62, time: '2h ago' },
  { id: 2, headline: 'Regulators signal tighter stablecoin oversight in Q3', source: 'Reuters', sentiment: 'Negative', score: -0.41, time: '4h ago' },
  { id: 3, headline: 'SOL network upgrade completes without incident', source: 'The Block', sentiment: 'Positive', score: 0.55, time: '6h ago' },
  { id: 4, headline: 'DOGE whale wallets show accumulation pattern', source: 'CryptoSlate', sentiment: 'Neutral', score: 0.08, time: '9h ago' },
  { id: 5, headline: 'Analysts warn of overheated leverage in perp markets', source: 'Bloomberg', sentiment: 'Negative', score: -0.38, time: '12h ago' },
  { id: 6, headline: 'ADA ecosystem TVL climbs to yearly high', source: 'CoinDesk', sentiment: 'Positive', score: 0.47, time: '15h ago' },
];

const fearGreedTimeline = Array.from({ length: 30 }, (_, i) => ({
  day: `D${i + 1}`,
  value: Math.max(5, Math.min(95, 45 + Math.sin(i / 4) * 25 + i * 0.5)),
}));

const sentimentTimeline = Array.from({ length: 30 }, (_, i) => ({
  day: `D${i + 1}`,
  value: Math.sin(i / 5) * 0.5 + Math.sin(i / 2) * 0.15,
}));

const newsVolume = Array.from({ length: 14 }, (_, i) => ({
  day: `D${i + 1}`,
  positive: Math.round(8 + Math.sin(i / 2) * 4 + i * 0.3),
  negative: Math.round(4 + Math.cos(i / 3) * 3),
  neutral: Math.round(3 + Math.sin(i / 4) * 2),
}));

const sentimentDistribution = [
  { name: 'Bullish', value: overallSentiment.bullishPct },
  { name: 'Neutral', value: overallSentiment.neutralPct },
  { name: 'Bearish', value: overallSentiment.bearishPct },
];

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

// ---- Fear & Greed gauge: semicircular dial with needle ----
function fearGreedColor(score) {
  if (score < 25) return RED;
  if (score < 45) return AMBER;
  if (score < 55) return '#E8C547';
  if (score < 75) return MINT;
  return '#2FBF80';
}

function fearGreedLabel(score) {
  if (score < 20) return 'Extreme Fear';
  if (score < 40) return 'Fear';
  if (score < 60) return 'Neutral';
  if (score < 80) return 'Greed';
  return 'Extreme Greed';
}

function FearGreedGauge({ score }) {
  const cx = 150, cy = 150, r = 110;
  const angle = 180 - (score / 100) * 180; // 0 -> 180deg (left), 100 -> 0deg (right)
  const rad = (angle * Math.PI) / 180;
  const needleX = cx + r * 0.82 * Math.cos(rad);
  const needleY = cy - r * 0.82 * Math.sin(rad);

  const segments = [
    { from: 0, to: 20, color: RED },
    { from: 20, to: 40, color: AMBER },
    { from: 40, to: 60, color: '#E8C547' },
    { from: 60, to: 80, color: MINT },
    { from: 80, to: 100, color: '#2FBF80' },
  ];

  const polarToCartesian = (angleDeg, radius) => {
    const a = (angleDeg * Math.PI) / 180;
    return { x: cx + radius * Math.cos(a), y: cy - radius * Math.sin(a) };
  };

  const arcPath = (fromPct, toPct, radius) => {
    const fromAngle = 180 - (fromPct / 100) * 180;
    const toAngle = 180 - (toPct / 100) * 180;
    const start = polarToCartesian(fromAngle, radius);
    const end = polarToCartesian(toAngle, radius);
    const largeArc = Math.abs(fromAngle - toAngle) > 180 ? 1 : 0;
    return `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 0 ${end.x} ${end.y}`;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <svg width="300" height="175" viewBox="0 0 300 175">
        {segments.map((seg, i) => (
          <path
            key={i}
            d={arcPath(seg.from, seg.to, r)}
            fill="none"
            stroke={seg.color}
            strokeWidth={18}
            strokeLinecap="butt"
            opacity={0.85}
          />
        ))}
        {/* Needle */}
        <line
          x1={cx} y1={cy} x2={needleX} y2={needleY}
          stroke="#F5F6F7" strokeWidth={3} strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={7} fill="#F5F6F7" />
        <circle cx={cx} cy={cy} r={3} fill="#0B0E11" />
      </svg>
      <div style={{ marginTop: -8, textAlign: 'center' }}>
        <div style={{ fontSize: 40, fontWeight: 800, color: fearGreedColor(score), fontFamily: 'ui-monospace, monospace', lineHeight: 1 }}>
          {score}
        </div>
        <div style={{ fontSize: 15, fontWeight: 700, color: fearGreedColor(score), marginTop: 6 }}>
          {fearGreedLabel(score)}
        </div>
      </div>
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

const sentimentTagColor = { Positive: MINT, Negative: RED, Neutral: '#9096A0' };

export default function Sentiment() {
  const [coinFilter, setCoinFilter] = useState('All');

  const filteredNews = coinFilter === 'All'
    ? newsSentiment
    : newsSentiment.filter((n) => n.headline.toUpperCase().includes(coinFilter));

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Sentiment</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            Market mood, news sentiment, and historical trends across tracked coins.
          </p>
        </div>
        <Select
          value={coinFilter}
          onChange={setCoinFilter}
          style={{ width: 160 }}
          options={[{ value: 'All', label: 'All coins' }, ...COINS.map((c) => ({ value: c, label: c }))]}
        />
      </div>

      {/* Fear & Greed Index + Overall Market Sentiment */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Fear & Greed Index">
          <div style={{ display: 'flex', justifyContent: 'center' }}>
            <FearGreedGauge score={fearGreed.score} />
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 12 }}>
            <StatBox label="Yesterday" value={fearGreed.yesterday} />
            <StatBox label="Last Week" value={fearGreed.lastWeek} />
            <StatBox label="Last Month" value={fearGreed.lastMonth} />
          </div>
        </Panel>

        <Panel title="Overall Market Sentiment">
          <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
            <div style={{ position: 'relative', width: 160, height: 160, flexShrink: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={sentimentDistribution} dataKey="value" nameKey="name"
                    cx="50%" cy="50%" innerRadius={55} outerRadius={75} paddingAngle={3} startAngle={90} endAngle={-270}
                  >
                    <Cell fill={MINT} />
                    <Cell fill="#9096A0" />
                    <Cell fill={RED} />
                  </Pie>
                </PieChart>
              </ResponsiveContainer>
              <div style={{
                position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)',
                textAlign: 'center',
              }}>
                <div style={{ fontSize: 24, fontWeight: 800, color: overallSentiment.score >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace' }}>
                  {overallSentiment.score >= 0 ? '+' : ''}{overallSentiment.score.toFixed(2)}
                </div>
                <div style={{ fontSize: 12, color: '#9096A0', fontWeight: 600 }}>{overallSentiment.label}</div>
              </div>
            </div>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
              <SentimentBar label="Bullish" pct={overallSentiment.bullishPct} color={MINT} />
              <SentimentBar label="Neutral" pct={overallSentiment.neutralPct} color="#9096A0" />
              <SentimentBar label="Bearish" pct={overallSentiment.bearishPct} color={RED} />
            </div>
          </div>
        </Panel>
      </div>

      {/* Fear & Greed Timeline + Sentiment Timeline */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Fear & Greed Timeline">
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={fearGreedTimeline} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="fgGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={AMBER} stopOpacity={0.35} />
                  <stop offset="95%" stopColor={AMBER} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={axisStyle} axisLine={false} tickLine={false} interval={4} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={[0, 100]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Area type="monotone" dataKey="value" stroke={AMBER} strokeWidth={2.5} fill="url(#fgGrad)" />
            </AreaChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Sentiment Timeline">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={sentimentTimeline} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={axisStyle} axisLine={false} tickLine={false} interval={4} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={[-1, 1]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="value" stroke={MINT} strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* News Volume + Sentiment Distribution */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="News Volume">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={newsVolume} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="day" tick={axisStyle} axisLine={false} tickLine={false} interval={2} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="positive" stackId="a" fill={MINT} radius={[0, 0, 0, 0]} />
              <Bar dataKey="neutral" stackId="a" fill="#9096A0" radius={[0, 0, 0, 0]} />
              <Bar dataKey="negative" stackId="a" fill={RED} radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Sentiment Distribution">
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={sentimentDistribution} dataKey="value" nameKey="name"
                cx="50%" cy="50%" innerRadius={50} outerRadius={78} paddingAngle={3}
              >
                <Cell fill={MINT} />
                <Cell fill="#9096A0" />
                <Cell fill={RED} />
              </Pie>
              <Legend verticalAlign="middle" align="right" layout="vertical" iconType="circle" wrapperStyle={{ fontSize: 12.5, color: '#9096A0' }} />
              <Tooltip contentStyle={tooltipStyle} />
            </PieChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* News Sentiment table */}
      <div style={{ marginBottom: 8 }}>
        <Panel title="News Sentiment">
          <Table
            size="small"
            pagination={{ pageSize: 6 }}
            dataSource={filteredNews.map((r) => ({ ...r, key: r.id }))}
            locale={{ emptyText: 'No news matches this coin filter.' }}
            columns={[
              { title: 'Headline', dataIndex: 'headline', key: 'headline', render: (t) => <span style={{ color: '#F5F6F7' }}>{t}</span> },
              { title: 'Source', dataIndex: 'source', key: 'source', render: (t) => <span style={{ color: '#9096A0' }}>{t}</span> },
              {
                title: 'Sentiment', dataIndex: 'sentiment', key: 'sentiment',
                render: (v) => (
                  <Tag style={{ background: 'rgba(255,255,255,0.06)', color: sentimentTagColor[v], border: 'none', borderRadius: 8, fontWeight: 600 }}>
                    {v}
                  </Tag>
                ),
              },
              {
                title: 'Score', dataIndex: 'score', key: 'score',
                render: (v) => <span style={{ color: v >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}</span>,
              },
              { title: 'Time', dataIndex: 'time', key: 'time', render: (t) => <span style={{ color: '#6B7280', fontSize: 13 }}>{t}</span> },
            ]}
          />
        </Panel>
      </div>
    </div>
  );
}

function SentimentBar({ label, pct, color }) {
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 4 }}>
        <span style={{ color: '#9096A0', fontWeight: 600 }}>{label}</span>
        <span style={{ color: '#F5F6F7', fontWeight: 700, fontFamily: 'ui-monospace, monospace' }}>{pct}%</span>
      </div>
      <div style={{ width: '100%', height: 8, borderRadius: 999, background: 'rgba(255,255,255,0.06)', overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 999 }} />
      </div>
    </div>
  );
}