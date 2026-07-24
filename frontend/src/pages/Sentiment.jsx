import { useState, useEffect, useCallback } from 'react';
import { Tag, Select, Table, Spin, Alert, Empty, Tooltip as AntTooltip } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import {
  AreaChart, Area, LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';
import { api } from '../lib/api';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const panel = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

const tooltipStyle = { background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 };
const axisStyle = { fill: '#6B7280', fontSize: 11 };

function Panel({ title, children, style, action, hint }) {
  return (
    <div style={{ ...panel, padding: 22, ...style }}>
      {(title || action) && (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
          {title && (
            <h3 style={{ fontSize: 15.5, fontWeight: 700, color: '#F5F6F7', margin: 0, display: 'flex', alignItems: 'center', gap: 6 }}>
              {title}
              {hint && (
                <AntTooltip title={hint}>
                  <InfoCircleOutlined style={{ fontSize: 12.5, color: '#6B7280', cursor: 'help' }} />
                </AntTooltip>
              )}
            </h3>
          )}
          {action}
        </div>
      )}
      {children}
    </div>
  );
}

// ---- Fear & Greed gauge: semicircular dial with needle ----
function fearGreedColor(score) {
  if (score == null) return '#6B7280';
  if (score < 25) return RED;
  if (score < 45) return AMBER;
  if (score < 55) return '#E8C547';
  if (score < 75) return MINT;
  return '#2FBF80';
}

function FearGreedGauge({ score, label }) {
  const cx = 150, cy = 150, r = 110;
  const pct = score ?? 50;
  const angle = 180 - (pct / 100) * 180;
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
          <path key={i} d={arcPath(seg.from, seg.to, r)} fill="none" stroke={seg.color} strokeWidth={18} strokeLinecap="butt" opacity={0.85} />
        ))}
        <line x1={cx} y1={cy} x2={needleX} y2={needleY} stroke="#F5F6F7" strokeWidth={3} strokeLinecap="round" />
        <circle cx={cx} cy={cy} r={7} fill="#F5F6F7" />
        <circle cx={cx} cy={cy} r={3} fill="#0B0E11" />
      </svg>
      <div style={{ marginTop: -8, textAlign: 'center' }}>
        <div style={{ fontSize: 40, fontWeight: 800, color: fearGreedColor(score), fontFamily: 'ui-monospace, monospace', lineHeight: 1 }}>
          {score ?? '—'}
        </div>
        <div style={{ fontSize: 15, fontWeight: 700, color: fearGreedColor(score), marginTop: 6 }}>
          {label ?? 'Not enough data'}
        </div>
      </div>
    </div>
  );
}

function StatBox({ label, value }) {
  return (
    <div style={{ ...panel, padding: 16 }}>
      <div style={{ color: '#9096A0', fontSize: 12, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 19, fontWeight: 700, color: '#F5F6F7', fontFamily: 'ui-monospace, monospace' }}>
        {value ?? '—'}
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

const sentimentTagColor = { Bullish: MINT, Bearish: RED, Neutral: '#9096A0' };

function fmtDate(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  return `${d.getMonth() + 1}/${d.getDate()}`;
}

function timeAgo(iso) {
  if (!iso) return '—';
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

export default function Sentiment() {
  const [coins, setCoins] = useState([]);
  const [coinsLoading, setCoinsLoading] = useState(true);
  const [coinsError, setCoinsError] = useState(null);
  const [selectedCoin, setSelectedCoin] = useState(null);

  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Discover which coins have real sentiment data -- not hardcoded, so
  // new coins show up automatically once sentiment_pipeline has run for them.
  const loadCoins = useCallback(() => {
    setCoinsLoading(true);
    setCoinsError(null);
    api.get('/api/sentiment/coins')
      .then((res) => {
        const list = res.data.coins || [];
        setCoins(list);
        setSelectedCoin((prev) => prev || list[0] || null);
      })
      .catch((err) => setCoinsError(err.message))
      .finally(() => setCoinsLoading(false));
  }, []);

  useEffect(() => { loadCoins(); }, [loadCoins]);

  const loadOverview = useCallback(() => {
    if (!selectedCoin) return;
    setLoading(true);
    setError(null);
    api.get(`/api/sentiment/${selectedCoin}`)
      .then((res) => setOverview(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [selectedCoin]);

  useEffect(() => { loadOverview(); }, [loadOverview]);

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16, marginBottom: 24 }}>
        <div>
          <h2 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Sentiment</h2>
          <p style={{ color: '#9096A0', fontSize: 14, marginTop: 4 }}>
            Reddit-derived market mood and community sentiment for tracked coins.
          </p>
        </div>
        <Select
          value={selectedCoin}
          onChange={setSelectedCoin}
          style={{ width: 160 }}
          loading={coinsLoading}
          disabled={coinsLoading || coins.length === 0}
          placeholder="No coins tracked yet"
          options={coins.map((c) => ({ value: c, label: c }))}
        />
      </div>

      {coinsError && (
        <Alert
          type="error" message="Couldn't load tracked coins" description={coinsError}
          action={<button onClick={loadCoins} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.15)', color: '#F5F6F7', borderRadius: 8, padding: '4px 12px', cursor: 'pointer' }}>Retry</button>}
          style={{ marginBottom: 20 }} showIcon
        />
      )}

      {!coinsLoading && !coinsError && coins.length === 0 && (
        <div style={{ ...panel, padding: 40, textAlign: 'center', marginBottom: 20 }}>
          <Empty
            description={
              <span style={{ color: '#9096A0' }}>
                No sentiment data yet. Run <code>sentiment_pipeline/main.py</code> to populate one or more coins.
              </span>
            }
          />
        </div>
      )}

      {error && selectedCoin && (
        <Alert
          type="error" message={`Couldn't load sentiment for ${selectedCoin}`} description={error}
          action={<button onClick={loadOverview} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.15)', color: '#F5F6F7', borderRadius: 8, padding: '4px 12px', cursor: 'pointer' }}>Retry</button>}
          style={{ marginBottom: 20 }} showIcon
        />
      )}

      {selectedCoin && loading && (
        <div style={{ display: 'flex', justifyContent: 'center', padding: '80px 0' }}>
          <Spin size="large" />
        </div>
      )}

      {selectedCoin && !loading && overview && (
        <>
          {/* Fear & Greed Index + Overall Market Sentiment */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <Panel
              title="Fear & Greed Index"
              hint="Derived from this coin's real bullish/neutral/bearish Reddit post distribution — not the external Fear & Greed Index (alternative.me), which this app doesn't fetch."
            >
              <div style={{ display: 'flex', justifyContent: 'center' }}>
                <FearGreedGauge score={overview.fear_greed.score} label={overview.fear_greed.label} />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10, marginTop: 12 }}>
                <StatBox label="Yesterday" value={overview.fear_greed.yesterday} />
                <StatBox label="Last Week" value={overview.fear_greed.last_week} />
                <StatBox label="Last Month" value={overview.fear_greed.last_month} />
              </div>
            </Panel>

            <Panel title="Overall Market Sentiment" hint={`Based on ${overview.overall.post_count} Reddit posts scored by CryptoBERT.`}>
              {overview.overall.post_count === 0 ? (
                <Empty description={<span style={{ color: '#9096A0' }}>No posts yet for {selectedCoin}.</span>} style={{ padding: '20px 0' }} />
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: 24 }}>
                  <div style={{ position: 'relative', width: 160, height: 160, flexShrink: 0 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={[
                            { name: 'Bullish', value: overview.overall.bullish_pct },
                            { name: 'Neutral', value: overview.overall.neutral_pct },
                            { name: 'Bearish', value: overview.overall.bearish_pct },
                          ]}
                          dataKey="value" nameKey="name"
                          cx="50%" cy="50%" innerRadius={55} outerRadius={75} paddingAngle={3} startAngle={90} endAngle={-270}
                        >
                          <Cell fill={MINT} /><Cell fill="#9096A0" /><Cell fill={RED} />
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                    <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', textAlign: 'center' }}>
                      <div style={{ fontSize: 24, fontWeight: 800, color: (overview.overall.score ?? 0) >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace' }}>
                        {overview.overall.score != null ? `${overview.overall.score >= 0 ? '+' : ''}${overview.overall.score.toFixed(2)}` : '—'}
                      </div>
                      <div style={{ fontSize: 12, color: '#9096A0', fontWeight: 600 }}>{overview.overall.label ?? '—'}</div>
                    </div>
                  </div>
                  <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 10 }}>
                    <SentimentBar label="Bullish" pct={overview.overall.bullish_pct} color={MINT} />
                    <SentimentBar label="Neutral" pct={overview.overall.neutral_pct} color="#9096A0" />
                    <SentimentBar label="Bearish" pct={overview.overall.bearish_pct} color={RED} />
                  </div>
                </div>
              )}
            </Panel>
          </div>

          {/* Fear & Greed Timeline + Sentiment Timeline */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <Panel title="Fear & Greed Timeline">
              {overview.fear_greed_timeline.length === 0 ? (
                <Empty description={<span style={{ color: '#9096A0' }}>Not enough history yet.</span>} style={{ padding: '30px 0' }} />
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={overview.fear_greed_timeline.map((p) => ({ ...p, label: fmtDate(p.date) }))} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <defs>
                      <linearGradient id="fgGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor={AMBER} stopOpacity={0.35} />
                        <stop offset="95%" stopColor={AMBER} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={[0, 100]} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Area type="monotone" dataKey="score" stroke={AMBER} strokeWidth={2.5} fill="url(#fgGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </Panel>
            <Panel title="Sentiment Timeline">
              {overview.sentiment_timeline.length === 0 ? (
                <Empty description={<span style={{ color: '#9096A0' }}>Not enough history yet.</span>} style={{ padding: '30px 0' }} />
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={overview.sentiment_timeline.map((p) => ({ ...p, label: fmtDate(p.date) }))} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={axisStyle} axisLine={false} tickLine={false} domain={[-1, 1]} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line type="monotone" dataKey="score" stroke={MINT} strokeWidth={2.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </Panel>
          </div>

          {/* Post Volume + Sentiment Distribution */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
            <Panel title="Post Volume" hint="Reddit posts per day, split by sentiment label.">
              {overview.post_volume.length === 0 ? (
                <Empty description={<span style={{ color: '#9096A0' }}>Not enough history yet.</span>} style={{ padding: '30px 0' }} />
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={overview.post_volume.map((p) => ({ ...p, label: fmtDate(p.day) }))} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="label" tick={axisStyle} axisLine={false} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Bar dataKey="bullish" stackId="a" fill={MINT} radius={[0, 0, 0, 0]} />
                    <Bar dataKey="neutral" stackId="a" fill="#9096A0" radius={[0, 0, 0, 0]} />
                    <Bar dataKey="bearish" stackId="a" fill={RED} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </Panel>
            <Panel title="Sentiment Distribution">
              {overview.overall.post_count === 0 ? (
                <Empty description={<span style={{ color: '#9096A0' }}>No posts yet.</span>} style={{ padding: '30px 0' }} />
              ) : (
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Bullish', value: overview.overall.bullish_pct },
                        { name: 'Neutral', value: overview.overall.neutral_pct },
                        { name: 'Bearish', value: overview.overall.bearish_pct },
                      ]}
                      dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={50} outerRadius={78} paddingAngle={3}
                    >
                      <Cell fill={MINT} /><Cell fill="#9096A0" /><Cell fill={RED} />
                    </Pie>
                    <Legend verticalAlign="middle" align="right" layout="vertical" iconType="circle" wrapperStyle={{ fontSize: 12.5, color: '#9096A0' }} />
                    <Tooltip contentStyle={tooltipStyle} />
                  </PieChart>
                </ResponsiveContainer>
              )}
            </Panel>
          </div>

          {/* Top Reddit posts -- stands in for the PDF's "News Sentiment" (no news source exists in this codebase) */}
          <div style={{ marginBottom: 8 }}>
            <Panel title="Top Reddit Posts" hint='Standing in for "News Sentiment" — this system tracks Reddit, not a news wire. Sorted by Reddit score (upvotes), not sentiment.'>
              <Table
                size="small"
                pagination={{ pageSize: 6 }}
                dataSource={overview.top_posts.map((r) => ({ ...r, key: r.post_id }))}
                locale={{ emptyText: 'No posts yet for this coin.' }}
                columns={[
                  { title: 'Title', dataIndex: 'title', key: 'title', render: (t) => <span style={{ color: '#F5F6F7' }}>{t || '(no title)'}</span> },
                  { title: 'Subreddit', dataIndex: 'subreddit', key: 'subreddit', render: (t) => <span style={{ color: '#9096A0' }}>r/{t}</span> },
                  {
                    title: 'Sentiment', dataIndex: 'sentiment_label', key: 'sentiment_label',
                    render: (v) => v ? (
                      <Tag style={{ background: 'rgba(255,255,255,0.06)', color: sentimentTagColor[v] || '#9096A0', border: 'none', borderRadius: 8, fontWeight: 600 }}>
                        {v}
                      </Tag>
                    ) : '—',
                  },
                  {
                    title: 'Score', dataIndex: 'sentiment_score', key: 'sentiment_score',
                    render: (v) => v == null ? '—' : <span style={{ color: v >= 0 ? MINT : RED, fontFamily: 'ui-monospace, monospace', fontWeight: 600 }}>{v >= 0 ? '+' : ''}{v.toFixed(2)}</span>,
                  },
                  { title: 'Upvotes', dataIndex: 'score', key: 'upvotes', align: 'right', render: (v) => <span style={{ color: '#9096A0', fontFamily: 'ui-monospace, monospace' }}>{v ?? '—'}</span> },
                  { title: 'Posted', dataIndex: 'created_utc', key: 'created_utc', render: (t) => <span style={{ color: '#6B7280', fontSize: 13 }}>{timeAgo(t)}</span> },
                ]}
              />
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}