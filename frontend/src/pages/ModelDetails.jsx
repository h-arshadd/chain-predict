import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Spin, Alert } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  BarChart, Bar, PieChart, Pie, Cell,
  ResponsiveContainer, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from 'recharts';
import { api } from '../lib/api';
import { METRIC_CARDS_COL_1, METRIC_CARDS_COL_2, fmtMetric } from '../lib/metricCards';
import Panel, { panelFlat as panel } from '../components/Panel';
import EmptyChart from '../components/EmptyChart';
import KeyValue from '../components/KeyValue';
import StatBox from '../components/StatBox';
import { tooltipStyle, tooltipLabelStyle, tooltipItemStyle, axisStyle } from '../lib/chartStyle';

const MINT = '#3DDC97';
const RED = '#F0466B';
const BLUE = '#4D9DE0';

const SIGNAL_COLORS = { Buy: MINT, Sell: RED, Hold: '#6B7280' };

function PillList({ items, color, mono }) {
  if (!items || items.length === 0) {
    return <span style={{ color: '#6B7280', fontSize: 13 }}>None recorded.</span>;
  }
  return (
    <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
      {items.map((it, i) => (
        <span
          key={i}
          style={{
            background: color ? `${color}14` : 'rgba(255,255,255,0.05)',
            border: `1px solid ${color ? `${color}33` : 'rgba(255,255,255,0.08)'}`,
            color: color || '#F5F6F7', fontSize: 12.5, fontWeight: 600, padding: '6px 12px', borderRadius: 999,
            fontFamily: mono ? 'ui-monospace, monospace' : undefined,
          }}
        >
          {it}
        </span>
      ))}
    </div>
  );
}

const fmtMetricName = (k) => k.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
const fmtNum = (v, digits = 4) => (typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(digits)) : (v ?? '—'));
const fmtPct = (v) => (v == null ? '—' : `${(v * 100).toFixed(2)}%`);

function algorithmLabel(algo) {
  if (!algo) return '—';
  const upper = new Set(['mlp', 'gru', 'knn', 'svm', 'svr']);
  return algo.split('_').map((w) => (upper.has(w.toLowerCase()) ? w.toUpperCase() : w.toLowerCase() === 'lstm' ? 'LSTM' : w[0].toUpperCase() + w.slice(1))).join(' ');
}

export default function ModelDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.get(`/api/ml-models/${id}`)
      .then((res) => setData(res.data))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '100px 0' }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ paddingTop: 8 }}>
        <Alert
          type="error"
          message={<span style={{ color: '#F5F6F7', fontWeight: 600 }}>Couldn't load this model run</span>}
          description={<span style={{ color: '#C9CDD3' }}>{error}</span>}
          action={<button onClick={load} style={backBtnStyle}>Retry</button>}
          showIcon
          style={{ background: 'rgba(240, 70, 107, 0.08)', border: '1px solid rgba(240, 70, 107, 0.3)' }}
        />
      </div>
    );
  }

  if (!data) return null;

  const dataPrep = data.data_prep || {};
  const split = data.split || {};
  const preprocessing = data.preprocessing || {};
  const model = data.model || {};
  const evaluation = data.evaluation || {};

  const mlMetrics = evaluation.ml_metrics || {};
  const tradingMetrics = evaluation.trading_metrics_summary || {};
  // Full quantstats dict -- only present for runs trained after this was
  // added (see build_evaluation_metadata() in metadata.py). Older runs
  // just won't show the extra cards below; the existing summary-based
  // Sharpe/Total Return/Max Drawdown cards further down still work off
  // trading_metrics either way.
  const tradingMetricsFull = evaluation.trading_metrics_full || null;
  const metricCardsCol1 = tradingMetricsFull
    ? METRIC_CARDS_COL_1.filter((m) => tradingMetricsFull[m.key] !== undefined)
    : [];
  const metricCardsCol2 = tradingMetricsFull
    ? METRIC_CARDS_COL_2.filter((m) => tradingMetricsFull[m.key] !== undefined)
    : [];
  const tradeSummary = evaluation.trade_summary || {};
  const signalCounts = evaluation.signal_counts || {};
  const winLoss = tradeSummary.win_loss || {};

  const featureColumns = preprocessing.feature_columns || [];
  const steps = preprocessing.steps || [];
  const hyperparameters = model.hyperparameters || model.configured_overrides || {};
  const architecture = model.architecture || null;
  const training = model.training || null;
  const classes = model.classes || null;

  // Chart data -- derived from the same dicts the key/value panels below
  // already read (ml_metrics / trading_metrics_summary / signal_counts /
  // trade_summary.win_loss). No new API fields needed; this is purely a
  // second, visual rendering of data that was previously only shown as
  // rows of numbers. Values are filtered to numbers only (some metrics
  // can be null when not computed for a given run) so a chart never
  // tries to plot a missing metric as a zero-height bar.
  const mlMetricsChartData = Object.entries(mlMetrics)
    .filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => ({ name: fmtMetricName(k), value: v }));

  // Trading metrics mixes very different scales in one dict (sharpe ~
  // -3..3, comp/max_drawdown as fractions, num_trades in the hundreds)
  // -- charting them all on one axis would flatten the small ones to
  // nothing, so this is split into "ratio-like" metrics (sharpe, sortino,
  // profit_factor, etc. -- roughly -5..10 range) and "percentage" metrics
  // (comp, max_drawdown, win_rate -- fractions of 1) as two separate bar
  // charts instead of guessing a shared scale.
  const _PERCENT_KEYS = new Set(['comp', 'max_drawdown', 'win_rate', 'cagr', 'volatility']);
  const tradingRatioChartData = Object.entries(tradingMetrics)
    .filter(([k, v]) => typeof v === 'number' && !_PERCENT_KEYS.has(k))
    .map(([k, v]) => ({ name: fmtMetricName(k), value: v }));
  const tradingPercentChartData = Object.entries(tradingMetrics)
    .filter(([k, v]) => typeof v === 'number' && _PERCENT_KEYS.has(k))
    .map(([k, v]) => ({ name: fmtMetricName(k), value: v * 100 }));

  const signalChartData = Object.entries(signalCounts)
    .filter(([, v]) => typeof v === 'number')
    .map(([k, v]) => ({ name: k, value: v }));

  const winLossChartData = (winLoss.wins != null || winLoss.losses != null)
    ? [
        { name: 'Wins', value: winLoss.wins || 0 },
        { name: 'Losses', value: winLoss.losses || 0 },
      ]
    : [];

  return (
    <div style={{ paddingTop: 8 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 24 }}>
        <button onClick={() => navigate(-1)} style={backBtnStyle}>
          <ArrowLeftOutlined />
        </button>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
            <h2 style={{ fontSize: 20, fontWeight: 700, color: '#F5F6F7', margin: 0, fontFamily: 'ui-monospace, monospace' }}>
              {data.symbol ? data.symbol.toUpperCase() : '—'}
              {' · '}
              {data.model_type ? data.model_type[0].toUpperCase() + data.model_type.slice(1) : '—'}
              {' · '}
              {algorithmLabel(data.algorithm)}
            </h2>
          </div>
          <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2 }}>
            {data.timeframe || '—'}
            {data.horizon != null && <> · horizon {data.horizon}</>}
            {' · '}Trained {data.trained_at ? new Date(data.trained_at).toLocaleString() : 'unknown'}
          </div>
        </div>
      </div>

      {/* Evaluation summary strip -- ml_metrics keys differ by model_type
          (mae/rmse for regression, accuracy/f1 for classification), so
          these are rendered generically off whatever keys are actually
          present rather than hardcoding one metric set. */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        {Object.entries(mlMetrics).map(([k, v]) => (
          <StatBox key={k} label={fmtMetricName(k)} value={fmtNum(v)} />
        ))}
        <StatBox
          label="Sharpe"
          value={fmtNum(tradingMetrics.sharpe, 2)}
          positive={tradingMetrics.sharpe == null ? undefined : tradingMetrics.sharpe >= 0}
        />
        <StatBox
          label="Total Return"
          value={fmtPct(tradingMetrics.comp)}
          positive={tradingMetrics.comp == null ? undefined : tradingMetrics.comp >= 0}
        />
        <StatBox
          label="Max Drawdown"
          value={fmtPct(tradingMetrics.max_drawdown)}
          positive={false}
        />
      </div>

      {/* Dataset Information + Training Information */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Dataset Information">
          <KeyValue label="Dataset" value={dataPrep.dataset_name} mono />
          <KeyValue label="Date Range" value={dataPrep.data ? `${dataPrep.data.start_date} → ${dataPrep.data.end_date}` : null} />
          <KeyValue label="Total Rows" value={dataPrep.total_rows} mono />
          <KeyValue label="Target Horizon" value={dataPrep.target?.horizon} mono />
          <KeyValue label="Noise Filter" value={dataPrep.target?.filter_noise != null ? (dataPrep.target.filter_noise ? `Yes (threshold ${dataPrep.target.noise_threshold})` : 'No') : null} />
          <KeyValue
            label="Train / Val / Test"
            value={
              split.train || split.test
                ? `${split.train?.rows ?? '—'} / ${split.validation?.rows ?? '—'} / ${split.test?.rows ?? '—'} rows`
                : null
            }
            mono
          />
          <div style={{ marginTop: 14 }}>
            <div style={{ color: '#9096A0', fontSize: 13, marginBottom: 8 }}>Feature Columns ({featureColumns.length})</div>
            <PillList items={featureColumns} mono />
          </div>
        </Panel>

        <Panel title="Preprocessing & Model Configuration">
          <KeyValue label="Algorithm" value={algorithmLabel(data.algorithm)} />
          <KeyValue label="Serialization Format" value={model.serialization_format} mono />
          <KeyValue label="Random Seed" value={model.random_seed} mono />
          {classes && <KeyValue label="Classes" value={classes.join(', ')} mono />}
          <div style={{ marginTop: 14 }}>
            <div style={{ color: '#9096A0', fontSize: 13, marginBottom: 8 }}>Preprocessing Steps (fit order)</div>
            <PillList items={steps.map((s) => s.method)} color={MINT} />
          </div>
        </Panel>
      </div>

      {/* Hyperparameters / Architecture */}
      <div style={{ display: 'grid', gridTemplateColumns: architecture ? '1fr 1fr' : '1fr', gap: 20, marginBottom: 20 }}>
        <Panel title={architecture ? 'Training Hyperparameters' : 'Hyperparameters'}>
          {Object.keys(hyperparameters).length > 0 ? (
            <PillList
              items={Object.entries(hyperparameters)
                .filter(([, v]) => v !== null)
                .map(([k, v]) => `${k}: ${v}`)}
              color={MINT}
              mono
            />
          ) : (
            <span style={{ color: '#6B7280', fontSize: 13 }}>No hyperparameters recorded.</span>
          )}
        </Panel>

        {architecture && (
          <Panel title="Network Architecture">
            <KeyValue label="Hidden Layers" value={architecture.hidden_layers} mono />
            <KeyValue label="Hidden Units" value={architecture.hidden_units} mono />
            <KeyValue label="Activation" value={architecture.activation} mono />
            <KeyValue label="Dropout" value={architecture.dropout} mono />
            <KeyValue label="Batch Norm" value={architecture.batch_norm != null ? (architecture.batch_norm ? 'Yes' : 'No') : null} />
            {training && (
              <>
                <KeyValue label="Optimizer" value={training.optimizer} mono />
                <KeyValue label="Learning Rate" value={training.learning_rate} mono />
                <KeyValue label="Batch Size" value={training.batch_size} mono />
                <KeyValue label="Epochs" value={training.epochs} mono />
                <KeyValue label="Early Stopping Patience" value={training.early_stopping_patience} mono />
                <KeyValue label="Loss" value={training.loss} mono />
              </>
            )}
          </Panel>
        )}
      </div>

      {/* Extended performance metrics -- pulled from
          evaluation.trading_metrics_full, the full quantstats dict
          evaluate_model() already computes but older runs never
          persisted (see build_evaluation_metadata() in metadata.py).
          Split into two equal-length cards instead of one big wall of
          tiles. Falls back to a plain note for runs trained before that
          field existed, rather than silently showing nothing. */}
      {metricCardsCol1.length > 0 || metricCardsCol2.length > 0 ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
          {metricCardsCol1.length > 0 && (
            <Panel title="Performance Metrics">
              {metricCardsCol1.map((m) => {
                const value = tradingMetricsFull[m.key];
                const positive = value == null ? undefined : m.invert ? value <= 0 : value >= 0;
                return (
                  <KeyValue
                    key={m.key}
                    label={m.label}
                    value={fmtMetric(value, m.kind)}
                    mono
                    color={positive === undefined ? undefined : positive ? MINT : RED}
                  />
                );
              })}
            </Panel>
          )}
          {metricCardsCol2.length > 0 && (
            <Panel title="Performance Metrics (cont.)">
              {metricCardsCol2.map((m) => {
                const value = tradingMetricsFull[m.key];
                const positive = value == null ? undefined : m.invert ? value <= 0 : value >= 0;
                return (
                  <KeyValue
                    key={m.key}
                    label={m.label}
                    value={fmtMetric(value, m.kind)}
                    mono
                    color={positive === undefined ? undefined : positive ? MINT : RED}
                  />
                );
              })}
            </Panel>
          )}
        </div>
      ) : (
        <div style={{ ...panel, padding: 16, marginBottom: 20, color: '#6B7280', fontSize: 13 }}>
          Extended performance metrics aren't available for this run -- it was trained before this was tracked. Retrain to see the full metric breakdown here.
        </div>
      )}

      {/* ML Metrics + Trading Metrics -- charted. Exact figures are still
          the tables below (rounding in a bar's tooltip is fine, rounding
          in the source-of-truth table is not), so these are a visual
          companion to that data, not a replacement for it. */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="ML Metrics">
          {mlMetricsChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={mlMetricsChartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => fmtNum(v)} />
                <Bar dataKey="value" radius={[6, 6, 6, 6]} fill={MINT} />
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart centered text="No ML metrics recorded for this run." />
          )}
        </Panel>
        <Panel title="Trading Metrics (signal-converted backtest)">
          {tradingRatioChartData.length > 0 || tradingPercentChartData.length > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              {tradingRatioChartData.length > 0 && (
                <ResponsiveContainer width="100%" height={110}>
                  <BarChart data={tradingRatioChartData} layout="vertical" margin={{ top: 0, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis type="number" tick={axisStyle} axisLine={false} tickLine={false} />
                    <YAxis type="category" dataKey="name" tick={axisStyle} axisLine={false} tickLine={false} width={100} />
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => fmtNum(v, 2)} />
                    <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                      {tradingRatioChartData.map((entry, i) => (
                        <Cell key={i} fill={entry.value >= 0 ? MINT : RED} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
              {tradingPercentChartData.length > 0 && (
                <ResponsiveContainer width="100%" height={110}>
                  <BarChart data={tradingPercentChartData} layout="vertical" margin={{ top: 0, right: 12, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                    <XAxis type="number" tick={axisStyle} axisLine={false} tickLine={false} tickFormatter={(v) => `${v.toFixed(0)}%`} />
                    <YAxis type="category" dataKey="name" tick={axisStyle} axisLine={false} tickLine={false} width={100} />
                    <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} formatter={(v) => `${v.toFixed(2)}%`} />
                    <Bar dataKey="value" radius={[0, 6, 6, 0]}>
                      {tradingPercentChartData.map((entry, i) => (
                        <Cell key={i} fill={entry.value >= 0 ? MINT : RED} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          ) : (
            <EmptyChart centered text="No trading metrics recorded for this run." />
          )}
        </Panel>
      </div>

      {/* Signal Distribution + Win/Loss -- both are small categorical
          counts (2-3 buckets), so a bar chart and a donut are enough;
          no need for anything more elaborate than that. */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Signal Distribution">
          {signalChartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={signalChartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                <XAxis dataKey="name" tick={axisStyle} axisLine={false} tickLine={false} />
                <YAxis tick={axisStyle} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
                <Bar dataKey="value" radius={[6, 6, 6, 6]}>
                  {signalChartData.map((entry, i) => (
                    <Cell key={i} fill={SIGNAL_COLORS[entry.name] || BLUE} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart centered text="No signal counts recorded for this run." />
          )}
        </Panel>
        <Panel title="Win / Loss">
          {winLossChartData.length > 0 && (winLossChartData[0].value + winLossChartData[1].value) > 0 ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={winLossChartData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={50}
                  outerRadius={78}
                  paddingAngle={2}
                >
                  {winLossChartData.map((entry, i) => (
                    <Cell key={i} fill={entry.name === 'Wins' ? MINT : RED} />
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle} labelStyle={tooltipLabelStyle} itemStyle={tooltipItemStyle} />
                <Legend wrapperStyle={{ fontSize: 12, color: '#9096A0' }} />
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <EmptyChart centered text="No trade win/loss data recorded for this run." />
          )}
        </Panel>
      </div>

      {/* ML Metrics + Trading Metrics (exact values) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="ML Metrics (exact values)">
          {Object.keys(mlMetrics).length > 0 ? (
            Object.entries(mlMetrics).map(([k, v]) => (
              <KeyValue key={k} label={fmtMetricName(k)} value={fmtNum(v)} mono />
            ))
          ) : (
            <span style={{ color: '#6B7280', fontSize: 13 }}>No ML metrics recorded for this run.</span>
          )}
        </Panel>
        <Panel title="Trading Metrics (exact values)">
          {Object.keys(tradingMetrics).length > 0 ? (
            Object.entries(tradingMetrics).map(([k, v]) => (
              <KeyValue
                key={k}
                label={fmtMetricName(k)}
                value={k.includes('drawdown') || k === 'comp' || k === 'win_rate' ? fmtPct(v) : fmtNum(v, 2)}
                mono
              />
            ))
          ) : (
            <span style={{ color: '#6B7280', fontSize: 13 }}>No trading metrics recorded for this run.</span>
          )}
        </Panel>
      </div>

      {/* Trade Summary + Signal Counts */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 8 }}>
        <Panel title="Trade Summary">
          {Object.keys(tradeSummary).length > 0 ? (
            <>
              <KeyValue label="Final Balance" value={fmtNum(tradeSummary.final_balance, 2)} mono />
              <KeyValue label="Total Net Profit" value={fmtNum(tradeSummary.total_net_profit, 2)} mono />
              <KeyValue label="Total Trades" value={tradeSummary.total_trades} mono />
              <KeyValue label="Wins" value={winLoss.wins} mono />
              <KeyValue label="Losses" value={winLoss.losses} mono />
              <KeyValue label="Win Rate" value={fmtPct(winLoss.win_rate)} mono />
            </>
          ) : (
            <span style={{ color: '#6B7280', fontSize: 13 }}>No trade summary recorded for this run.</span>
          )}
        </Panel>
        <Panel title="Signal Counts">
          {Object.keys(signalCounts).length > 0 ? (
            Object.entries(signalCounts).map(([k, v]) => (
              <KeyValue key={k} label={k} value={v} mono />
            ))
          ) : (
            <span style={{ color: '#6B7280', fontSize: 13 }}>No signal counts recorded for this run.</span>
          )}
        </Panel>
      </div>
    </div>
  );
}

const backBtnStyle = {
  width: 36, height: 36, borderRadius: 10, border: '1px solid rgba(255,255,255,0.1)',
  background: 'rgba(255,255,255,0.04)', color: '#F5F6F7', cursor: 'pointer',
  display: 'flex', alignItems: 'center', justifyContent: 'center',
};