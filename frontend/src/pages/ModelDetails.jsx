import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Spin, Alert } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { api } from '../lib/api';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

const KIND_LABELS = {
  regressor: 'Traditional Regressor',
  classifier: 'Traditional Classifier',
  deep_learning_regressor: 'Deep Learning Regressor',
  deep_learning_classifier: 'Deep Learning Classifier',
};

const KIND_COLORS = {
  regressor: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  classifier: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  deep_learning_regressor: { bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  deep_learning_classifier: { bg: 'rgba(255,138,92,0.14)', fg: AMBER },
};

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
    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, padding: '8px 0', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
      <span style={{ color: '#9096A0', fontSize: 13, flexShrink: 0 }}>{label}</span>
      <span style={{ color: '#F5F6F7', fontSize: 13, fontWeight: 600, fontFamily: mono ? 'ui-monospace, monospace' : undefined, textAlign: 'right' }}>
        {value ?? '—'}
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
          message="Couldn't load this model run"
          description={error}
          action={<button onClick={load} style={backBtnStyle}>Retry</button>}
          showIcon
        />
      </div>
    );
  }

  if (!data) return null;

  const kindStyle = KIND_COLORS[data.model_kind] || KIND_COLORS.regressor;
  const dataPrep = data.data_prep || {};
  const split = data.split || {};
  const preprocessing = data.preprocessing || {};
  const model = data.model || {};
  const evaluation = data.evaluation || {};

  const mlMetrics = evaluation.ml_metrics || {};
  const tradingMetrics = evaluation.trading_metrics_summary || {};
  const tradeSummary = evaluation.trade_summary || {};
  const signalCounts = evaluation.signal_counts || {};
  const winLoss = tradeSummary.win_loss || {};

  const featureColumns = preprocessing.feature_columns || [];
  const steps = preprocessing.steps || [];
  const hyperparameters = model.hyperparameters || model.configured_overrides || {};
  const architecture = model.architecture || null;
  const training = model.training || null;
  const classes = model.classes || null;

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
              {data.run_id}
            </h2>
            <Tag style={{ background: kindStyle.bg, color: kindStyle.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
              {KIND_LABELS[data.model_kind] || data.model_kind}
            </Tag>
          </div>
          <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2 }}>
            {algorithmLabel(data.algorithm)} · {data.symbol ? data.symbol.toUpperCase() : '—'} · {data.exchange || '—'} · {data.timeframe || '—'}
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

      {/* ML Metrics + Trading Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="ML Metrics">
          {Object.keys(mlMetrics).length > 0 ? (
            Object.entries(mlMetrics).map(([k, v]) => (
              <KeyValue key={k} label={fmtMetricName(k)} value={fmtNum(v)} mono />
            ))
          ) : (
            <span style={{ color: '#6B7280', fontSize: 13 }}>No ML metrics recorded for this run.</span>
          )}
        </Panel>
        <Panel title="Trading Metrics (signal-converted backtest)">
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