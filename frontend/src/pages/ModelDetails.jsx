import { useParams, useNavigate } from 'react-router-dom';
import { Tag, Table } from 'antd';
import { ArrowLeftOutlined } from '@ant-design/icons';
import {
  LineChart, Line, BarChart, Bar, ResponsiveContainer,
  XAxis, YAxis, CartesianGrid, Tooltip, Cell,
} from 'recharts';

const MINT = '#3DDC97';
const RED = '#F0466B';
const AMBER = '#FF8A5C';

// ---- placeholder data — replace with GET /api/models/{id} once backend exists ----
const model = {
  name: 'BTC-4h-LSTM-v3',
  type: 'LSTM',
  symbol: 'BTCUSDT',
  timeframe: '4h',
  status: 'Deployed',
  trainingDate: '2026-07-18',
  dataset: {
    dataset: 'Bybit OHLCV + on-chain flow, 2019–2026',
    features: ['Close', 'Volume', 'RSI(14)', 'EMA(20)', 'EMA(50)', 'ATR(14)', 'Funding Rate', 'OI Delta'],
    target: 'Next 4h close direction (up/down)',
    dateRange: '2019-01-01 → 2026-06-30',
    trainTestSplit: '80% train / 20% test (walk-forward)',
  },
  training: {
    algorithm: 'LSTM (2-layer, 128 hidden units)',
    hyperparameters: { learningRate: 0.0008, batchSize: 64, epochs: 120, dropout: 0.2, sequenceLength: 60 },
    preprocessing: 'Missing-value forward fill, outlier clipping at 3σ',
    featureEngineering: 'Log returns, rolling volatility, lagged indicators (t-1, t-2, t-3)',
    scaling: 'Min-max scaling per feature, fit on train split only',
    stationarity: 'ADF test passed after first-order differencing on price series',
  },
  evaluation: {
    mlMetrics: { accuracy: 68.4, precision: 66.1, recall: 71.2, f1: 68.5, auc: 0.74 },
    backtestMetrics: { sharpe: 1.7, sortino: 2.1, maxDrawdown: -11.2, winRate: 62.3, totalReturn: 27.8 },
    predictionSummary: { totalPredictions: 3240, correctPredictions: 2216, avgConfidence: 71.4 },
  },
};

const accuracyHistory = Array.from({ length: 20 }, (_, i) => ({
  epoch: i * 6,
  train: 50 + i * 1.1 + Math.sin(i / 3) * 1.5,
  val: 48 + i * 0.9 + Math.sin(i / 3 + 1) * 2,
}));

const confusionCounts = [
  { name: 'True Positive', value: 1120 },
  { name: 'True Negative', value: 1096 },
  { name: 'False Positive', value: 512 },
  { name: 'False Negative', value: 512 },
];

const predictionHistory = [
  { id: 1, timestamp: '2026-07-22 12:00', predicted: 'Up', actual: 'Up', confidence: 74.2, correct: true },
  { id: 2, timestamp: '2026-07-22 08:00', predicted: 'Down', actual: 'Down', confidence: 68.9, correct: true },
  { id: 3, timestamp: '2026-07-22 04:00', predicted: 'Up', actual: 'Down', confidence: 55.1, correct: false },
  { id: 4, timestamp: '2026-07-22 00:00', predicted: 'Up', actual: 'Up', confidence: 81.6, correct: true },
  { id: 5, timestamp: '2026-07-21 20:00', predicted: 'Down', actual: 'Down', confidence: 70.3, correct: true },
];

const panel = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

const tooltipStyle = { background: '#161B21', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 10, fontSize: 12 };
const axisStyle = { fill: '#6B7280', fontSize: 11 };

const statusColors = {
  Deployed: { bg: 'rgba(61,220,151,0.12)', fg: MINT },
  Training: { bg: 'rgba(255,138,92,0.14)', fg: AMBER },
  Archived: { bg: 'rgba(255,255,255,0.06)', fg: '#9096A0' },
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

export default function ModelDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const statusStyle = statusColors[model.status] || statusColors.Archived;
  const hp = model.training.hyperparameters;

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
            <h2 style={{ fontSize: 22, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>{model.name}</h2>
            <Tag style={{ background: statusStyle.bg, color: statusStyle.fg, border: 'none', borderRadius: 8, fontWeight: 600 }}>
              {model.status}
            </Tag>
          </div>
          <div style={{ color: '#9096A0', fontSize: 13, marginTop: 2 }}>
            {model.type} &middot; {model.symbol} &middot; {model.timeframe} &middot; Trained {model.trainingDate} &middot; Model ID {id}
          </div>
        </div>
      </div>

      {/* Evaluation summary strip */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: 14, marginBottom: 20 }}>
        <StatBox label="Accuracy" value={`${model.evaluation.mlMetrics.accuracy}%`} positive />
        <StatBox label="F1 Score" value={model.evaluation.mlMetrics.f1} />
        <StatBox label="AUC" value={model.evaluation.mlMetrics.auc} />
        <StatBox label="Backtest Sharpe" value={model.evaluation.backtestMetrics.sharpe} />
        <StatBox label="Backtest Return" value={`+${model.evaluation.backtestMetrics.totalReturn}%`} positive />
        <StatBox label="Max Drawdown" value={`${model.evaluation.backtestMetrics.maxDrawdown}%`} positive={false} />
      </div>

      {/* Dataset Information + Training Information */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Dataset Information">
          <KeyValue label="Dataset" value={model.dataset.dataset} />
          <KeyValue label="Target" value={model.dataset.target} />
          <KeyValue label="Date Range" value={model.dataset.dateRange} />
          <KeyValue label="Train/Test Split" value={model.dataset.trainTestSplit} />
          <div style={{ marginTop: 14 }}>
            <div style={{ color: '#9096A0', fontSize: 13, marginBottom: 8 }}>Features</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {model.dataset.features.map((f) => (
                <span
                  key={f}
                  style={{
                    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.08)',
                    color: '#F5F6F7', fontSize: 12.5, fontWeight: 600, padding: '6px 12px', borderRadius: 999,
                  }}
                >
                  {f}
                </span>
              ))}
            </div>
          </div>
        </Panel>

        <Panel title="Training Information">
          <KeyValue label="Algorithm" value={model.training.algorithm} />
          <KeyValue label="Preprocessing" value={model.training.preprocessing} />
          <KeyValue label="Feature Engineering" value={model.training.featureEngineering} />
          <KeyValue label="Scaling" value={model.training.scaling} />
          <KeyValue label="Stationarity" value={model.training.stationarity} />
          <div style={{ marginTop: 14 }}>
            <div style={{ color: '#9096A0', fontSize: 13, marginBottom: 8 }}>Hyperparameters</div>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {Object.entries(hp).map(([k, v]) => (
                <span
                  key={k}
                  style={{
                    background: 'rgba(61,220,151,0.08)', border: '1px solid rgba(61,220,151,0.18)',
                    color: MINT, fontSize: 12.5, fontWeight: 600, padding: '6px 12px', borderRadius: 999,
                    fontFamily: 'ui-monospace, monospace',
                  }}
                >
                  {k}: {v}
                </span>
              ))}
            </div>
          </div>
        </Panel>
      </div>

      {/* ML Metrics + Backtest Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="ML Metrics">
          <KeyValue label="Accuracy" value={`${model.evaluation.mlMetrics.accuracy}%`} mono />
          <KeyValue label="Precision" value={`${model.evaluation.mlMetrics.precision}%`} mono />
          <KeyValue label="Recall" value={`${model.evaluation.mlMetrics.recall}%`} mono />
          <KeyValue label="F1 Score" value={model.evaluation.mlMetrics.f1} mono />
          <KeyValue label="AUC" value={model.evaluation.mlMetrics.auc} mono />
        </Panel>
        <Panel title="Backtest Metrics">
          <KeyValue label="Sharpe Ratio" value={model.evaluation.backtestMetrics.sharpe} mono />
          <KeyValue label="Sortino Ratio" value={model.evaluation.backtestMetrics.sortino} mono />
          <KeyValue label="Max Drawdown" value={`${model.evaluation.backtestMetrics.maxDrawdown}%`} mono />
          <KeyValue label="Win Rate" value={`${model.evaluation.backtestMetrics.winRate}%`} mono />
          <KeyValue label="Total Return" value={`+${model.evaluation.backtestMetrics.totalReturn}%`} mono />
        </Panel>
      </div>

      {/* Training Curve + Confusion Breakdown */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        <Panel title="Training vs Validation Accuracy">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={accuracyHistory} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="epoch" tick={axisStyle} axisLine={false} tickLine={false} interval={3} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="train" stroke={MINT} strokeWidth={2.5} dot={false} name="Train" />
              <Line type="monotone" dataKey="val" stroke={AMBER} strokeWidth={2.5} dot={false} name="Validation" />
            </LineChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Prediction Breakdown">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={confusionCounts} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
              <XAxis dataKey="name" tick={{ ...axisStyle, fontSize: 10 }} axisLine={false} tickLine={false} />
              <YAxis tick={axisStyle} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="value" radius={[6, 6, 6, 6]}>
                {confusionCounts.map((entry, i) => (
                  <Cell key={i} fill={entry.name.startsWith('True') ? MINT : RED} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </div>

      {/* Prediction Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, marginBottom: 20 }}>
        <StatBox label="Total Predictions" value={model.evaluation.predictionSummary.totalPredictions} />
        <StatBox label="Correct Predictions" value={model.evaluation.predictionSummary.correctPredictions} positive />
        <StatBox label="Avg Confidence" value={`${model.evaluation.predictionSummary.avgConfidence}%`} />
      </div>

      {/* Recent Prediction History */}
      <div style={{ marginBottom: 8 }}>
        <Panel title="Recent Prediction History">
          <Table
            size="small"
            pagination={false}
            dataSource={predictionHistory.map((r) => ({ ...r, key: r.id }))}
            columns={[
              { title: 'Timestamp', dataIndex: 'timestamp', key: 'timestamp' },
              {
                title: 'Predicted', dataIndex: 'predicted', key: 'predicted',
                render: (v) => <span style={{ color: v === 'Up' ? MINT : RED, fontWeight: 600 }}>{v}</span>,
              },
              {
                title: 'Actual', dataIndex: 'actual', key: 'actual',
                render: (v) => <span style={{ color: v === 'Up' ? MINT : RED, fontWeight: 600 }}>{v}</span>,
              },
              { title: 'Confidence', dataIndex: 'confidence', key: 'confidence', render: (v) => `${v}%` },
              {
                title: 'Result', dataIndex: 'correct', key: 'correct',
                render: (v) => (
                  <Tag style={{
                    background: v ? 'rgba(61,220,151,0.12)' : 'rgba(240,70,107,0.14)',
                    color: v ? MINT : RED, border: 'none', borderRadius: 8, fontWeight: 600,
                  }}>
                    {v ? 'Correct' : 'Incorrect'}
                  </Tag>
                ),
              },
            ]}
          />
        </Panel>
      </div>
    </div>
  );
}