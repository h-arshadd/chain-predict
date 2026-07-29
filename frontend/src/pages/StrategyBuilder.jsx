import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Select, Spin, InputNumber, Checkbox, Radio, Input, message, Empty, DatePicker } from 'antd';
import { PlayCircleOutlined, SaveOutlined, RiseOutlined, SwapOutlined, ThunderboltOutlined, FundOutlined, LineChartOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import { api } from '../lib/api';
import Panel, { panelGradient as panel } from '../components/Panel';

// Same palette every other page uses (Backtests.jsx, Dashboard.jsx).
const MINT = '#3DDC97';
const AMBER = '#FF8A5C';
const BLUE = '#5B9CF6';

// Playbook entries only carry a strategy_name from the API -- no
// category/description field exists server-side. Infer a rough category +
// icon from the name itself (it already encodes indicator patterns like
// ema/rsi/macd) purely for a friendlier list item; falls back to a generic
// "Strategy" label when nothing matches.
function inferStrategyMeta(name) {
  const n = name.toLowerCase();
  if (n.includes('trend') || n.includes('ema') || n.includes('sma')) {
    return { icon: <RiseOutlined />, color: MINT, label: 'Trend following' };
  }
  if (n.includes('rsi') || n.includes('reversal') || n.includes('mean')) {
    return { icon: <SwapOutlined />, color: AMBER, label: 'Mean reversion' };
  }
  if (n.includes('macd') || n.includes('cross')) {
    return { icon: <LineChartOutlined />, color: BLUE, label: 'Momentum' };
  }
  if (n.includes('scalp') || n.includes('extreme')) {
    return { icon: <ThunderboltOutlined />, color: AMBER, label: 'Scalping' };
  }
  return { icon: <FundOutlined />, color: '#9096A0', label: 'Strategy' };
}

const COMBINE_RULES = ['AND', 'OR', 'MAJORITY', 'WEIGHTED'];

// Only Bybit is currently run (see STRATEGY_BUILDER_SPEC.md) -- exchange
// is fixed here rather than offered as a picker the backend would just
// override anyway (POST /api/strategies/build defaults to "bybit").
const EXCHANGE = 'bybit';
const COINS = ['btc', 'eth', 'sol', 'doge', 'ada', 'ltc', 'mina', 'sui'];

// Same guard Backtests.jsx uses -- no candle data in the DB before this,
// for any coin/exchange (see crypto_pipeline/data/*/config_*.yml).
const EARLIEST_DATA_DATE = dayjs('2024-01-01');

// Just the common shortcuts shown as quick-pick suggestions -- the field
// itself accepts ANY timeframe (see TIMEFRAME_RE), since the backend
// resamples from raw 1-minute candles and isn't limited to this list.
const TIME_HORIZON_PRESETS = ['5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w'];

// Mirrors crypto_pipeline.data.data_downloader.normalize_timeframe on the
// backend -- a number followed by min/m (minutes), h (hours), d (days),
// or w (weeks). Checked client-side just for instant feedback; the
// backend re-validates/normalizes this exact same shape regardless.
const TIMEFRAME_RE = /^\d+\s*(min|m|h|d|w)$/i;

// Mirrors crypto_pipeline.data.data_downloader.normalize_timeframe --
// same shorthand ("15m") -> pandas-valid form ("15min") mapping, so what
// this page sends to /api/ml-models?timeframe= matches the normalized
// form time_horizon is actually stored as (build_and_save_strategy
// normalizes it the same way before saving).
const TIMEFRAME_UNIT_MAP = { min: 'min', m: 'min', h: 'h', d: 'D', w: 'W' };
function normalizeTimeHorizon(value) {
  const match = TIMEFRAME_RE.exec((value || '').trim().replace(/\s+/g, ''));
  if (!match) return null;
  const amount = match[0].match(/^\d+/)[0];
  const unit = match[1].toLowerCase();
  return `${amount}${TIMEFRAME_UNIT_MAP[unit]}`;
}

const primaryBtnStyle = {
  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8, width: '100%',
  padding: '12px 18px', borderRadius: 10, border: 'none', background: MINT, color: '#0B0F12',
  fontWeight: 700, fontSize: 13.5, cursor: 'pointer',
};

const secondaryBtnStyle = {
  ...primaryBtnStyle,
  background: 'rgba(255,255,255,0.06)',
  color: '#F5F6F7',
  border: '1px solid rgba(255,255,255,0.1)',
};

export default function StrategyBuilder() {
  const navigate = useNavigate();

  // --- Panel 1: Playbook ---
  const [playbooks, setPlaybooks] = useState([]);
  const [playbooksLoading, setPlaybooksLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState([]);
  const [persistBars, setPersistBars] = useState({}); // playbook_id -> int

  // --- Panel 2: Build ---
  const [combineRule, setCombineRule] = useState('AND');
  const [mlModels, setMlModels] = useState([]);
  const [mlModelsLoading, setMlModelsLoading] = useState(false);
  const [selectedMlRunIds, setSelectedMlRunIds] = useState([]);
  const [coin, setCoin] = useState('btc');
  const [timeHorizon, setTimeHorizon] = useState('1h');
  const timeHorizonValid = TIMEFRAME_RE.test((timeHorizon || '').trim());

  // --- Panel 3: Backtest / save ---
  const [strategyName, setStrategyName] = useState('');
  const [takeProfitValue, setTakeProfitValue] = useState(2);
  const [stopLossValue, setStopLossValue] = useState(1);
  // Backtest-only -- not part of buildStrategyPayload(), since a saved
  // strategy config has no date range of its own (see StrategyBuildRequest);
  // only the "Run Backtest" action needs one, same as Backtests.jsx's own
  // New Backtest modal.
  const [dateRange, setDateRange] = useState([dayjs().subtract(3, 'month'), dayjs()]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    api.get('/api/playbook?limit=200')
      .then((res) => setPlaybooks(res.data))
      .catch((err) => message.error(`Couldn't load playbook: ${err.message}`))
      .finally(() => setPlaybooksLoading(false));
  }, []);

  // ML models are restricted to the strategy's own time_horizon (PDF:
  // "Only models trained on the same timeframe ... should be available").
  // Normalized the same way the backend stores time_horizon, so e.g.
  // typing "1H" or "60min" still matches models saved as "1h". Skipped
  // entirely while the field doesn't parse yet, rather than querying
  // with a half-typed value on every keystroke.
  useEffect(() => {
    const normalized = normalizeTimeHorizon(timeHorizon);
    if (!normalized) {
      setMlModels([]);
      setSelectedMlRunIds([]);
      return;
    }
    setMlModelsLoading(true);
    setSelectedMlRunIds([]);
    api.get(`/api/ml-models?timeframe=${encodeURIComponent(normalized)}&limit=200`)
      .then((res) => setMlModels(res.data))
      .catch(() => setMlModels([])) // endpoint/filter may not exist yet -- fail quiet, ML step is optional
      .finally(() => setMlModelsLoading(false));
  }, [timeHorizon]);

  const togglePlaybook = (playbookId) => {
    setSelectedIds((prev) =>
      prev.includes(playbookId) ? prev.filter((id) => id !== playbookId) : [...prev, playbookId]
    );
  };

  const selectedPlaybooks = useMemo(
    () => playbooks.filter((p) => selectedIds.includes(p.playbook_id)),
    [playbooks, selectedIds]
  );

  // Default name: single playbook entry, nothing else selected -> reuse
  // its own name (still editable). Matches assemble.default_strategy_name.
  useEffect(() => {
    if (selectedPlaybooks.length === 1 && selectedMlRunIds.length === 0) {
      setStrategyName(selectedPlaybooks[0].strategy_name);
    }
  }, [selectedPlaybooks, selectedMlRunIds]);

  const buildComponents = () => [
    ...selectedIds.map((playbook_id) => ({
      kind: 'playbook',
      playbook_id,
      persist_bars: persistBars[playbook_id] || 0,
    })),
    ...selectedMlRunIds.map((run_id) => ({
      kind: 'ml_model',
      run_id,
      persist_bars: 0,
    })),
  ];

  const canBuild = selectedIds.length + selectedMlRunIds.length >= 1
    && strategyName.trim().length > 0
    && timeHorizonValid;

  // Shape shared by both actions -- saveStrategy sends it as the whole
  // body to /api/strategies/build; runBacktest nests it under
  // ad_hoc_strategy in the /api/backtests body (see AdHocStrategyConfig
  // on the backend). Same fields either way.
  const buildStrategyPayload = () => ({
    strategy_name: strategyName.trim(),
    components: buildComponents(),
    combine_rule: combineRule,
    coin,
    exchange: EXCHANGE,
    time_horizon: timeHorizon,
    take_profit_type: takeProfitValue != null ? 'percentage' : null,
    take_profit_value: takeProfitValue,
    stop_loss_type: stopLossValue != null ? 'percentage' : null,
    stop_loss_value: stopLossValue,
  });

  const validateBeforeSubmit = () => {
    if (canBuild) return true;
    message.warning(
      !timeHorizonValid
        ? 'Enter a valid timeframe (e.g. "15m", "1h", "4h", "1d").'
        : 'Select at least one playbook entry or ML model, and enter a strategy name.'
    );
    return false;
  };

  // "Save Strategy Only" -- the ONLY action that writes to
  // metadata.strategy. Per Strategy_Builder_Module.pdf: backtesting and
  // "Saving Strategies" are described as two separate steps ("If the
  // user is satisfied with the results, the strategy can be saved") --
  // running a backtest must never implicitly save a strategy.
  const saveStrategy = async () => {
    if (!validateBeforeSubmit()) return null;
    setSaving(true);
    try {
      const res = await api.post('/api/strategies/build', buildStrategyPayload());
      message.success(`Strategy "${res.data.strategy_name}" saved.`);
      return res.data.strategy_id;
    } catch (err) {
      message.error(err.message || 'Failed to save strategy');
      return null;
    } finally {
      setSaving(false);
    }
  };

  // "Run Backtest" -- runs against the assembled components directly.
  // Posts the full definition inline (ad_hoc_strategy) instead of a
  // strategy_id, so no metadata.strategy row is created. Only
  // metadata.backtest gets a row, same as any other backtest request.
  //
  // start_date/end_date are required here (unlike saveStrategy) --
  // without them the backend silently fell back to backtest/config.yaml's
  // hardcoded default window (2026-04-01 -> 2026-06-01) regardless of
  // what was actually selected, so every ad-hoc run tested the same
  // fixed 2-month slice no matter the coin or intent.
  const runBacktest = async () => {
    if (!validateBeforeSubmit()) return;
    if (!dateRange || !dateRange[0] || !dateRange[1]) {
      message.warning('Select a date range to backtest.');
      return;
    }
    setSaving(true);
    try {
      const res = await api.post('/api/backtests', {
        ad_hoc_strategy: buildStrategyPayload(),
        start_date: dateRange[0].format('YYYY-MM-DD'),
        end_date: dateRange[1].format('YYYY-MM-DD'),
      });
      message.success('Backtest submitted — running in the background.');
      navigate(`/backtests/${res.data.backtest_id}`);
    } catch (err) {
      message.error(err.message || 'Failed to start backtest');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div style={{ paddingTop: 8 }}>
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, color: '#F5F6F7', margin: 0 }}>Strategy Builder</h1>
        <p style={{ color: '#9096A0', fontSize: 13.5, marginTop: 6 }}>
          Combine playbook strategies and ML models into a new strategy, then backtest it.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1.3fr 1fr', gap: 20, alignItems: 'start' }}>
        {/* ---------------- Panel 1: Playbook ---------------- */}
        <Panel title="1. Playbook" hint="Select one or more reusable strategies" variant="gradient" style={{ maxHeight: 640, overflowY: 'auto' }}>
          {playbooksLoading ? (
            <div style={{ textAlign: 'center', padding: 40 }}><Spin /></div>
          ) : playbooks.length === 0 ? (
            <Empty description={<span style={{ color: '#9096A0' }}>No playbook entries yet</span>} />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {playbooks.map((p) => {
                const checked = selectedIds.includes(p.playbook_id);
                const meta = inferStrategyMeta(p.strategy_name);
                return (
                  <div
                    key={p.playbook_id}
                    onClick={() => togglePlaybook(p.playbook_id)}
                    style={{
                      display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10,
                      padding: '10px 12px', borderRadius: 12, cursor: 'pointer',
                      background: checked ? 'rgba(61,220,151,0.10)' : 'rgba(255,255,255,0.03)',
                      border: `1px solid ${checked ? 'rgba(61,220,151,0.35)' : 'rgba(255,255,255,0.06)'}`,
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
                      <div style={{
                        width: 32, height: 32, borderRadius: 9, flexShrink: 0,
                        background: `${meta.color}18`, color: meta.color,
                        display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 15,
                      }}>
                        {meta.icon}
                      </div>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontWeight: 600, color: '#F5F6F7', fontSize: 13.5, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                          {p.strategy_name}
                        </div>
                        <div style={{ fontSize: 11.5, color: '#8A909A', marginTop: 1 }}>{meta.label}</div>
                      </div>
                    </div>
                    <Checkbox checked={checked} onChange={() => togglePlaybook(p.playbook_id)} />
                  </div>
                );
              })}
            </div>
          )}
        </Panel>

        {/* ---------------- Panel 2: Build Strategy ---------------- */}
        <Panel title="2. Build Strategy" hint="Combine strategies and ML models" variant="gradient">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <div>
              <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>SELECTED STRATEGIES</div>
              {selectedPlaybooks.length === 0 ? (
                <div style={{ color: '#5C6370', fontSize: 13 }}>Pick from the Playbook on the left.</div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {selectedPlaybooks.map((p) => (
                    <div key={p.playbook_id} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                      <span style={{ color: '#F5F6F7', fontSize: 13, fontWeight: 600 }}>{p.strategy_name}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span style={{ color: '#9096A0', fontSize: 11.5 }}>Persist Bars</span>
                        <InputNumber
                          min={0}
                          size="small"
                          style={{ width: 64 }}
                          value={persistBars[p.playbook_id] || 0}
                          onChange={(v) => setPersistBars((prev) => ({ ...prev, [p.playbook_id]: v || 0 }))}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div>
              <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>OPTIONAL: ADD ML MODEL (same timeframe)</div>
              <Select
                mode="multiple"
                allowClear
                loading={mlModelsLoading}
                placeholder={mlModels.length === 0 ? 'No models trained on this timeframe' : 'Select ML models'}
                style={{ width: '100%' }}
                value={selectedMlRunIds}
                onChange={setSelectedMlRunIds}
                options={mlModels.map((m) => ({ value: m.run_id, label: `${m.algorithm} — ${m.symbol} · ${m.timeframe}` }))}
              />
            </div>

            <div>
              <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>COMBINE LOGIC</div>
              <Radio.Group value={combineRule} onChange={(e) => setCombineRule(e.target.value)}>
                {COMBINE_RULES.map((r) => (
                  <Radio.Button key={r} value={r}>{r}</Radio.Button>
                ))}
              </Radio.Group>
            </div>

            <div style={{ display: 'flex', gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>COIN</div>
                <Select style={{ width: '100%' }} value={coin} onChange={setCoin}
                  options={COINS.map((c) => ({ value: c, label: `${c.toUpperCase()} · Bybit` }))} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>TIMEFRAME</div>
                <Select
                  mode="tags"
                  maxCount={1}
                  style={{ width: '100%' }}
                  value={timeHorizon ? [timeHorizon] : []}
                  status={timeHorizonValid ? undefined : 'error'}
                  placeholder="e.g. 15m, 1h, 4h, 1d"
                  // Pick a preset or type your own -- anything the
                  // backend can resample from 1m candles (see
                  // normalize_timeframe on the backend for the exact
                  // accepted shape: <number><min|m|h|d|w>). tags mode +
                  // maxCount=1 is what lets a typed value commit as the
                  // selection without needing to already be an option.
                  options={TIME_HORIZON_PRESETS.map((t) => ({ value: t, label: t }))}
                  onChange={(vals) => setTimeHorizon(vals[vals.length - 1] || '')}
                />
                {!timeHorizonValid && (
                  <div style={{ color: '#F0466B', fontSize: 11, marginTop: 4 }}>
                    Use a number + unit: min/m, h, d, or w (e.g. "15m", "4h", "1d").
                  </div>
                )}
              </div>
            </div>
          </div>
        </Panel>

        {/* ---------------- Panel 3: Save / Backtest ---------------- */}
        <Panel title="3. Save & Backtest" hint="Name it, set TP/SL, then run or save" variant="gradient">
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>STRATEGY NAME</div>
              <Input
                placeholder={`${coin.toUpperCase()}_${timeHorizon}_...`}
                value={strategyName}
                onChange={(e) => setStrategyName(e.target.value)}
              />
            </div>

            <div style={{ display: 'flex', gap: 12 }}>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>TAKE PROFIT (%)</div>
                <InputNumber min={0} step={0.1} style={{ width: '100%' }} value={takeProfitValue} onChange={setTakeProfitValue} />
              </div>
              <div style={{ flex: 1 }}>
                <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>STOP LOSS (%)</div>
                <InputNumber min={0} step={0.1} style={{ width: '100%' }} value={stopLossValue} onChange={setStopLossValue} />
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 0, padding: '12px 16px', borderRadius: 12, background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingRight: 14 }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: MINT, flexShrink: 0 }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: '#D5D8DC' }}>{EXCHANGE}</span>
              </div>
              <div style={{ width: 1, height: 14, background: 'rgba(255,255,255,0.08)' }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0 14px' }}>
                <span style={{ width: 6, height: 6, borderRadius: '50%', background: AMBER, flexShrink: 0 }} />
                <span style={{ fontSize: 13, fontWeight: 600, color: '#D5D8DC' }}>{coin.toUpperCase()}</span>
              </div>
              <div style={{ width: 1, height: 14, background: 'rgba(255,255,255,0.08)' }} />
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, paddingLeft: 14 }}>
                <span style={{ fontSize: 13, fontWeight: 600, color: '#9096A0' }}>{timeHorizon}</span>
              </div>
            </div>

            <div>
              <div style={{ fontSize: 12.5, color: '#9096A0', marginBottom: 8, fontWeight: 600 }}>BACKTEST DATE RANGE</div>
              <DatePicker.RangePicker
                style={{ width: '100%' }}
                value={dateRange}
                onChange={setDateRange}
                disabledDate={(current) => !!current && (current < EARLIEST_DATA_DATE.startOf('day') || current > dayjs().endOf('day'))}
              />
            </div>

            <button style={primaryBtnStyle} disabled={!canBuild || saving} onClick={runBacktest}>
              <PlayCircleOutlined /> Run Backtest
            </button>
            <button style={secondaryBtnStyle} disabled={!canBuild || saving} onClick={saveStrategy}>
              <SaveOutlined /> Save Strategy Only
            </button>
          </div>
        </Panel>
      </div>
    </div>
  );
}