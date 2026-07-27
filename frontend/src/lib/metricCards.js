// Subset of compute_stats()'s dynamically-discovered quantstats metrics
// dict (data.stats.metrics) to headline as cards -- the full ~55-metric
// dict still exists underneath, these are just the ones worth a card.
// "pct" metrics are ratios quantstats returns as fractions (0.42 -> 42%),
// "ratio" metrics are shown as-is to 2 decimals. `invert` means a LOWER
// number is the better outcome (drawdown, losses, risk measures), so the
// green/red coloring flips for those.
export const METRIC_CARDS = [
  { key: 'sharpe', label: 'Sharpe Ratio', kind: 'ratio' },
  { key: 'smart_sharpe', label: 'Smart Sharpe', kind: 'ratio' },
  { key: 'sortino', label: 'Sortino Ratio', kind: 'ratio' },
  { key: 'smart_sortino', label: 'Smart Sortino', kind: 'ratio' },
  { key: 'adjusted_sortino', label: 'Adjusted Sortino', kind: 'ratio' },
  { key: 'calmar', label: 'Calmar Ratio', kind: 'ratio' },
  { key: 'omega', label: 'Omega Ratio', kind: 'ratio' },
  { key: 'max_drawdown', label: 'Max Drawdown', kind: 'pct', invert: true },
  { key: 'cagr', label: 'CAGR', kind: 'pct' },
  { key: 'comp', label: 'Total Compounded Return', kind: 'pct' },
  { key: 'volatility', label: 'Volatility (ann.)', kind: 'pct', invert: true },
  { key: 'win_rate', label: 'Win Rate', kind: 'pct' },
  { key: 'win_loss_ratio', label: 'Win/Loss Ratio', kind: 'ratio' },
  { key: 'profit_factor', label: 'Profit Factor', kind: 'ratio' },
  { key: 'profit_ratio', label: 'Profit Ratio', kind: 'ratio' },
  { key: 'payoff_ratio', label: 'Payoff Ratio', kind: 'ratio' },
  { key: 'gain_to_pain_ratio', label: 'Gain to Pain Ratio', kind: 'ratio' },
  { key: 'best', label: 'Best Period', kind: 'pct' },
  { key: 'worst', label: 'Worst Period', kind: 'pct', invert: true },
  { key: 'avg_return', label: 'Avg Return', kind: 'pct' },
  { key: 'avg_win', label: 'Avg Win', kind: 'pct' },
  { key: 'avg_loss', label: 'Avg Loss', kind: 'pct', invert: true },
  { key: 'expected_return', label: 'Expected Return', kind: 'pct' },
  { key: 'expected_shortfall', label: 'Expected Shortfall (CVaR)', kind: 'pct', invert: true },
  { key: 'value_at_risk', label: 'Value at Risk', kind: 'pct', invert: true },
  { key: 'conditional_value_at_risk', label: 'Conditional VaR', kind: 'pct', invert: true },
  { key: 'kelly_criterion', label: 'Kelly Criterion', kind: 'pct' },
  { key: 'risk_of_ruin', label: 'Risk of Ruin', kind: 'pct', invert: true },
  { key: 'tail_ratio', label: 'Tail Ratio', kind: 'ratio' },
  { key: 'recovery_factor', label: 'Recovery Factor', kind: 'ratio' },
  { key: 'ulcer_index', label: 'Ulcer Index', kind: 'ratio', invert: true },
  { key: 'ulcer_performance_index', label: 'Ulcer Performance Index', kind: 'ratio' },
  { key: 'serenity_index', label: 'Serenity Index', kind: 'ratio' },
  { key: 'common_sense_ratio', label: 'Common Sense Ratio', kind: 'ratio' },
  { key: 'exposure', label: 'Exposure', kind: 'pct' },
  { key: 'consecutive_wins', label: 'Max Consecutive Wins', kind: 'plain' },
  { key: 'consecutive_losses', label: 'Max Consecutive Losses', kind: 'plain', invert: true },
  { key: 'skew', label: 'Skew', kind: 'ratio' },
  { key: 'kurtosis', label: 'Kurtosis', kind: 'ratio' },
];

/** Format a metric value according to its `kind` ('pct' | 'ratio' | 'plain'). */
export function fmtMetric(value, kind) {
  if (value == null || Number.isNaN(value)) return '—';
  if (kind === 'pct') return `${(value * 100).toFixed(2)}%`;
  if (kind === 'plain') return `${value}`;
  return value.toFixed(2);
}

// Two-card grouping for METRIC_CARDS -- split down the middle by count
// (not by kind) so both headline cards render the same number of rows
// and end up the same height, rather than a "ratios vs percentages"
// split that comes out lopsided (22 vs 17).
const _MIDPOINT = Math.ceil(METRIC_CARDS.length / 2);
export const METRIC_CARDS_COL_1 = METRIC_CARDS.slice(0, _MIDPOINT);
export const METRIC_CARDS_COL_2 = METRIC_CARDS.slice(_MIDPOINT);