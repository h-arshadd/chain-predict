const MINT = '#3DDC97';
const RED = '#F0466B';

/** Format a number as USD currency, or '—' if null/undefined. */
export const fmtUsd = (v) =>
  v == null ? '—' : v.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 });

/**
 * Color for a PnL-style value: green if positive, red if negative,
 * `nullColor` if the value is missing, `zeroColor` if it's exactly 0.
 * Most pages use a dimmer grey for missing vs. zero; a couple of list
 * pages use the same light color for both, so both are overridable.
 */
export const pnlColor = (v, nullColor = '#6B7280', zeroColor = '#9096A0') =>
  v == null ? nullColor : v > 0 ? MINT : v < 0 ? RED : zeroColor;