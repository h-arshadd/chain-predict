/**
 * Turn compute_stats()'s {timestamp: value} records into a recharts-friendly
 * array, e.g. plots.drawdown.drawdown_series -> [{ ts, dd, label }, ...].
 */
export function recordsToSeries(records, valueKey = 'value') {
  if (!records) return [];
  return Object.entries(records)
    .map(([ts, val]) => ({ ts, [valueKey]: val, label: new Date(ts).toLocaleDateString() }))
    .filter((d) => d[valueKey] != null);
}

/** Turn plots.monthly_heatmap.monthly_returns into rows for the monthly-returns table. */
export function monthlyHeatmapToRows(monthly) {
  if (!monthly) return [];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return Object.entries(monthly)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([year, monthVals]) => ({
      year,
      cells: months.map((_, i) => monthVals[i + 1] ?? monthVals[String(i + 1)] ?? null),
    }));
}

/** Background color for a monthly-returns heatmap cell, scaled by magnitude. */
export function heatColor(v) {
  if (v == null) return 'rgba(255,255,255,0.03)';
  const intensity = Math.min(Math.abs(v) * 6, 1);
  return v >= 0 ? `rgba(61,220,151,${0.15 + intensity * 0.6})` : `rgba(240,70,107,${0.15 + intensity * 0.6})`;
}