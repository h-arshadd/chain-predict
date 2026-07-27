import { Tooltip as AntTooltip } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';

// Two background treatments used across the app. "flat" is a plain dark
// glass card (Details pages, Sentiment); "gradient" is a subtle diagonal
// gradient glass card (list/table pages: Backtests, Dashboard, Deployment,
// Models, Wallets). Pass variant="gradient" to opt into the latter.
export const panelFlat = {
  background: 'rgba(21, 26, 31, 0.75)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.07)',
  borderRadius: 22,
};

export const panelGradient = {
  background: 'linear-gradient(155deg, rgba(30, 36, 34, 0.8) 0%, rgba(19, 23, 27, 0.8) 100%)',
  backdropFilter: 'blur(16px)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 20,
};

/**
 * Card wrapper used for every dashboard/details section.
 *
 * - title: optional header text
 * - right / action: optional element rendered on the right of the header (aliases of each other)
 * - hint: optional tooltip icon next to the title
 * - column: stretches content in a flex column (used where children need to fill height, e.g. charts)
 * - variant: 'flat' (default) or 'gradient'
 */
export default function Panel({ title, children, style, right, action, hint, column = false, variant = 'flat' }) {
  const headerRight = right ?? action;
  const base = variant === 'gradient' ? panelGradient : panelFlat;

  return (
    <div style={{ ...base, padding: 22, ...(column ? { display: 'flex', flexDirection: 'column' } : {}), ...style }}>
      {(title || headerRight) && (
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
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
          {headerRight}
        </div>
      )}
      {column ? <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>{children}</div> : children}
    </div>
  );
}