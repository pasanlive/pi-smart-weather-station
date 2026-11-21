import type { GaugeHistoryPoint } from "../types/nowcast";

/**
 * Generates a compact inline SVG 24-hour water-level sparkline string for Leaflet popups.
 */
export function renderGaugeSparklineSvg(
  history: GaugeHistoryPoint[],
  alertLevel: number | null,
  strokeColor = "#38bdf8"
): string {
  const width = 210;
  const height = 46;
  const pad = 4;

  if (!history || history.length < 2) {
    return `<div style="font-size:11px;color:#94a3b8;padding:4px 0;">Insufficient 24h history for sparkline</div>`;
  }

  const vals = history.map((h) => h.water_level_ft);
  if (alertLevel !== null) vals.push(alertLevel);
  const minV = Math.min(...vals) - 0.5;
  const maxV = Math.max(...vals) + 0.5;
  const span = Math.max(0.5, maxV - minV);

  const pts = history
    .map((pt, idx) => {
      const x = pad + (idx / Math.max(1, history.length - 1)) * (width - pad * 2);
      const y = height - pad - ((pt.water_level_ft - minV) / span) * (height - pad * 2);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");

  let alertLine = "";
  if (alertLevel !== null) {
    const ay = height - pad - ((alertLevel - minV) / span) * (height - pad * 2);
    alertLine = `<line x1="${pad}" y1="${ay.toFixed(1)}" x2="${width - pad}" y2="${ay.toFixed(1)}" stroke="#facc15" stroke-width="1" stroke-dasharray="3,2" opacity="0.7" />`;
  }

  return `
    <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="display:block;background:#0f172a;border:1px solid #1e293b;border-radius:6px;margin-top:4px;">
      ${alertLine}
      <polyline fill="none" stroke="${strokeColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" points="${pts}" />
    </svg>
  `;
}
