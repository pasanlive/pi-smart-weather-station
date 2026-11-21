import { Waves, AlertTriangle } from "lucide-react";
import type { BasinRiskLevel, RiverBasinRisk } from "../types/nowcast";

interface BasinRiskPanelProps {
  basins: RiverBasinRisk[];
  selectedBasinId: string | null;
  onSelectBasin: (basinId: string) => void;
}

const RISK_STYLE: Record<BasinRiskLevel, string> = {
  NORMAL: "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  WATCH: "border-yellow-500/45 bg-yellow-500/15 text-yellow-200",
  WARNING: "border-orange-500/50 bg-orange-500/20 text-orange-200",
  CRITICAL: "border-rose-500/60 bg-rose-500/25 text-rose-200",
};

function formatEta(etaMin: number | null): string {
  if (etaMin === 0) return "NOW";
  if (etaMin !== null) return `+${etaMin}m`;
  return "—";
}

export function BasinRiskPanel({ basins, selectedBasinId, onSelectBasin }: BasinRiskPanelProps) {
  return (
    <section
      aria-label="Wet-Zone River Basin Flood Risk"
      className="rounded-lg border border-slate-800 bg-slate-900/75 p-3"
    >
      <div className="mb-2 flex items-center justify-between">
        <h2 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-sky-300">
          <Waves className="h-3.5 w-3.5" />
          River Basin Flood Risk
        </h2>
        <span className="font-mono text-[10px] text-slate-400">4 Wet-Zone Basins</span>
      </div>

      <div className="grid grid-cols-1 gap-1.5">
        {basins.map((b) => {
          const isSelected = selectedBasinId === b.basin_id;
          return (
            <button
              key={b.basin_id}
              type="button"
              onClick={() => onSelectBasin(b.basin_id)}
              className={`rounded-md border px-2.5 py-2 text-left transition ${
                isSelected
                  ? "border-cyan-400/70 bg-cyan-950/35"
                  : "border-slate-800 bg-slate-950/60 hover:border-slate-700 hover:bg-slate-900"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="truncate text-xs font-semibold text-slate-100">{b.name}</span>
                  {b.upstream_surge ? (
                    <span
                      title="Upstream surge detected"
                      className="inline-flex items-center gap-0.5 rounded bg-amber-500/25 px-1 py-0.5 font-mono text-[9px] font-bold text-amber-200"
                    >
                      <AlertTriangle className="h-2.5 w-2.5" />
                      SURGE
                    </span>
                  ) : null}
                </div>

                <div className="flex shrink-0 items-center gap-1.5">
                  <span
                    className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-semibold ${
                      b.eta_min !== null
                        ? "border-cyan-500/40 bg-cyan-500/15 text-cyan-200"
                        : "border-slate-700 bg-slate-800/80 text-slate-400"
                    }`}
                  >
                    ETA {formatEta(b.eta_min)}
                  </span>
                  <span
                    className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold ${RISK_STYLE[b.risk_level]}`}
                  >
                    {b.risk_level}
                  </span>
                </div>
              </div>

              <div className="mt-1.5 grid grid-cols-3 gap-1 font-mono text-[10px] text-slate-400">
                <span>
                  24h: <strong className="text-slate-200">{b.rainfall_24h_mm.toFixed(0)} mm</strong>
                </span>
                <span>
                  Now/+60: <strong className="text-slate-200">{b.storm_coverage_pct.toFixed(0)}%/{b.projected_60m_pct.toFixed(0)}%</strong>
                </span>
                <span className="text-right">
                  Peak: <strong className="text-cyan-300">{b.peak_coverage_pct.toFixed(0)}%</strong>
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}
