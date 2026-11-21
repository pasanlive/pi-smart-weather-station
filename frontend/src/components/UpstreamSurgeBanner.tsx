import { AlertTriangle } from "lucide-react";
import type { UpstreamSurgeWarning } from "../types/nowcast";

interface UpstreamSurgeBannerProps {
  warnings: UpstreamSurgeWarning[];
  onSelectBasin?: (basinId: string) => void;
}

export function UpstreamSurgeBanner({ warnings, onSelectBasin }: UpstreamSurgeBannerProps) {
  if (!warnings || warnings.length === 0) return null;

  return (
    <section aria-label="Upstream Surge Warnings" className="space-y-1.5">
      {warnings.map((w) => (
        <button
          key={w.basin_id}
          type="button"
          onClick={() => onSelectBasin?.(w.basin_id)}
          className="w-full rounded-lg border border-amber-500/60 bg-amber-500/15 p-2.5 text-left transition hover:bg-amber-500/20"
        >
          <div className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-1.5 text-xs font-bold uppercase tracking-wide text-amber-300">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
              Upstream Surge • {w.basin_name}
            </span>
            <span className="rounded bg-amber-500/25 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-amber-200">
              +{w.max_rate_of_rise_ft_hr.toFixed(2)} ft/hr
            </span>
          </div>
          <p className="mt-1 text-[11px] leading-snug text-amber-100/90">{w.message}</p>
          <div className="mt-1 flex items-center justify-between text-[10px] font-mono text-amber-200/80">
            <span>Crest ETA: {w.estimated_crest_window}</span>
            <span>Watch: {w.downstream_stations.slice(0, 2).join(", ")}</span>
          </div>
        </button>
      ))}
    </section>
  );
}
