import { Cpu, Activity, RefreshCw, Radio, Waves, Map, Sparkles } from "lucide-react";
import type { HydrologySourceStatus, SystemStatusResponse } from "../types/nowcast";

export type DashboardTab = "nowcast" | "active-learning";

interface StatusHeaderProps {
  status: SystemStatusResponse | null;
  hydrologyStatus: HydrologySourceStatus;
  wsConnected: boolean;
  isRefreshing: boolean;
  onRefresh: () => void;
  activeTab: DashboardTab;
  onTabChange: (tab: DashboardTab) => void;
  pendingAlCount: number;
}

const HYDRO_BADGE: Record<HydrologySourceStatus, { label: string; cls: string }> = {
  arcgis_live: {
    label: "Irrigation Dept ArcGIS • Live",
    cls: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  },
  fallback: {
    label: "Hydrology • Fallback",
    cls: "bg-sky-500/15 text-sky-300 border-sky-500/40",
  },
  stale: {
    label: "Irrigation Dept • Stale",
    cls: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  },
  unavailable: {
    label: "Irrigation Dept • Unavailable",
    cls: "bg-rose-500/15 text-rose-300 border-rose-500/40",
  },
};

function formatSltTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Asia/Colombo",
    }) + " SLT";
  } catch {
    return iso;
  }
}

export function StatusHeader({
  status,
  hydrologyStatus,
  wsConnected,
  isRefreshing,
  onRefresh,
  activeTab,
  onTabChange,
  pendingAlCount,
}: StatusHeaderProps) {
  const isHailo = status?.inference_backend === "hailo-8";
  const hydro = HYDRO_BADGE[hydrologyStatus] ?? HYDRO_BADGE.unavailable;

  const handleTabKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      onTabChange(activeTab === "nowcast" ? "active-learning" : "nowcast");
    }
  };

  return (
    <header className="border-b border-slate-800/90 bg-slate-950/95 px-3.5 py-2.5">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="inline-flex h-2.5 w-2.5 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee]" />
            <h1 className="truncate text-sm font-bold tracking-wide text-slate-100 uppercase">
              SL Edge Nowcast
            </h1>
            <span
              className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-mono ${
                wsConnected
                  ? "bg-emerald-500/15 text-emerald-300"
                  : "bg-slate-800 text-slate-400"
              }`}
            >
              <Radio className="h-2.5 w-2.5" />
              {wsConnected ? "LIVE" : "POLL"}
            </span>
          </div>
          <p className="mt-0.5 text-[11px] text-slate-400">
            Himawari-9 IR • Optical Flow • Wet-Zone Hydrology
          </p>
        </div>

        <button
          type="button"
          onClick={onRefresh}
          disabled={isRefreshing}
          className="inline-flex shrink-0 items-center gap-1.5 rounded-md border border-cyan-500/40 bg-cyan-500/10 px-2.5 py-1.5 text-xs font-medium text-cyan-200 transition hover:bg-cyan-500/20 disabled:opacity-50"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${isRefreshing ? "animate-spin" : ""}`} />
          {isRefreshing ? "Syncing" : "Refresh"}
        </button>
      </div>

      {/* Hardware / Inference Engine & Hydrology Source Badges */}
      <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
        <span
          className={`inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium ${
            isHailo
              ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
              : "border-cyan-500/40 bg-cyan-500/15 text-cyan-200"
          }`}
        >
          <Cpu className="h-3 w-3" />
          {isHailo ? "Hailo-8 26 TOPS" : "CPU Fallback"}
          {status ? (
            <span className="font-mono text-[10px] opacity-85">
              • {status.last_inference_ms.toFixed(1)} ms
            </span>
          ) : null}
        </span>

        <span
          className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-medium ${hydro.cls}`}
        >
          <Waves className="h-3 w-3" />
          {hydro.label}
        </span>
      </div>

      {/* Pi 5 Telemetry Strip */}
      <div className="mt-2 grid grid-cols-3 gap-1.5 rounded-md border border-slate-800/80 bg-slate-900/70 px-2.5 py-1.5 text-[11px] font-mono">
        <div className="flex items-center gap-1 text-slate-300">
          <Activity className="h-3 w-3 text-cyan-400" />
          <span>CPU {status ? `${status.cpu_percent.toFixed(0)}%` : "—"}</span>
        </div>
        <div className="text-center text-slate-300">
          RAM {status ? `${status.ram_used_gb.toFixed(1)}/${status.ram_total_gb.toFixed(0)}G` : "—"}
        </div>
        <div className="text-right text-slate-400">
          {formatSltTime(status?.last_frame_timestamp)}
        </div>
      </div>

      {/* Dashboard Mode Tabs: Live Nowcast vs Active Learning QA */}
      <div
        role="tablist"
        aria-label="Dashboard workspace view"
        onKeyDown={handleTabKeyDown}
        className="dashboard-tablist mt-2.5 gap-1 border-t border-slate-800/80 pt-2"
      >
        <button
          type="button"
          id="tab-nowcast"
          role="tab"
          aria-selected={activeTab === "nowcast"}
          tabIndex={activeTab === "nowcast" ? 0 : -1}
          onClick={() => onTabChange("nowcast")}
          className={`inline-flex items-center gap-1.5 rounded-t px-2.5 py-1.5 text-xs font-semibold transition ${
            activeTab === "nowcast"
              ? "text-cyan-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Map className="h-3.5 w-3.5" aria-hidden="true" />
          Live Nowcast
        </button>

        <button
          type="button"
          id="tab-active-learning"
          role="tab"
          aria-selected={activeTab === "active-learning"}
          tabIndex={activeTab === "active-learning" ? 0 : -1}
          onClick={() => onTabChange("active-learning")}
          className={`inline-flex items-center gap-1.5 rounded-t px-2.5 py-1.5 text-xs font-semibold transition ${
            activeTab === "active-learning"
              ? "text-cyan-200"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
          Active Learning
          {pendingAlCount > 0 ? (
            <span className="rounded-full bg-amber-500/20 px-1.5 py-0.2 font-mono text-[10px] font-bold text-amber-300 ring-1 ring-amber-500/40">
              {pendingAlCount}
            </span>
          ) : null}
        </button>
      </div>
    </header>
  );
}
