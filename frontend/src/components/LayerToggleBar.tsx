import { Layers } from "lucide-react";
import type { LayerVisibility } from "../types/nowcast";

interface LayerToggleBarProps {
  layers: LayerVisibility;
  onToggle: (key: keyof LayerVisibility) => void;
  gaugeCount: number;
  stormCount: number;
}

const ITEMS: { key: keyof LayerVisibility; label: string; dot: string }[] = [
  { key: "satellite", label: "IR Loop", dot: "bg-slate-200" },
  { key: "districts", label: "Districts", dot: "bg-emerald-400" },
  { key: "seaZones", label: "Sea Zones", dot: "bg-blue-400" },
  { key: "floodBasins", label: "Flood Basins", dot: "bg-sky-400" },
  { key: "riverGauges", label: "River Gauges", dot: "bg-teal-400" },
  { key: "stormMasks", label: "Storm Masks", dot: "bg-rose-400" },
  { key: "projectedTrajectories", label: "+30/60/120m", dot: "bg-amber-400" },
  { key: "velocityArrows", label: "Vectors", dot: "bg-cyan-400" },
  { key: "surfaceStations", label: "Stations", dot: "bg-violet-400" },
];

export function LayerToggleBar({ layers, onToggle, gaugeCount, stormCount }: LayerToggleBarProps) {
  return (
    <div
      role="toolbar"
      aria-label="Map Layers"
      className="pointer-events-auto flex flex-wrap items-center gap-1.5 rounded-xl border border-slate-700/80 bg-slate-950/90 px-2.5 py-2 shadow-xl backdrop-blur-md"
    >
      <span className="mr-1 flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-slate-400">
        <Layers className="h-3.5 w-3.5 text-cyan-400" />
        Layers
      </span>

      {ITEMS.map((item) => {
        const checked = layers[item.key];
        const extra =
          item.key === "riverGauges"
            ? ` (${gaugeCount})`
            : item.key === "stormMasks"
            ? ` (${stormCount})`
            : "";

        return (
          <label
            key={item.key}
            className={`inline-flex cursor-pointer select-none items-center gap-1.5 rounded-lg border px-2 py-1 text-[11px] font-medium transition ${
              checked
                ? "border-cyan-500/45 bg-slate-900 text-slate-100"
                : "border-slate-800 bg-slate-950/70 text-slate-400 hover:text-slate-200"
            }`}
          >
            <input
              type="checkbox"
              checked={checked}
              onChange={() => onToggle(item.key)}
              {...({ switch: "" } as Record<string, string>)}
              className="h-3.5 w-6"
            />
            <span className={`h-2 w-2 rounded-full ${item.dot}`} />
            <span>
              {item.label}
              {extra}
            </span>
          </label>
        );
      })}
    </div>
  );
}
