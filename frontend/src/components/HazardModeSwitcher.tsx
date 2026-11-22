import { CloudRain, CloudLightning, Wind, ShieldAlert, Layers } from "lucide-react";
import type { HazardMapMode } from "../types/nowcast";

interface HazardModeSwitcherProps {
  mode: HazardMapMode;
  onChange: (mode: HazardMapMode) => void;
}

const MODES: { id: HazardMapMode; label: string; icon: typeof CloudRain }[] = [
  { id: "combined", label: "Combined Max", icon: ShieldAlert },
  { id: "rainfall", label: "Rainfall", icon: CloudRain },
  { id: "thunderstorm", label: "Thunderstorm", icon: CloudLightning },
  { id: "high_wind", label: "High Wind", icon: Wind },
  { id: "threat", label: "Threat Tier", icon: Layers },
];

export function HazardModeSwitcher({ mode, onChange }: HazardModeSwitcherProps) {
  return (
    <div
      role="group"
      aria-label="Hazard Probability Layer Switcher"
      className="pointer-events-auto inline-flex flex-wrap items-center gap-2 rounded-xl border border-slate-700/80 bg-slate-950/90 px-2.5 py-1.5 shadow-xl backdrop-blur-md"
    >
      <div className="flex flex-wrap items-center gap-1">
        {MODES.map((m) => {
          const Icon = m.icon;
          const active = mode === m.id;
          return (
            <button
              key={m.id}
              type="button"
              aria-pressed={active}
              onClick={() => onChange(m.id)}
              className={`inline-flex items-center gap-1 rounded-lg border px-2 py-1 text-[11px] font-medium transition ${
                active
                  ? "border-cyan-400/70 bg-cyan-500/20 text-cyan-100 shadow-[0_0_10px_rgba(34,211,238,0.18)]"
                  : "border-slate-800 bg-slate-900/70 text-slate-400 hover:border-slate-700 hover:text-slate-200"
              }`}
            >
              <Icon className="h-3 w-3" aria-hidden="true" />
              <span>{m.label}</span>
            </button>
          );
        })}
      </div>

      {mode !== "threat" ? (
        <div
          aria-label="Probability color ramp legend"
          className="flex items-center gap-2 border-l border-slate-800 pl-2 font-mono text-[10px] text-slate-400"
        >
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-sm border border-slate-600 bg-slate-700/35" />
            &lt;30%
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-sm bg-yellow-400/75" />
            30–50%
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-sm bg-orange-500/85" />
            50–75%
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="h-2.5 w-2.5 rounded-sm bg-rose-500/90" />
            &gt;75%
          </span>
        </div>
      ) : null}
    </div>
  );
}
