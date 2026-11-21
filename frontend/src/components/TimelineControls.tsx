import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import type { FrameMetadata } from "../types/nowcast";

interface TimelineControlsProps {
  frames: FrameMetadata[];
  activeIndex: number;
  isPlaying: boolean;
  opacity: number;
  onTogglePlay: () => void;
  onStep: (delta: number) => void;
  onSelectIndex: (idx: number) => void;
  onOpacityChange: (val: number) => void;
}

function formatTimestamps(iso: string | undefined): { utc: string; slt: string } {
  if (!iso) return { utc: "—", slt: "—" };
  try {
    const d = new Date(iso);
    const utc = d.toISOString().slice(11, 16) + " UTC";
    const slt =
      d.toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "Asia/Colombo",
      }) + " SLT";
    return { utc, slt };
  } catch {
    return { utc: iso, slt: "—" };
  }
}

export function TimelineControls({
  frames,
  activeIndex,
  isPlaying,
  opacity,
  onTogglePlay,
  onStep,
  onSelectIndex,
  onOpacityChange,
}: TimelineControlsProps) {
  const current = frames[activeIndex];
  const { utc, slt } = formatTimestamps(current?.timestamp);

  return (
    <div className="pointer-events-auto flex flex-wrap items-center gap-3 rounded-xl border border-slate-700/80 bg-slate-950/92 px-3.5 py-2.5 shadow-2xl backdrop-blur-md">
      {/* Playback Controls */}
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onStep(-1)}
          disabled={frames.length < 2}
          aria-label="Previous satellite frame"
          className="rounded-md border border-slate-700 bg-slate-900 p-1.5 text-slate-200 transition hover:border-cyan-400/50 hover:text-cyan-300 disabled:opacity-40"
        >
          <SkipBack className="h-3.5 w-3.5" />
        </button>

        <button
          type="button"
          onClick={onTogglePlay}
          disabled={frames.length < 2}
          aria-label={isPlaying ? "Pause satellite loop" : "Play satellite loop"}
          className="inline-flex items-center gap-1 rounded-md border border-cyan-500/50 bg-cyan-500/20 px-2.5 py-1.5 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/30 disabled:opacity-40"
        >
          {isPlaying ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
          <span>{isPlaying ? "Pause" : "Play"}</span>
        </button>

        <button
          type="button"
          onClick={() => onStep(1)}
          disabled={frames.length < 2}
          aria-label="Next satellite frame"
          className="rounded-md border border-slate-700 bg-slate-900 p-1.5 text-slate-200 transition hover:border-cyan-400/50 hover:text-cyan-300 disabled:opacity-40"
        >
          <SkipForward className="h-3.5 w-3.5" />
        </button>
      </div>

      {/* Frame Scrubber */}
      <div className="flex min-w-[180px] flex-1 items-center gap-2.5">
        <input
          type="range"
          min={0}
          max={Math.max(0, frames.length - 1)}
          value={Math.min(activeIndex, Math.max(0, frames.length - 1))}
          onChange={(e) => onSelectIndex(Number(e.target.value))}
          disabled={frames.length < 2}
          aria-label="Satellite frame timeline"
          className="w-full cursor-pointer"
        />
        <span className="shrink-0 font-mono text-[11px] text-slate-400">
          {frames.length > 0 ? `${activeIndex + 1}/${frames.length}` : "0/0"}
        </span>
      </div>

      {/* Timestamp Badge */}
      <div className="flex items-center gap-1.5 rounded-md border border-slate-800 bg-slate-900 px-2.5 py-1 font-mono text-[11px]">
        <span className="font-semibold text-cyan-300">{slt}</span>
        <span className="text-slate-600">|</span>
        <span className="text-slate-400">{utc}</span>
      </div>

      {/* IR Opacity Slider */}
      <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
        <span>IR</span>
        <input
          type="range"
          min={15}
          max={100}
          value={Math.round(opacity * 100)}
          onChange={(e) => onOpacityChange(Number(e.target.value) / 100)}
          aria-label="Satellite overlay opacity"
          className="w-16 cursor-pointer"
        />
      </label>
    </div>
  );
}
