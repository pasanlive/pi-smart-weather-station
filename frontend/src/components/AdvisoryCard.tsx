import { Sparkles, FileText } from "lucide-react";

interface AdvisoryCardProps {
  advisoryText: string;
  advisorySource: string;
  stormCount: number;
}

export function AdvisoryCard({ advisoryText, advisorySource, stormCount }: AdvisoryCardProps) {
  const isOllama = advisorySource.startsWith("ollama");

  return (
    <section
      aria-label="Weather & Flood Advisory"
      className="rounded-lg border border-slate-800 bg-slate-900/75 p-3"
    >
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <h2 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-cyan-300">
          {isOllama ? <Sparkles className="h-3.5 w-3.5" /> : <FileText className="h-3.5 w-3.5" />}
          Operational Advisory
        </h2>
        <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
          {stormCount} cells • {isOllama ? advisorySource : "deterministic"}
        </span>
      </div>
      <p className="text-xs leading-relaxed text-slate-200">
        {advisoryText || "Initializing Himawari-9 satellite loop and Sri Lanka threat telemetry..."}
      </p>
    </section>
  );
}
