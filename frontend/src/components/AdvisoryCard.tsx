import { Sparkles, FileText, Cpu } from "lucide-react";

interface AdvisoryCardProps {
  advisoryText: string;
  advisorySource: string;
  advisoryModel?: string | null;
  advisoryLatencyMs?: number;
  stormCount: number;
}

function formatLatency(ms?: number): string {
  if (!ms || ms <= 0) return "";
  if (ms >= 1000) return ` • ${(ms / 1000).toFixed(1)}s`;
  return ` • ${Math.round(ms)}ms`;
}

export function AdvisoryCard({
  advisoryText,
  advisorySource,
  advisoryModel,
  advisoryLatencyMs,
  stormCount,
}: AdvisoryCardProps) {
  const isOllama = advisorySource === "ollama" || advisorySource.startsWith("ollama:");
  const isFallback = advisorySource === "template_fallback";
  const modelName =
    advisoryModel ||
    (advisorySource.startsWith("ollama:") ? advisorySource.slice("ollama:".length) : "qwen2.5:3b");

  const pillLabel = isOllama
    ? `Local LLM • ${modelName}${formatLatency(advisoryLatencyMs)}`
    : isFallback
    ? "Template (Fallback)"
    : "Template";

  const pillClass = isOllama
    ? "border-violet-500/40 bg-violet-500/15 text-violet-200"
    : isFallback
    ? "border-amber-500/35 bg-amber-500/10 text-amber-200"
    : "border-slate-700 bg-slate-800/80 text-slate-300";

  return (
    <section
      aria-label="Weather & Flood Advisory"
      className="rounded-lg border border-slate-800 bg-slate-900/75 p-3"
    >
      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-1.5">
        <h2 className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wider text-cyan-300">
          {isOllama ? <Sparkles className="h-3.5 w-3.5 text-violet-300" /> : <FileText className="h-3.5 w-3.5" />}
          Threat Advisory
        </h2>
        <div className="flex items-center gap-1.5">
          <span className="font-mono text-[10px] text-slate-500">{stormCount} cells</span>
          <span
            className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-[10px] font-medium ${pillClass}`}
          >
            {isOllama ? <Cpu className="h-2.5 w-2.5" aria-hidden="true" /> : null}
            {pillLabel}
          </span>
        </div>
      </div>
      <p className="text-xs leading-relaxed text-slate-200">
        {advisoryText || "Initializing Himawari-9 satellite loop and Sri Lanka threat telemetry..."}
      </p>
    </section>
  );
}
