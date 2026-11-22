import { useState } from "react";
import { Bell, ChevronDown, ChevronRight, Send, CheckCircle2, AlertCircle } from "lucide-react";
import type {
  AlertHistoryItem,
  AlertStatusResponse,
  TelegramTestResponse,
} from "../types/nowcast";

interface TelegramAlertsPanelProps {
  alertStatus: AlertStatusResponse | null;
  history: AlertHistoryItem[];
  onSendTest: () => Promise<TelegramTestResponse>;
  onSendSummary: () => Promise<TelegramTestResponse>;
}

const LEVEL_BADGE: Record<string, string> = {
  CRITICAL: "border-rose-500/50 bg-rose-500/20 text-rose-200",
  SEVERE: "border-rose-500/50 bg-rose-500/20 text-rose-200",
  WARNING: "border-orange-500/50 bg-orange-500/20 text-orange-200",
  HIGH: "border-orange-500/50 bg-orange-500/20 text-orange-200",
  TEST: "border-cyan-500/40 bg-cyan-500/15 text-cyan-200",
  INFO: "border-sky-500/40 bg-sky-500/15 text-sky-200",
};

function formatSltShort(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    return (
      new Date(iso).toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
        timeZone: "Asia/Colombo",
      }) + " SLT"
    );
  } catch {
    return iso;
  }
}

export function TelegramAlertsPanel({
  alertStatus,
  history,
  onSendTest,
  onSendSummary,
}: TelegramAlertsPanelProps) {
  const [sendingAction, setSendingAction] = useState<"test" | "summary" | null>(null);
  const [expanded, setExpanded] = useState(false);
  const [toast, setToast] = useState<{ ok: boolean; text: string } | null>(null);

  const isConnected = Boolean(alertStatus?.telegram_ready);

  const handleDispatch = async (kind: "test" | "summary") => {
    setSendingAction(kind);
    setToast(null);
    try {
      const res = kind === "test" ? await onSendTest() : await onSendSummary();
      setToast({ ok: res.ok, text: res.detail });
    } catch (err) {
      setToast({
        ok: false,
        text: err instanceof Error ? err.message : "Failed to dispatch Telegram message",
      });
    } finally {
      setSendingAction(null);
      window.setTimeout(() => setToast(null), 6500);
    }
  };

  return (
    <section
      aria-label="Telegram Bot Alerts"
      className="rounded-lg border border-slate-800 bg-slate-900/75 p-3"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <Bell className="h-3.5 w-3.5 shrink-0 text-cyan-400" />
          <h2 className="truncate text-xs font-semibold uppercase tracking-wider text-slate-200">
            Telegram Alerts
          </h2>
          <span
            className={`rounded-full border px-1.5 py-0.5 font-mono text-[9px] font-bold uppercase ${
              isConnected
                ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-300"
                : "border-slate-700 bg-slate-950 text-slate-400"
            }`}
          >
            {isConnected ? "Connected" : "Not Configured"}
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            onClick={() => handleDispatch("summary")}
            disabled={sendingAction !== null}
            className="inline-flex items-center gap-1 rounded-md border border-sky-500/40 bg-sky-500/15 px-2 py-1 text-[11px] font-medium text-sky-200 transition hover:bg-sky-500/25 disabled:opacity-50"
          >
            {sendingAction === "summary" ? "Sending…" : "Summary"}
          </button>
          <button
            type="button"
            onClick={() => handleDispatch("test")}
            disabled={sendingAction !== null}
            className="inline-flex items-center gap-1 rounded-md border border-cyan-500/40 bg-cyan-500/15 px-2 py-1 text-[11px] font-medium text-cyan-200 transition hover:bg-cyan-500/25 disabled:opacity-50"
          >
            <Send className={`h-3 w-3 ${sendingAction === "test" ? "animate-pulse" : ""}`} />
            {sendingAction === "test" ? "Sending…" : "Test"}
          </button>
        </div>
      </div>

      {/* Cooldown, Hourly Summary & last dispatched summary */}
      <div className="mt-1.5 flex flex-wrap items-center justify-between gap-1 font-mono text-[10px] text-slate-400">
        <span>
          Hourly: <strong className="text-slate-300">{alertStatus?.hourly_summary_enabled ? `${alertStatus.hourly_summary_interval_min}m` : "Off"}</strong>
          {" • "}Cooldown: <strong className="text-slate-300">{alertStatus?.cooldown_min ?? 30}m</strong>
        </span>
        <span>Last: {formatSltShort(alertStatus?.last_dispatched_at)}</span>
      </div>

      {/* Toast Feedback */}
      {toast ? (
        <div
          role="status"
          className={`mt-2 flex items-start gap-1.5 rounded-md border px-2.5 py-1.5 text-[11px] ${
            toast.ok
              ? "border-emerald-500/40 bg-emerald-500/15 text-emerald-200"
              : "border-amber-500/40 bg-amber-500/15 text-amber-200"
          }`}
        >
          {toast.ok ? (
            <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400" />
          ) : (
            <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-400" />
          )}
          <span className="leading-snug">{toast.text}</span>
        </div>
      ) : null}

      {/* Collapsible Recent Alerts Log (last 10) */}
      <div className="mt-2 border-t border-slate-800/80 pt-2">
        <button
          type="button"
          onClick={() => setExpanded((e) => !e)}
          aria-expanded={expanded}
          className="flex w-full items-center justify-between text-left text-[11px] font-medium text-slate-300 hover:text-cyan-200"
        >
          <span className="flex items-center gap-1">
            {expanded ? (
              <ChevronDown className="h-3.5 w-3.5 text-slate-400" />
            ) : (
              <ChevronRight className="h-3.5 w-3.5 text-slate-400" />
            )}
            Recent Alerts ({history.length})
          </span>
          <span className="font-mono text-[10px] text-slate-500">
            {expanded ? "Hide" : "Show log"}
          </span>
        </button>

        {expanded ? (
          <div className="mt-2 max-h-44 space-y-1.5 overflow-y-auto pr-0.5">
            {history.length === 0 ? (
              <p className="rounded border border-slate-800/70 bg-slate-950/60 px-2.5 py-2 text-[11px] text-slate-500">
                No alerts dispatched yet. Severe storm or river flood warnings will appear here.
              </p>
            ) : (
              history.slice(0, 10).map((item) => {
                const isSent = item.status.startsWith("sent");
                return (
                  <div
                    key={item.id}
                    className="rounded border border-slate-800 bg-slate-950/70 px-2 py-1.5 text-[10px]"
                  >
                    <div className="flex items-center justify-between gap-1.5">
                      <span className="truncate font-semibold text-slate-200">{item.zone_name}</span>
                      <div className="flex shrink-0 items-center gap-1">
                        <span
                          className={`rounded border px-1 py-0.2 font-mono text-[9px] font-bold ${
                            LEVEL_BADGE[item.level] ?? "border-slate-700 bg-slate-800 text-slate-300"
                          }`}
                        >
                          {item.level}
                        </span>
                        <span
                          className={`font-mono text-[9px] ${
                            isSent ? "text-emerald-400" : "text-rose-400"
                          }`}
                        >
                          {isSent ? "SENT" : "FAIL"}
                        </span>
                      </div>
                    </div>
                    {item.caption_preview ? (
                      <p className="mt-0.5 line-clamp-2 text-[10px] leading-snug text-slate-400">
                        {item.caption_preview}
                      </p>
                    ) : null}
                    <div className="mt-0.5 flex items-center justify-between font-mono text-[9px] text-slate-500">
                      <span>{item.channel}</span>
                      <span>{formatSltShort(item.sent_at)}</span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        ) : null}
      </div>
    </section>
  );
}

