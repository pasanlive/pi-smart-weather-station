import { useState } from "react";
import { AlertTriangle, Send, Eye, Waves, MapPin, CheckCircle2, AlertCircle } from "lucide-react";
import type {
  DistrictThreat,
  HourlyDigestTestResponse,
  SeaZoneHazard,
} from "../types/nowcast";

interface HourlyDigestPanelProps {
  districts: DistrictThreat[];
  seaZones: SeaZoneHazard[];
  thresholdPct: number;
  onTriggerDigest: (dispatch: boolean) => Promise<HourlyDigestTestResponse>;
  onSelectDistrict?: (districtId: string) => void;
}

export function HourlyDigestPanel({
  districts,
  seaZones,
  thresholdPct,
  onTriggerDigest,
  onSelectDistrict,
}: HourlyDigestPanelProps) {
  const [busy, setBusy] = useState<"preview" | "send" | null>(null);
  const [previewCaption, setPreviewCaption] = useState<string | null>(null);
  const [toast, setToast] = useState<{ ok: boolean; text: string } | null>(null);

  const qualifyingDistricts = districts
    .filter((d) => d.hazards && d.hazards.max_hazard_prob > thresholdPct)
    .sort((a, b) => (b.hazards?.max_hazard_prob ?? 0) - (a.hazards?.max_hazard_prob ?? 0));

  const qualifyingSea = seaZones
    .filter((s) => s.hazards.max_hazard_prob > thresholdPct)
    .sort((a, b) => b.hazards.max_hazard_prob - a.hazards.max_hazard_prob);

  const totalQualifying = qualifyingDistricts.length + qualifyingSea.length;

  const runAction = async (dispatch: boolean) => {
    setBusy(dispatch ? "send" : "preview");
    setToast(null);
    try {
      const res = await onTriggerDigest(dispatch);
      setToast({ ok: res.ok, text: res.detail });
      setPreviewCaption(res.caption_preview || null);
    } catch (err) {
      setToast({
        ok: false,
        text: err instanceof Error ? err.message : "Failed to run hourly digest check",
      });
    } finally {
      setBusy(null);
      window.setTimeout(() => setToast(null), 6500);
    }
  };

  return (
    <section
      aria-label="Hourly Hazard Digest Preview"
      className="rounded-lg border border-slate-800 bg-slate-900/75 p-3"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <AlertTriangle
            className={`h-3.5 w-3.5 shrink-0 ${
              totalQualifying > 0 ? "text-orange-400" : "text-emerald-400"
            }`}
          />
          <h2 className="truncate text-xs font-semibold uppercase tracking-wider text-slate-200">
            Hourly Digest (&gt;{thresholdPct}%)
          </h2>
          <span
            className={`rounded-full border px-1.5 py-0.5 font-mono text-[9px] font-bold ${
              totalQualifying > 0
                ? "border-orange-500/50 bg-orange-500/20 text-orange-200"
                : "border-emerald-500/35 bg-emerald-500/15 text-emerald-300"
            }`}
          >
            {totalQualifying} Active
          </span>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            onClick={() => runAction(false)}
            disabled={busy !== null}
            className="inline-flex items-center gap-1 rounded-md border border-slate-700 bg-slate-800/80 px-2 py-1 text-[10px] font-medium text-slate-200 transition hover:border-cyan-500/40 hover:text-cyan-200 disabled:opacity-50"
          >
            <Eye className="h-3 w-3" />
            {busy === "preview" ? "…" : "Preview"}
          </button>
          <button
            type="button"
            onClick={() => runAction(true)}
            disabled={busy !== null}
            className="inline-flex items-center gap-1 rounded-md border border-orange-500/45 bg-orange-500/15 px-2 py-1 text-[10px] font-medium text-orange-200 transition hover:bg-orange-500/25 disabled:opacity-50"
          >
            <Send className="h-3 w-3" />
            {busy === "send" ? "Sending…" : "Send"}
          </button>
        </div>
      </div>

      {totalQualifying === 0 ? (
        <p className="mt-1.5 text-[11px] text-slate-400">
          All 25 districts &amp; 8 sea zones are &le;{thresholdPct}% probability. Next top-of-hour digest will skip silently.
        </p>
      ) : (
        <div className="mt-2 space-y-1.5">
          {qualifyingDistricts.length > 0 ? (
            <div>
              <div className="mb-1 flex items-center gap-1 font-mono text-[10px] font-semibold text-slate-400">
                <MapPin className="h-3 w-3 text-amber-400" />
                DISTRICTS ({qualifyingDistricts.length})
              </div>
              <div className="flex flex-wrap gap-1">
                {qualifyingDistricts.slice(0, 8).map((d) => {
                  const p = d.hazards?.max_hazard_prob ?? 0;
                  return (
                    <button
                      key={d.district_id}
                      type="button"
                      onClick={() => onSelectDistrict?.(d.district_id)}
                      className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-semibold transition ${
                        p > 75
                          ? "border-rose-500/50 bg-rose-500/20 text-rose-200"
                          : "border-orange-500/45 bg-orange-500/15 text-orange-200"
                      }`}
                    >
                      {d.name} {p}%
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}

          {qualifyingSea.length > 0 ? (
            <div>
              <div className="mb-1 flex items-center gap-1 font-mono text-[10px] font-semibold text-slate-400">
                <Waves className="h-3 w-3 text-sky-400" />
                SEA AREAS ({qualifyingSea.length})
              </div>
              <div className="flex flex-wrap gap-1">
                {qualifyingSea.map((s) => {
                  const p = s.hazards.max_hazard_prob;
                  return (
                    <span
                      key={s.zone_id}
                      className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-semibold ${
                        p > 75
                          ? "border-rose-500/50 bg-rose-500/20 text-rose-200"
                          : "border-orange-500/45 bg-orange-500/15 text-orange-200"
                      }`}
                    >
                      {s.name} {p}%
                      {s.hazards.wave_height_m !== null ? ` • ${s.hazards.wave_height_m.toFixed(1)}m` : ""}
                    </span>
                  );
                })}
              </div>
            </div>
          ) : null}
        </div>
      )}

      {toast ? (
        <div
          role="status"
          className={`mt-2 flex items-start gap-1.5 rounded-md border px-2 py-1.5 text-[11px] ${
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

      {previewCaption ? (
        <pre className="mt-2 max-h-36 overflow-auto whitespace-pre-wrap rounded border border-slate-800 bg-slate-950/80 p-2 font-mono text-[10px] leading-relaxed text-slate-300">
          {previewCaption.replace(/<[^>]+>/g, "")}
        </pre>
      ) : null}
    </section>
  );
}
