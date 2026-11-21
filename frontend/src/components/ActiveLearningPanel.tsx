import { useCallback, useEffect, useId, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  ChevronLeft,
  ChevronRight,
  Database as DatabaseIcon,
  Download,
  Edit3,
  Eye,
  EyeOff,
  HardDrive,
  RefreshCw,
  Waves,
  X,
} from "lucide-react";
import { api } from "../api/client";
import type {
  ActiveLearningBasinSnapshot,
  ActiveLearningPolygon,
  ActiveLearningSample,
  ActiveLearningStats,
  ActiveLearningStatus,
} from "../types/nowcast";

interface ActiveLearningPanelProps {
  stats: ActiveLearningStats | null;
  onStatsChange: (stats: ActiveLearningStats) => void;
  isActive: boolean;
}

const CLASS_STYLE: Record<
  number,
  { name: string; stroke: string; fill: string; badgeCls: string }
> = {
  0: {
    name: "deep_convective",
    stroke: "#f43f5e",
    fill: "rgba(244, 63, 94, 0.24)",
    badgeCls: "border-rose-500/50 bg-rose-500/15 text-rose-200",
  },
  1: {
    name: "cyclonic_band",
    stroke: "#f59e0b",
    fill: "rgba(245, 158, 11, 0.22)",
    badgeCls: "border-amber-500/50 bg-amber-500/15 text-amber-200",
  },
  2: {
    name: "monsoon_shield",
    stroke: "#22d3ee",
    fill: "rgba(34, 211, 238, 0.20)",
    badgeCls: "border-cyan-500/50 bg-cyan-500/15 text-cyan-200",
  },
};

const TRIGGER_BADGE: Record<string, string> = {
  low_conf: "border-amber-500/50 bg-amber-500/15 text-amber-200",
  sensor_mismatch: "border-rose-500/50 bg-rose-500/15 text-rose-200",
  boundary_artifact: "border-violet-500/50 bg-violet-500/15 text-violet-200",
};

function parseClientYoloSeg(txt: string): ActiveLearningPolygon[] {
  const out: ActiveLearningPolygon[] = [];
  for (const raw of txt.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const parts = line.split(/\s+/);
    if (parts.length < 7 || (parts.length - 1) % 2 !== 0) continue;
    const cid = Number.parseInt(parts[0], 10);
    if (!Number.isFinite(cid)) continue;
    const nums = parts.slice(1).map((v) => Number.parseFloat(v));
    if (nums.some((n) => !Number.isFinite(n) || n < 0 || n > 1)) continue;
    const points: [number, number][] = [];
    for (let i = 0; i < nums.length; i += 2) {
      points.push([nums[i], nums[i + 1]]);
    }
    if (points.length >= 3) {
      out.push({
        class_id: cid,
        label: CLASS_STYLE[cid]?.name ?? `class_${cid}`,
        points,
      });
    }
  }
  return out;
}

function formatUtcAndSlt(iso: string | null | undefined): string {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    const slt = d.toLocaleTimeString("en-GB", {
      hour: "2-digit",
      minute: "2-digit",
      timeZone: "Asia/Colombo",
    });
    return `${iso.slice(0, 16).replace("T", " ")} UTC (${slt} SLT)`;
  } catch {
    return iso;
  }
}

export function ActiveLearningPanel({
  stats,
  onStatsChange,
  isActive,
}: ActiveLearningPanelProps) {
  const [statusFilter, setStatusFilter] = useState<ActiveLearningStatus>("pending");
  const [samples, setSamples] = useState<ActiveLearningSample[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showEditor, setShowEditor] = useState(false);
  const [editedLabelTxt, setEditedLabelTxt] = useState("");
  const [labelDirty, setLabelDirty] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string>("");
  const [errorMessage, setErrorMessage] = useState<string>("");

  const labelEditorId = useId();
  const storageProgressId = useId();

  const currentSample = samples[currentIndex] ?? null;

  const loadQueue = useCallback(
    async (filter: ActiveLearningStatus) => {
      setLoading(true);
      setErrorMessage("");
      try {
        const res = await api.getActiveLearningQueue(filter, 50);
        setSamples(res.samples);
        onStatsChange(res.stats);
        setCurrentIndex((prev) => {
          if (res.samples.length === 0) return 0;
          return Math.min(prev, res.samples.length - 1);
        });
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : "Failed to load review queue");
      } finally {
        setLoading(false);
      }
    },
    [onStatsChange]
  );

  useEffect(() => {
    if (isActive) {
      void loadQueue(statusFilter);
    }
  }, [isActive, statusFilter, loadQueue]);

  // Sync editor buffer whenever active sample changes
  useEffect(() => {
    setEditedLabelTxt(currentSample?.label_txt ?? "");
    setLabelDirty(false);
  }, [currentSample?.id, currentSample?.label_txt]);

  const renderedPolygons = useMemo(() => {
    if (!currentSample) return [];
    if (labelDirty) {
      return parseClientYoloSeg(editedLabelTxt);
    }
    return currentSample.polygons;
  }, [currentSample, labelDirty, editedLabelTxt]);

  const handleDecision = useCallback(
    async (decision: "approved" | "rejected") => {
      if (!currentSample || submitting) return;
      setSubmitting(true);
      setErrorMessage("");
      try {
        const payload: { status: "approved" | "rejected"; updated_label_txt?: string } = {
          status: decision,
        };
        if (labelDirty) {
          payload.updated_label_txt = editedLabelTxt;
        }
        const res = await api.submitActiveLearningDecision(currentSample.id, payload);
        onStatsChange(res.stats);
        setStatusMessage(
          `${decision === "approved" ? "Approved" : "Rejected"} ${currentSample.id}${
            labelDirty ? " (label updated)" : ""
          }.`
        );

        setSamples((prev) => {
          if (statusFilter === decision) {
            return prev.map((s) => (s.id === currentSample.id ? res.sample : s));
          }
          const next = prev.filter((s) => s.id !== currentSample.id);
          setCurrentIndex((idx) => (next.length === 0 ? 0 : Math.min(idx, next.length - 1)));
          return next;
        });
        setLabelDirty(false);
      } catch (err) {
        setErrorMessage(err instanceof Error ? err.message : "Failed to save decision");
      } finally {
        setSubmitting(false);
      }
    },
    [currentSample, submitting, labelDirty, editedLabelTxt, onStatsChange, statusFilter]
  );

  const stepSample = useCallback(
    (delta: number) => {
      if (samples.length < 2) return;
      setCurrentIndex((idx) => (idx + delta + samples.length) % samples.length);
    },
    [samples.length]
  );

  const handleExportZip = useCallback(async () => {
    if (!stats || stats.approved === 0 || exporting) return;
    setExporting(true);
    setErrorMessage("");
    try {
      const resp = await fetch(api.getActiveLearningExportUrl());
      if (!resp.ok) {
        let msg = `Export failed (HTTP ${resp.status})`;
        try {
          const data = (await resp.json()) as { detail?: string };
          if (data?.detail) msg = data.detail;
        } catch {
          // ignore
        }
        throw new Error(msg);
      }
      const blob = await resp.blob();
      const cd = resp.headers.get("content-disposition") || "";
      const match = cd.match(/filename="?([^"]+)"?/i);
      const filename =
        match?.[1] || `active_learning_batch_${new Date().toISOString().slice(0, 10).replace(/-/g, "")}.zip`;

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      setStatusMessage(`Exported ${stats.approved} approved sample(s) to ${filename}.`);
    } catch (err) {
      setErrorMessage(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExporting(false);
    }
  }, [stats, exporting]);

  // Fast Triage Keyboard Shortcuts (A Approve, R Reject, ArrowLeft/ArrowRight Navigate, O Overlay)
  useEffect(() => {
    if (!isActive) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "TEXTAREA" ||
          target.tagName === "INPUT" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      if (e.key === "a" || e.key === "A") {
        e.preventDefault();
        void handleDecision("approved");
      } else if (e.key === "r" || e.key === "R") {
        e.preventDefault();
        void handleDecision("rejected");
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        stepSample(-1);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        stepSample(1);
      } else if (e.key === "o" || e.key === "O") {
        e.preventDefault();
        setShowOverlay((v) => !v);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isActive, handleDecision, stepSample]);

  const storagePct = stats?.storage_pct ?? 0;
  const progressClass =
    storagePct >= 85
      ? "al-storage-progress is-critical"
      : storagePct >= 65
        ? "al-storage-progress is-warn"
        : "al-storage-progress";

  const basinsList: ActiveLearningBasinSnapshot[] = useMemo(() => {
    const snap = currentSample?.meta?.basins_snapshot;
    if (!snap) return [];
    return Object.values(snap);
  }, [currentSample]);

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-slate-950 text-slate-100">
      {/* Queue Header */}
      <div className="border-b border-slate-800 bg-slate-900/75 px-4 py-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-2">
              <DatabaseIcon className="h-4 w-4 text-cyan-400" aria-hidden="true" />
              <h2 className="text-sm font-bold tracking-wide text-slate-100 uppercase">
                Active Learning QA Queue
              </h2>
              <span
                data-testid="al-pending-badge"
                className="inline-flex items-center rounded-full border border-amber-500/40 bg-amber-500/15 px-2 py-0.5 font-mono text-[11px] font-semibold text-amber-200"
              >
                {stats?.pending ?? 0} pending
              </span>
            </div>

            {/* Queue Status Filter Pills */}
            <div className="ml-1 inline-flex rounded-md border border-slate-800 bg-slate-950/80 p-0.5">
              {(["pending", "approved", "rejected"] as const).map((st) => {
                const count = stats ? stats[st] : 0;
                const selected = statusFilter === st;
                return (
                  <button
                    key={st}
                    type="button"
                    onClick={() => setStatusFilter(st)}
                    aria-pressed={selected}
                    className={`min-h-[28px] rounded px-2.5 py-1 text-xs font-medium capitalize transition ${
                      selected
                        ? "bg-cyan-500/20 text-cyan-200 ring-1 ring-cyan-500/40"
                        : "text-slate-400 hover:text-slate-200"
                    }`}
                  >
                    {st} <span className="ml-0.5 font-mono text-[11px] opacity-80">({count})</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Pi 5 SD-Card Storage Usage Bar */}
            <div className="w-52">
              <div className="mb-1 flex items-center justify-between text-[11px] text-slate-300">
                <label htmlFor={storageProgressId} className="inline-flex items-center gap-1 font-medium">
                  <HardDrive className="h-3 w-3 text-cyan-400" aria-hidden="true" />
                  Storage
                </label>
                <span className="font-mono text-[10px] text-slate-400">
                  {stats ? `${stats.storage_used_mb.toFixed(1)} / ${stats.max_storage_mb} MB` : "—"} •{" "}
                  {stats ? `${stats.captures_today}/${stats.max_daily_captures} today` : ""}
                </span>
              </div>
              <progress
                id={storageProgressId}
                className={progressClass}
                value={stats?.storage_used_mb ?? 0}
                max={stats?.max_storage_mb || 2048}
              />
            </div>

            <button
              type="button"
              onClick={() => void loadQueue(statusFilter)}
              disabled={loading}
              aria-label="Reload active learning queue"
              className="inline-flex min-h-[32px] items-center gap-1.5 rounded-md border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-xs font-medium text-slate-200 transition hover:bg-slate-800 disabled:opacity-50"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} aria-hidden="true" />
              Sync
            </button>

            <button
              type="button"
              data-testid="al-export-btn"
              onClick={() => void handleExportZip()}
              disabled={!stats || stats.approved === 0 || exporting}
              className="inline-flex min-h-[32px] items-center gap-1.5 rounded-md border border-emerald-500/50 bg-emerald-500/15 px-3 py-1.5 text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/25 disabled:cursor-not-allowed disabled:opacity-45"
            >
              <Download className="h-3.5 w-3.5" aria-hidden="true" />
              {exporting ? "Bundling…" : `Export Approved ZIP (${stats?.approved ?? 0})`}
            </button>
          </div>
        </div>

        {/* Centralized polite live status & error alerts */}
        <div className="mt-2 flex items-center justify-between gap-2 text-xs">
          <div role="status" aria-live="polite" className="min-h-[1rem] text-emerald-300">
            {statusMessage}
          </div>
          {errorMessage ? (
            <div role="alert" className="text-rose-300">
              {errorMessage}
            </div>
          ) : null}
        </div>
      </div>

      {/* Main Visual QA Card Area */}
      <div className="flex min-h-0 flex-1 overflow-y-auto p-4">
        {loading && samples.length === 0 ? (
          <div className="flex flex-1 items-center justify-center text-sm text-slate-400">
            Loading active-learning samples…
          </div>
        ) : !currentSample ? (
          <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-900/35 p-8 text-center">
            <DatabaseIcon className="mb-2 h-8 w-8 text-slate-600" aria-hidden="true" />
            <p className="text-sm font-medium text-slate-300">
              No <span className="capitalize">{statusFilter}</span> samples in queue
            </p>
            <p className="mt-1 max-w-md text-xs text-slate-400">
              Uncertain storm detections (0.25–0.55 confidence) and cross-sensor mismatches (heavy Open-Meteo rainfall or rising Irrigation Dept gauges without deep convective detection) are automatically queued every 10-minute cycle.
            </p>
          </div>
        ) : (
          <div className="mx-auto grid w-full max-w-6xl grid-cols-1 gap-4 xl:grid-cols-[minmax(360px,540px)_1fr]">
            {/* Left Column: 640x640 Himawari-9 IR Frame + Toggleable SVG Polygon Overlay */}
            <section className="flex flex-col rounded-xl border border-slate-800 bg-slate-900/70 p-3">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-semibold text-slate-200">
                    Sample {currentIndex + 1} / {samples.length}
                  </span>
                  <span
                    data-testid="al-trigger-badge"
                    className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[11px] font-semibold ${
                      TRIGGER_BADGE[currentSample.trigger_type] ??
                      "border-slate-700 bg-slate-800 text-slate-200"
                    }`}
                  >
                    <AlertTriangle className="h-3 w-3" aria-hidden="true" />
                    {currentSample.trigger_detail}
                  </span>
                </div>

                <button
                  type="button"
                  data-testid="al-overlay-toggle"
                  onClick={() => setShowOverlay((v) => !v)}
                  aria-pressed={showOverlay}
                  className="inline-flex min-h-[28px] items-center gap-1.5 rounded-md border border-slate-700 bg-slate-950 px-2.5 py-1 text-xs font-medium text-slate-200 transition hover:bg-slate-800"
                >
                  {showOverlay ? (
                    <Eye className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                  ) : (
                    <EyeOff className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
                  )}
                  Polygons {showOverlay ? "ON" : "OFF"}
                  <kbd className="rounded bg-slate-800 px-1 font-mono text-[10px] text-slate-400">O</kbd>
                </button>
              </div>

              {/* Aspect-square 640x640 frame + aligned SVG overlay */}
              <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-slate-800 bg-black">
                <img
                  src={currentSample.frame_url}
                  alt={`Raw 640 by 640 Himawari-9 Band 13 infrared frame captured at ${currentSample.frame_timestamp}`}
                  className="h-full w-full object-contain select-none"
                />

                {showOverlay ? (
                  <svg
                    data-testid="al-polygon-svg"
                    viewBox="0 0 640 640"
                    role="img"
                    aria-label={`${renderedPolygons.length} pseudo-labeled storm polygon overlays`}
                    className="pointer-events-none absolute inset-0 h-full w-full"
                  >
                    {renderedPolygons.map((poly, idx) => {
                      const style = CLASS_STYLE[poly.class_id] ?? CLASS_STYLE[2];
                      const ptsStr = poly.points
                        .map(([x, y]) => `${(x * 640).toFixed(1)},${(y * 640).toFixed(1)}`)
                        .join(" ");
                      const anchor = poly.points[0] ?? [0.05, 0.05];
                      const lx = Math.min(540, Math.max(8, anchor[0] * 640));
                      const ly = Math.min(625, Math.max(18, anchor[1] * 640 - 6));
                      return (
                        <g key={`${poly.class_id}-${idx}`}>
                          <polygon
                            points={ptsStr}
                            fill={style.fill}
                            stroke={style.stroke}
                            strokeWidth="2.5"
                            strokeLinejoin="round"
                          />
                          <rect
                            x={lx - 3}
                            y={ly - 13}
                            width={Math.max(96, poly.label.length * 7.2)}
                            height={17}
                            rx={3}
                            fill="rgba(2, 6, 23, 0.85)"
                            stroke={style.stroke}
                            strokeWidth="1"
                          />
                          <text
                            x={lx + 2}
                            y={ly - 1}
                            fill="#f8fafc"
                            fontSize="11"
                            fontFamily="monospace"
                          >
                            [{poly.class_id}] {poly.label}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                ) : null}
              </div>

              {/* Class Legend */}
              <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 text-[11px]">
                <div className="flex flex-wrap items-center gap-1.5">
                  {Object.entries(CLASS_STYLE).map(([cid, spec]) => (
                    <span
                      key={cid}
                      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 font-mono ${spec.badgeCls}`}
                    >
                      [{cid}] {spec.name}
                    </span>
                  ))}
                </div>
                <span className="font-mono text-slate-400">
                  {renderedPolygons.length} polygon{renderedPolygons.length === 1 ? "" : "s"}
                </span>
              </div>

              {/* Fast Triage Controls */}
              <div className="mt-3 grid grid-cols-4 gap-2 border-t border-slate-800 pt-3">
                <button
                  type="button"
                  onClick={() => stepSample(-1)}
                  disabled={samples.length < 2}
                  className="inline-flex min-h-[38px] items-center justify-center gap-1 rounded-md border border-slate-700 bg-slate-950 px-2.5 text-xs font-medium text-slate-200 transition hover:bg-slate-800 disabled:opacity-40"
                >
                  <ChevronLeft className="h-4 w-4" aria-hidden="true" />
                  Prev
                  <kbd className="rounded bg-slate-800 px-1 font-mono text-[10px] text-slate-400">←</kbd>
                </button>

                <button
                  type="button"
                  data-testid="al-approve-btn"
                  onClick={() => void handleDecision("approved")}
                  disabled={submitting}
                  className="inline-flex min-h-[38px] items-center justify-center gap-1.5 rounded-md border border-emerald-500/60 bg-emerald-500/20 px-3 text-xs font-semibold text-emerald-100 transition hover:bg-emerald-500/30 disabled:opacity-50"
                >
                  <Check className="h-4 w-4" aria-hidden="true" />
                  Approve
                  <kbd className="rounded bg-emerald-950/80 px-1.5 font-mono text-[10px] text-emerald-200">A</kbd>
                </button>

                <button
                  type="button"
                  data-testid="al-reject-btn"
                  onClick={() => void handleDecision("rejected")}
                  disabled={submitting}
                  className="inline-flex min-h-[38px] items-center justify-center gap-1.5 rounded-md border border-rose-500/60 bg-rose-500/20 px-3 text-xs font-semibold text-rose-100 transition hover:bg-rose-500/30 disabled:opacity-50"
                >
                  <X className="h-4 w-4" aria-hidden="true" />
                  Reject
                  <kbd className="rounded bg-rose-950/80 px-1.5 font-mono text-[10px] text-rose-200">R</kbd>
                </button>

                <button
                  type="button"
                  onClick={() => stepSample(1)}
                  disabled={samples.length < 2}
                  className="inline-flex min-h-[38px] items-center justify-center gap-1 rounded-md border border-slate-700 bg-slate-950 px-2.5 text-xs font-medium text-slate-200 transition hover:bg-slate-800 disabled:opacity-40"
                >
                  Next
                  <ChevronRight className="h-4 w-4" aria-hidden="true" />
                  <kbd className="rounded bg-slate-800 px-1 font-mono text-[10px] text-slate-400">→</kbd>
                </button>
              </div>
            </section>

            {/* Right Column: Sample Metadata, Hydrology Context & Label Editor */}
            <div className="flex flex-col gap-3">
              {/* Sample Telemetry Metadata Card */}
              <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
                <h3 className="text-xs font-bold tracking-wider text-slate-300 uppercase">
                  Capture Diagnostics
                </h3>
                <dl className="mt-2.5 grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                  <div>
                    <dt className="text-[11px] text-slate-400">Sample ID</dt>
                    <dd className="truncate font-mono text-slate-200">{currentSample.id}</dd>
                  </div>
                  <div>
                    <dt className="text-[11px] text-slate-400">Review Status</dt>
                    <dd className="font-mono uppercase text-cyan-300">{currentSample.status}</dd>
                  </div>
                  <div>
                    <dt className="text-[11px] text-slate-400">Frame Timestamp</dt>
                    <dd className="font-mono text-[11px] text-slate-200">
                      {formatUtcAndSlt(currentSample.frame_timestamp)}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-[11px] text-slate-400">Active Model</dt>
                    <dd className="truncate font-mono text-[11px] text-slate-200">
                      {currentSample.model_filename}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-[11px] text-slate-400">Confidence Band</dt>
                    <dd className="font-mono text-slate-200">
                      {currentSample.min_confidence !== null
                        ? `${currentSample.min_confidence.toFixed(2)} – ${(
                            currentSample.max_confidence ?? currentSample.min_confidence
                          ).toFixed(2)}`
                        : "None (0 detections)"}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-[11px] text-slate-400">Pseudo-Label Source</dt>
                    <dd className="font-mono text-slate-200">
                      {currentSample.meta.pseudo_label_source ?? "model_raw"}
                    </dd>
                  </div>
                </dl>
              </section>

              {/* Basin Rainfall & River Gauge Context */}
              <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
                <div className="flex items-center justify-between">
                  <h3 className="inline-flex items-center gap-1.5 text-xs font-bold tracking-wider text-slate-300 uppercase">
                    <Waves className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                    Basin Rainfall &amp; River Gauge Context
                  </h3>
                  <span className="text-[11px] text-slate-400">
                    Open-Meteo + Irrigation Dept ArcGIS
                  </span>
                </div>

                {basinsList.length === 0 ? (
                  <p className="mt-2 text-xs text-slate-400">No basin hydrology snapshot recorded.</p>
                ) : (
                  <div className="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {basinsList.map((basin) => {
                      const heavyRain = basin.rainfall_1h_mm >= 20;
                      const risingGauge = basin.gauges.some(
                        (g) => g.trend === "RISING" && g.rate_of_rise_ft_hr >= 0.3
                      );
                      const highlighted = heavyRain || risingGauge;
                      return (
                        <div
                          key={basin.basin_id}
                          className={`rounded-lg border p-2.5 text-xs ${
                            highlighted
                              ? "border-rose-500/45 bg-rose-500/10"
                              : "border-slate-800 bg-slate-950/70"
                          }`}
                        >
                          <div className="flex items-center justify-between gap-1">
                            <span className="font-semibold text-slate-100">{basin.name}</span>
                            <span className="font-mono text-[10px] text-slate-400">
                              Deep Conv: {basin.deep_convective_pct.toFixed(0)}%
                            </span>
                          </div>
                          <div className="mt-1 flex items-center justify-between font-mono text-[11px]">
                            <span className={heavyRain ? "font-semibold text-rose-300" : "text-slate-300"}>
                              1h: {basin.rainfall_1h_mm.toFixed(1)} mm/hr
                            </span>
                            <span className="text-slate-400">
                              24h: {basin.rainfall_24h_mm.toFixed(1)} mm
                            </span>
                          </div>
                          {basin.gauges.length > 0 ? (
                            <ul className="mt-1.5 space-y-1 border-t border-slate-800/80 pt-1.5">
                              {basin.gauges.map((g) => (
                                <li
                                  key={g.station_id}
                                  className="flex items-center justify-between font-mono text-[10px]"
                                >
                                  <span className="text-slate-300">{g.station_name}</span>
                                  <span
                                    className={
                                      g.trend === "RISING" && g.rate_of_rise_ft_hr >= 0.3
                                        ? "font-semibold text-amber-300"
                                        : "text-slate-400"
                                    }
                                  >
                                    {g.water_level_ft.toFixed(1)} ft ({g.trend}{" "}
                                    {g.rate_of_rise_ft_hr >= 0 ? "+" : ""}
                                    {g.rate_of_rise_ft_hr.toFixed(2)} ft/hr)
                                  </span>
                                </li>
                              ))}
                            </ul>
                          ) : null}
                        </div>
                      );
                    })}
                  </div>
                )}
              </section>

              {/* Optional YOLOv8-seg Label Editor */}
              <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-3.5">
                <div className="flex items-center justify-between">
                  <h3 className="inline-flex items-center gap-1.5 text-xs font-bold tracking-wider text-slate-300 uppercase">
                    <Edit3 className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                    YOLOv8-seg Pseudo-Label (`label.txt`)
                  </h3>
                  <button
                    type="button"
                    onClick={() => setShowEditor((v) => !v)}
                    aria-expanded={showEditor}
                    className="min-h-[26px] rounded border border-slate-700 bg-slate-950 px-2 py-0.5 text-[11px] text-slate-300 hover:bg-slate-800"
                  >
                    {showEditor ? "Hide Editor" : "Edit Polygons"}
                  </button>
                </div>

                {showEditor ? (
                  <div className="mt-2">
                    <label
                      htmlFor={labelEditorId}
                      className="mb-1 block text-[11px] text-slate-400"
                    >
                      Format: <code>&lt;class_id&gt; x1 y1 x2 y2 ... xn yn</code> (normalized 0.0–1.0). Saved automatically when you click Approve.
                    </label>
                    <textarea
                      id={labelEditorId}
                      rows={5}
                      value={editedLabelTxt}
                      onChange={(e) => {
                        setEditedLabelTxt(e.target.value);
                        setLabelDirty(true);
                      }}
                      className="w-full rounded-md border border-slate-700 bg-slate-950 p-2 font-mono text-[11px] text-slate-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-sky-400"
                    />
                  </div>
                ) : (
                  <p className="mt-1.5 font-mono text-[11px] text-slate-400">
                    {renderedPolygons.length > 0
                      ? `${renderedPolygons.length} normalized polygon(s) ready for YOLOv8-seg export.`
                      : "Empty label (clear-sky / background frame)."}
                  </p>
                )}
              </section>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
