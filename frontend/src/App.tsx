import { useCallback, useEffect, useState } from "react";
import { PanelLeftOpen, PanelRightClose, PanelRightOpen, Bell } from "lucide-react";
import { api, createNowcastWebSocket } from "./api/client";
import type {
  ActiveLearningStats,
  AlertHistoryItem,
  AlertStatusResponse,
  FrameMetadata,
  HazardMapMode,
  LayerVisibility,
  NowcastPayload,
  SystemStatusResponse,
} from "./types/nowcast";
import { WeatherMap } from "./components/WeatherMap";
import { LayerToggleBar } from "./components/LayerToggleBar";
import { HazardModeSwitcher } from "./components/HazardModeSwitcher";
import { TimelineControls } from "./components/TimelineControls";
import { StatusHeader, type DashboardTab } from "./components/StatusHeader";
import { UpstreamSurgeBanner } from "./components/UpstreamSurgeBanner";
import { AdvisoryCard } from "./components/AdvisoryCard";
import { HourlyDigestPanel } from "./components/HourlyDigestPanel";
import { TelegramAlertsPanel } from "./components/TelegramAlertsPanel";
import { BasinRiskPanel } from "./components/BasinRiskPanel";
import { DistrictThreatTable } from "./components/DistrictThreatTable";
import { ActiveLearningPanel } from "./components/ActiveLearningPanel";
import { ResizeHandle } from "./components/ResizeHandle";

const LEFT_DEFAULT_WIDTH = 390;
const RIGHT_DEFAULT_WIDTH = 360;
const MIN_PANEL_WIDTH = 280;
const MAX_LEFT_WIDTH = 620;
const MAX_RIGHT_WIDTH = 560;

function readSavedNumber(key: string, fallback: number, min: number, max: number): number {
  try {
    const raw = window.localStorage.getItem(key);
    if (!raw) return fallback;
    const n = Number(raw);
    if (!Number.isFinite(n)) return fallback;
    return Math.max(min, Math.min(max, Math.round(n)));
  } catch {
    return fallback;
  }
}

function readSavedBool(key: string, fallback: boolean): boolean {
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    return raw === "true";
  } catch {
    return fallback;
  }
}

export default function App() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [frames, setFrames] = useState<FrameMetadata[]>([]);
  const [districtsGeoJson, setDistrictsGeoJson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [seaZonesGeoJson, setSeaZonesGeoJson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [nowcast, setNowcast] = useState<NowcastPayload | null>(null);
  const [hazardMode, setHazardMode] = useState<HazardMapMode>("combined");

  const [activeFrameIndex, setActiveFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [irOpacity, setIrOpacity] = useState(0.72);
  const [wsConnected, setWsConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const [selectedDistrictId, setSelectedDistrictId] = useState<string | null>(null);
  const [selectedBasinId, setSelectedBasinId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<DashboardTab>("nowcast");
  const [alStats, setAlStats] = useState<ActiveLearningStats | null>(null);
  const [alertStatus, setAlertStatus] = useState<AlertStatusResponse | null>(null);
  const [alertHistory, setAlertHistory] = useState<AlertHistoryItem[]>([]);

  // Collapsible & resizable left/right panel state (persisted in localStorage)
  const [leftOpen, setLeftOpen] = useState<boolean>(() => readSavedBool("sl_left_panel_open", true));
  const [rightOpen, setRightOpen] = useState<boolean>(() => readSavedBool("sl_right_panel_open", true));
  const [leftWidth, setLeftWidth] = useState<number>(() =>
    readSavedNumber("sl_left_panel_width", LEFT_DEFAULT_WIDTH, MIN_PANEL_WIDTH, MAX_LEFT_WIDTH)
  );
  const [rightWidth, setRightWidth] = useState<number>(() =>
    readSavedNumber("sl_right_panel_width", RIGHT_DEFAULT_WIDTH, MIN_PANEL_WIDTH, MAX_RIGHT_WIDTH)
  );
  const [isDesktop, setIsDesktop] = useState<boolean>(() =>
    typeof window !== "undefined" ? window.innerWidth >= 1024 : true
  );

  useEffect(() => {
    const mq = window.matchMedia("(min-width: 1024px)");
    const update = () => setIsDesktop(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem("sl_left_panel_open", String(leftOpen));
      window.localStorage.setItem("sl_right_panel_open", String(rightOpen));
      window.localStorage.setItem("sl_left_panel_width", String(leftWidth));
      window.localStorage.setItem("sl_right_panel_width", String(rightWidth));
    } catch {
      // ignore storage quota errors
    }
  }, [leftOpen, rightOpen, leftWidth, rightWidth]);

  const [layers, setLayers] = useState<LayerVisibility>({
    satellite: true,
    districts: true,
    seaZones: true,
    floodBasins: true,
    riverGauges: true,
    stormMasks: true,
    projectedTrajectories: true,
    velocityArrows: true,
    surfaceStations: true,
  });

  const loadAll = useCallback(async () => {
    try {
      const [st, fr, nc] = await Promise.all([
        api.getStatus(),
        api.getFrames(),
        api.getLatestNowcast(),
      ]);
      setStatus(st);
      setFrames(fr);
      setNowcast(nc);
      if (fr.length > 0) {
        setActiveFrameIndex((prev) => (prev >= fr.length ? fr.length - 1 : prev));
      }
    } catch (err) {
      console.error("Failed loading nowcast telemetry:", err);
    }
  }, []);

  useEffect(() => {
    api
      .getDistrictsGeoJSON()
      .then(setDistrictsGeoJson)
      .catch((err) => console.error("Failed loading districts GeoJSON:", err));

    api
      .getSeaZonesGeoJSON()
      .then(setSeaZonesGeoJson)
      .catch((err) => console.error("Failed loading sea zones GeoJSON:", err));

    void loadAll();
    void api.getActiveLearningStats().then(setAlStats).catch(() => {});
    void api.getAlertStatus().then(setAlertStatus).catch(() => {});
    void api
      .getAlertHistory(10)
      .then((res) => setAlertHistory(res.alerts))
      .catch(() => {});

    const cleanupWs = createNowcastWebSocket(
      (payload) => {
        setNowcast(payload);
        void api.getFrames().then(setFrames).catch(() => {});
        void api.getStatus().then(setStatus).catch(() => {});
        void api.getActiveLearningStats().then(setAlStats).catch(() => {});
        void api.getAlertStatus().then(setAlertStatus).catch(() => {});
        void api
          .getAlertHistory(10)
          .then((res) => setAlertHistory(res.alerts))
          .catch(() => {});
      },
      setWsConnected
    );

    const statusInterval = window.setInterval(() => {
      void api.getStatus().then(setStatus).catch(() => {});
      void api.getFrames().then(setFrames).catch(() => {});
      void api.getActiveLearningStats().then(setAlStats).catch(() => {});
      void api.getAlertStatus().then(setAlertStatus).catch(() => {});
      void api
        .getAlertHistory(10)
        .then((res) => setAlertHistory(res.alerts))
        .catch(() => {});
    }, 30000);

    return () => {
      cleanupWs();
      window.clearInterval(statusInterval);
    };
  }, [loadAll]);

  // Animate 2-hour Himawari-9 IR frame loop smoothly at ~1.6 fps
  useEffect(() => {
    if (!isPlaying || frames.length < 2) return;
    const timer = window.setInterval(() => {
      setActiveFrameIndex((idx) => (idx + 1) % frames.length);
    }, 650);
    return () => window.clearInterval(timer);
  }, [isPlaying, frames.length]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    try {
      const nc = await api.refreshNowcast();
      setNowcast(nc);
      const [st, fr] = await Promise.all([api.getStatus(), api.getFrames()]);
      setStatus(st);
      setFrames(fr);
      if (fr.length > 0) setActiveFrameIndex(fr.length - 1);
    } catch (err) {
      console.error("Manual refresh failed:", err);
    } finally {
      setIsRefreshing(false);
    }
  };

  const toggleLayer = (key: keyof LayerVisibility) => {
    setLayers((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const stepFrame = (delta: number) => {
    if (frames.length < 2) return;
    setIsPlaying(false);
    setActiveFrameIndex((idx) => (idx + delta + frames.length) % frames.length);
  };

  const surgeCount = nowcast?.upstream_surge_warning.length ?? 0;

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-slate-950 text-slate-100 lg:flex-row">
      {/* LEFT PANEL: System Info Header + River Basin Flood Risk + District Threat Matrix */}
      {leftOpen ? (
        <>
          <aside
            aria-label="System Status, River Basins and District Threats"
            style={isDesktop ? { width: `${leftWidth}px` } : undefined}
            className="flex max-h-[42vh] w-full shrink-0 flex-col border-b border-slate-800 bg-slate-950 lg:h-full lg:max-h-none lg:border-b-0 lg:border-r"
          >
            <StatusHeader
              status={status}
              hydrologyStatus={nowcast?.hydrology_source_status ?? status?.hydrology_source_status ?? "unavailable"}
              alertStatus={alertStatus}
              wsConnected={wsConnected}
              isRefreshing={isRefreshing}
              onRefresh={handleRefresh}
              activeTab={activeTab}
              onTabChange={setActiveTab}
              pendingAlCount={alStats?.pending ?? 0}
              onCollapse={() => setLeftOpen(false)}
            />

            <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3">
              <BasinRiskPanel
                basins={nowcast?.river_basins ?? []}
                selectedBasinId={selectedBasinId}
                onSelectBasin={(id) => {
                  setSelectedBasinId(id);
                  setLayers((prev) => ({ ...prev, floodBasins: true }));
                }}
              />

              <DistrictThreatTable
                districts={nowcast?.districts ?? []}
                selectedDistrictId={selectedDistrictId}
                onSelectDistrict={setSelectedDistrictId}
              />
            </div>
          </aside>

          <ResizeHandle
            side="left"
            width={leftWidth}
            minWidth={MIN_PANEL_WIDTH}
            maxWidth={MAX_LEFT_WIDTH}
            defaultWidth={LEFT_DEFAULT_WIDTH}
            label="Resize left system and threats panel"
            onResize={setLeftWidth}
          />
        </>
      ) : null}

      {/* CENTER WORKSPACE: Live Interactive Leaflet Map OR Active Learning Review QA */}
      <main className="relative min-h-[36vh] min-w-0 flex-1 lg:h-full">
        {/* Re-open Left Panel Floating Button when collapsed */}
        {!leftOpen ? (
          <button
            type="button"
            onClick={() => setLeftOpen(true)}
            aria-label="Expand left system and threats panel"
            className="absolute left-3 top-3 z-[650] inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/45 bg-slate-950/92 px-2.5 py-2 text-xs font-semibold text-cyan-200 shadow-xl backdrop-blur-md transition hover:bg-slate-900"
          >
            <PanelLeftOpen className="h-4 w-4" />
            <span className="hidden sm:inline">System &amp; Threats</span>
          </button>
        ) : null}

        {/* Re-open Right Panel Floating Button when collapsed */}
        {!rightOpen ? (
          <button
            type="button"
            onClick={() => setRightOpen(true)}
            aria-label="Expand right advisory and alerts panel"
            className="absolute right-14 top-3 z-[650] inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/45 bg-slate-950/92 px-2.5 py-2 text-xs font-semibold text-cyan-200 shadow-xl backdrop-blur-md transition hover:bg-slate-900"
          >
            <PanelRightOpen className="h-4 w-4" />
            <span className="hidden sm:inline">Alerts &amp; Advisory</span>
            {surgeCount > 0 ? (
              <span className="rounded-full bg-amber-500/30 px-1.5 py-0.2 font-mono text-[10px] font-bold text-amber-200">
                {surgeCount}
              </span>
            ) : null}
          </button>
        ) : null}

        <div
          role="tabpanel"
          aria-labelledby="tab-nowcast"
          hidden={activeTab !== "nowcast"}
          className="relative h-full w-full"
        >
          <WeatherMap
            frames={frames}
            activeFrameIndex={activeFrameIndex}
            irOpacity={irOpacity}
            districtsGeoJson={districtsGeoJson}
            seaZonesGeoJson={seaZonesGeoJson}
            nowcast={nowcast}
            layers={layers}
            hazardMode={hazardMode}
            selectedDistrictId={selectedDistrictId}
            selectedBasinId={selectedBasinId}
            onSelectDistrict={setSelectedDistrictId}
            onSelectBasin={(id) => {
              setSelectedBasinId(id);
              setLayers((prev) => ({ ...prev, floodBasins: true }));
            }}
          />

          {/* Floating Top-Left Layer Toggle & Hazard Probability Switcher Deck */}
          <div
            className={`pointer-events-none absolute top-3 z-[600] flex flex-col items-start gap-2 ${
              !leftOpen ? "left-3 sm:left-44" : "left-3"
            } ${!rightOpen ? "right-14 sm:right-52" : "right-14"}`}
          >
            <LayerToggleBar
              layers={layers}
              onToggle={toggleLayer}
              gaugeCount={nowcast?.gauges.length ?? 0}
              stormCount={nowcast?.storm_cells.length ?? 0}
            />
            <HazardModeSwitcher mode={hazardMode} onChange={setHazardMode} />
          </div>

          {/* Floating Bottom Satellite Timeline Playback Deck */}
          <div className="pointer-events-none absolute bottom-3 left-3 right-3 z-[600] max-w-2xl">
            <TimelineControls
              frames={frames}
              activeIndex={activeFrameIndex}
              isPlaying={isPlaying}
              opacity={irOpacity}
              onTogglePlay={() => setIsPlaying((p) => !p)}
              onStep={stepFrame}
              onSelectIndex={(idx) => {
                setIsPlaying(false);
                setActiveFrameIndex(idx);
              }}
              onOpacityChange={setIrOpacity}
            />
          </div>
        </div>

        <div
          role="tabpanel"
          aria-labelledby="tab-active-learning"
          hidden={activeTab !== "active-learning"}
          className="h-full w-full"
        >
          <ActiveLearningPanel
            stats={alStats}
            onStatsChange={setAlStats}
            isActive={activeTab === "active-learning"}
          />
        </div>
      </main>

      {/* RIGHT PANEL: Upstream Surges + Threat Advisory + Hourly Digest + Telegram Alerts */}
      {rightOpen ? (
        <>
          <ResizeHandle
            side="right"
            width={rightWidth}
            minWidth={MIN_PANEL_WIDTH}
            maxWidth={MAX_RIGHT_WIDTH}
            defaultWidth={RIGHT_DEFAULT_WIDTH}
            label="Resize right advisory and alerts panel"
            onResize={setRightWidth}
          />

          <aside
            aria-label="Advisories and Alert Channels"
            style={isDesktop ? { width: `${rightWidth}px` } : undefined}
            className="flex max-h-[38vh] w-full shrink-0 flex-col border-t border-slate-800 bg-slate-950 lg:h-full lg:max-h-none lg:border-l lg:border-t-0"
          >
            <div className="flex items-center justify-between border-b border-slate-800/90 bg-slate-950/95 px-3.5 py-2.5">
              <div className="flex items-center gap-2">
                <Bell className="h-3.5 w-3.5 text-cyan-400" />
                <h2 className="text-xs font-bold uppercase tracking-wider text-slate-200">
                  Intelligence &amp; Alerts
                </h2>
              </div>
              <button
                type="button"
                onClick={() => setRightOpen(false)}
                title="Collapse right panel"
                aria-label="Collapse right panel"
                className="inline-flex items-center justify-center rounded-md border border-slate-700 bg-slate-900 p-1.5 text-slate-300 transition hover:border-cyan-500/50 hover:text-cyan-200"
              >
                <PanelRightClose className="h-4 w-4" />
              </button>
            </div>

            <div className="flex min-h-0 flex-1 flex-col gap-2.5 overflow-y-auto p-3">
              <UpstreamSurgeBanner
                warnings={nowcast?.upstream_surge_warning ?? []}
                onSelectBasin={(id) => {
                  setSelectedBasinId(id);
                  setLayers((prev) => ({ ...prev, floodBasins: true }));
                }}
              />

              <AdvisoryCard
                advisoryText={nowcast?.advisory_text ?? ""}
                advisorySource={nowcast?.advisory_source ?? "template"}
                advisoryModel={nowcast?.advisory_model}
                advisoryLatencyMs={nowcast?.advisory_latency_ms}
                stormCount={nowcast?.storm_cells.length ?? 0}
              />

              <HourlyDigestPanel
                districts={nowcast?.districts ?? []}
                seaZones={nowcast?.sea_zones ?? []}
                thresholdPct={nowcast?.hourly_digest_threshold_pct ?? 50}
                onSelectDistrict={setSelectedDistrictId}
                onTriggerDigest={async (dispatch) => {
                  const res = await api.triggerHourlyDigestTest(dispatch);
                  if (dispatch) {
                    void api.getAlertStatus().then(setAlertStatus).catch(() => {});
                    void api
                      .getAlertHistory(10)
                      .then((h) => setAlertHistory(h.alerts))
                      .catch(() => {});
                  }
                  return res;
                }}
              />

              <TelegramAlertsPanel
                alertStatus={alertStatus}
                history={alertHistory}
                onSendTest={async () => {
                  const res = await api.sendTelegramTestAlert();
                  void api.getAlertStatus().then(setAlertStatus).catch(() => {});
                  void api
                    .getAlertHistory(10)
                    .then((h) => setAlertHistory(h.alerts))
                    .catch(() => {});
                  return res;
                }}
                onSendSummary={async () => {
                  const res = await api.sendTelegramHourlySummary();
                  void api.getAlertStatus().then(setAlertStatus).catch(() => {});
                  void api
                    .getAlertHistory(10)
                    .then((h) => setAlertHistory(h.alerts))
                    .catch(() => {});
                  return res;
                }}
              />
            </div>
          </aside>
        </>
      ) : null}
    </div>
  );
}
