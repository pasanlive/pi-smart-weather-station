import { useCallback, useEffect, useState } from "react";
import { api, createNowcastWebSocket } from "./api/client";
import type {
  ActiveLearningStats,
  FrameMetadata,
  LayerVisibility,
  NowcastPayload,
  SystemStatusResponse,
} from "./types/nowcast";
import { WeatherMap } from "./components/WeatherMap";
import { LayerToggleBar } from "./components/LayerToggleBar";
import { TimelineControls } from "./components/TimelineControls";
import { StatusHeader, type DashboardTab } from "./components/StatusHeader";
import { UpstreamSurgeBanner } from "./components/UpstreamSurgeBanner";
import { AdvisoryCard } from "./components/AdvisoryCard";
import { BasinRiskPanel } from "./components/BasinRiskPanel";
import { DistrictThreatTable } from "./components/DistrictThreatTable";
import { ActiveLearningPanel } from "./components/ActiveLearningPanel";

export default function App() {
  const [status, setStatus] = useState<SystemStatusResponse | null>(null);
  const [frames, setFrames] = useState<FrameMetadata[]>([]);
  const [districtsGeoJson, setDistrictsGeoJson] = useState<GeoJSON.FeatureCollection | null>(null);
  const [nowcast, setNowcast] = useState<NowcastPayload | null>(null);

  const [activeFrameIndex, setActiveFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [irOpacity, setIrOpacity] = useState(0.72);
  const [wsConnected, setWsConnected] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const [selectedDistrictId, setSelectedDistrictId] = useState<string | null>(null);
  const [selectedBasinId, setSelectedBasinId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<DashboardTab>("nowcast");
  const [alStats, setAlStats] = useState<ActiveLearningStats | null>(null);

  const [layers, setLayers] = useState<LayerVisibility>({
    satellite: true,
    districts: true,
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

    void loadAll();
    void api.getActiveLearningStats().then(setAlStats).catch(() => {});

    const cleanupWs = createNowcastWebSocket(
      (payload) => {
        setNowcast(payload);
        void api.getFrames().then(setFrames).catch(() => {});
        void api.getStatus().then(setStatus).catch(() => {});
        void api.getActiveLearningStats().then(setAlStats).catch(() => {});
      },
      setWsConnected
    );

    // Periodic status telemetry refresh (CPU/RAM/buffered frames/active-learning queue)
    const statusInterval = window.setInterval(() => {
      void api.getStatus().then(setStatus).catch(() => {});
      void api.getFrames().then(setFrames).catch(() => {});
      void api.getActiveLearningStats().then(setAlStats).catch(() => {});
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

  return (
    <div className="flex h-screen w-screen flex-col overflow-hidden bg-slate-950 text-slate-100 lg:flex-row">
      {/* Main Workspace Area: Live Interactive Map OR Active Learning Review QA */}
      <main className="relative h-[58vh] flex-1 lg:h-full">
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
            nowcast={nowcast}
            layers={layers}
            selectedDistrictId={selectedDistrictId}
            selectedBasinId={selectedBasinId}
            onSelectDistrict={setSelectedDistrictId}
            onSelectBasin={(id) => {
              setSelectedBasinId(id);
              setLayers((prev) => ({ ...prev, floodBasins: true }));
            }}
          />

          {/* Floating Top-Left Layer Toggle Deck */}
          <div className="pointer-events-none absolute left-3 top-3 right-14 z-[600]">
            <LayerToggleBar
              layers={layers}
              onToggle={toggleLayer}
              gaugeCount={nowcast?.gauges.length ?? 0}
              stormCount={nowcast?.storm_cells.length ?? 0}
            />
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

      {/* Right Side Intelligence Panel */}
      <aside className="flex h-[42vh] w-full flex-col border-t border-slate-800 bg-slate-950 lg:h-full lg:w-[430px] lg:border-l lg:border-t-0">
        <StatusHeader
          status={status}
          hydrologyStatus={nowcast?.hydrology_source_status ?? status?.hydrology_source_status ?? "unavailable"}
          wsConnected={wsConnected}
          isRefreshing={isRefreshing}
          onRefresh={handleRefresh}
          activeTab={activeTab}
          onTabChange={setActiveTab}
          pendingAlCount={alStats?.pending ?? 0}
        />

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
            stormCount={nowcast?.storm_cells.length ?? 0}
          />

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
    </div>
  );
}
