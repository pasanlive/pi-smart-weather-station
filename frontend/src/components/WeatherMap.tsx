import { useEffect, useRef } from "react";
import L from "leaflet";
import type {
  BasinRiskLevel,
  DistrictThreat,
  FrameMetadata,
  GaugeStatus,
  LayerVisibility,
  NowcastPayload,
  ThreatLevel,
} from "../types/nowcast";
import { renderGaugeSparklineSvg } from "./GaugeSparkline";

interface WeatherMapProps {
  frames: FrameMetadata[];
  activeFrameIndex: number;
  irOpacity: number;
  districtsGeoJson: GeoJSON.FeatureCollection | null;
  nowcast: NowcastPayload | null;
  layers: LayerVisibility;
  selectedDistrictId: string | null;
  selectedBasinId: string | null;
  onSelectDistrict: (id: string) => void;
  onSelectBasin: (id: string) => void;
}

// NASA GIBS BBOX: 75.0,3.0,86.0,12.0 -> Leaflet LatLngBounds [[south, west], [north, east]]
const IR_BOUNDS: L.LatLngBoundsExpression = [
  [3.0, 75.0],
  [12.0, 86.0],
];

const DISTRICT_COLORS: Record<ThreatLevel, { stroke: string; fill: string; fillOpacity: number }> = {
  LOW: { stroke: "#10b981", fill: "#059669", fillOpacity: 0.08 },
  MODERATE: { stroke: "#eab308", fill: "#facc15", fillOpacity: 0.22 },
  HIGH: { stroke: "#f97316", fill: "#fb923c", fillOpacity: 0.34 },
  SEVERE: { stroke: "#f43f5e", fill: "#e11d48", fillOpacity: 0.46 },
};

const BASIN_COLORS: Record<BasinRiskLevel, { stroke: string; fill: string; fillOpacity: number }> = {
  NORMAL: { stroke: "#22c55e", fill: "#22c55e", fillOpacity: 0.16 },
  WATCH: { stroke: "#facc15", fill: "#eab308", fillOpacity: 0.26 },
  WARNING: { stroke: "#fb923c", fill: "#f97316", fillOpacity: 0.34 },
  CRITICAL: { stroke: "#f43f5e", fill: "#ef4444", fillOpacity: 0.42 },
};

const GAUGE_COLORS: Record<GaugeStatus, string> = {
  NORMAL: "#22c55e",
  ALERT: "#facc15",
  MINOR_FLOOD: "#f97316",
  MAJOR_FLOOD: "#ef4444",
};

const STORM_COLORS: Record<string, string> = {
  deep_convective: "#f43f5e",
  cyclonic_band: "#f59e0b",
  monsoon_shield: "#38bdf8",
};

function formatEta(etaMin: number | null): string {
  if (etaMin === 0) return "NOW";
  if (etaMin !== null) return `+${etaMin}m`;
  return "None";
}

function formatAge(iso: string): string {
  try {
    const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
    if (mins < 60) return `${mins}m ago`;
    return `${(mins / 60).toFixed(1)}h ago`;
  } catch {
    return "—";
  }
}

export function WeatherMap({
  frames,
  activeFrameIndex,
  irOpacity,
  districtsGeoJson,
  nowcast,
  layers,
  selectedDistrictId,
  selectedBasinId,
  onSelectDistrict,
  onSelectBasin,
}: WeatherMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const irOverlayRef = useRef<L.ImageOverlay | null>(null);

  const districtGroupRef = useRef<L.LayerGroup | null>(null);
  const basinGroupRef = useRef<L.LayerGroup | null>(null);
  const stormGroupRef = useRef<L.LayerGroup | null>(null);
  const trajectoryGroupRef = useRef<L.LayerGroup | null>(null);
  const vectorGroupRef = useRef<L.LayerGroup | null>(null);
  const gaugeGroupRef = useRef<L.LayerGroup | null>(null);
  const stationGroupRef = useRef<L.LayerGroup | null>(null);

  const districtLayerMapRef = useRef<Map<string, L.Layer>>(new Map());
  const basinLayerMapRef = useRef<Map<string, L.Layer>>(new Map());

  // 1. Initialize Leaflet map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [7.87, 80.77],
      zoom: 8,
      minZoom: 6,
      maxZoom: 12,
      zoomControl: false,
      preferCanvas: true,
    });

    L.control.zoom({ position: "topright" }).addTo(map);

    // Lightweight dark CartoDB basemap ideal for Pi 5 rendering
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap &copy; CARTO | Himawari-9 NASA GIBS",
      subdomains: "abcd",
      maxZoom: 19,
    }).addTo(map);

    // Ordered panes for clean visual layering
    map.createPane("irPane").style.zIndex = "250";
    map.createPane("districtPane").style.zIndex = "350";
    map.createPane("basinPane").style.zIndex = "380";
    map.createPane("trajectoryPane").style.zIndex = "420";
    map.createPane("stormPane").style.zIndex = "450";
    map.createPane("vectorPane").style.zIndex = "500";
    map.createPane("markerPaneCustom").style.zIndex = "560";

    districtGroupRef.current = L.layerGroup().addTo(map);
    basinGroupRef.current = L.layerGroup().addTo(map);
    trajectoryGroupRef.current = L.layerGroup().addTo(map);
    stormGroupRef.current = L.layerGroup().addTo(map);
    vectorGroupRef.current = L.layerGroup().addTo(map);
    gaugeGroupRef.current = L.layerGroup().addTo(map);
    stationGroupRef.current = L.layerGroup().addTo(map);

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // 2. Himawari-9 IR animated image overlay
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const activeFrame = frames[activeFrameIndex];
    if (!layers.satellite || !activeFrame) {
      if (irOverlayRef.current) {
        irOverlayRef.current.remove();
        irOverlayRef.current = null;
      }
      return;
    }

    if (!irOverlayRef.current) {
      irOverlayRef.current = L.imageOverlay(activeFrame.image_url, IR_BOUNDS, {
        opacity: irOpacity,
        pane: "irPane",
        interactive: false,
      }).addTo(map);
    } else {
      irOverlayRef.current.setUrl(activeFrame.image_url);
      irOverlayRef.current.setOpacity(irOpacity);
    }
  }, [frames, activeFrameIndex, irOpacity, layers.satellite]);

  // 3. Sri Lanka 25-District choropleth layer
  useEffect(() => {
    const group = districtGroupRef.current;
    if (!group) return;
    group.clearLayers();
    districtLayerMapRef.current.clear();

    if (!layers.districts || !districtsGeoJson) return;

    const threatById = new Map<string, DistrictThreat>();
    for (const d of nowcast?.districts ?? []) {
      threatById.set(d.district_id, d);
    }

    L.geoJSON(districtsGeoJson, {
      pane: "districtPane",
      style: (feature) => {
        const did = String(feature?.properties?.district_id ?? "");
        const threat = threatById.get(did);
        const lvl: ThreatLevel = threat?.threat_level ?? "LOW";
        const c = DISTRICT_COLORS[lvl];
        const selected = did === selectedDistrictId;
        return {
          color: selected ? "#22d3ee" : c.stroke,
          weight: selected ? 2.5 : 1.1,
          fillColor: c.fill,
          fillOpacity: selected ? Math.min(0.6, c.fillOpacity + 0.15) : c.fillOpacity,
        };
      },
      onEachFeature: (feature, layer) => {
        const did = String(feature?.properties?.district_id ?? "");
        const name = String(feature?.properties?.name ?? did);
        const t = threatById.get(did);
        districtLayerMapRef.current.set(did, layer);

        const tooltipHtml = t
          ? `<div><strong>${name}</strong> • <span style="font-weight:700">${t.threat_level}</span><br/>ETA: ${formatEta(t.eta_min)} | Peak Cover: ${t.peak_coverage_pct.toFixed(0)}%<br/>Rain: ${t.precipitation_mm.toFixed(1)} mm | Gusts: ${t.wind_gusts_kmh.toFixed(0)} km/h</div>`
          : `<strong>${name}</strong>`;
        layer.bindTooltip(tooltipHtml, { sticky: true });
        layer.on("click", () => onSelectDistrict(did));
      },
    }).addTo(group);
  }, [districtsGeoJson, nowcast?.districts, layers.districts, selectedDistrictId, onSelectDistrict]);

  // 4. Wet-Zone Flood Basins overlay (dashed borders, risk-colored fills, rich click popups)
  useEffect(() => {
    const group = basinGroupRef.current;
    if (!group) return;
    group.clearLayers();
    basinLayerMapRef.current.clear();

    if (!layers.floodBasins || !nowcast?.river_basins_geojson) return;

    L.geoJSON(nowcast.river_basins_geojson, {
      pane: "basinPane",
      style: (feature) => {
        const risk = (feature?.properties?.risk_level ?? "NORMAL") as BasinRiskLevel;
        const bid = String(feature?.properties?.basin_id ?? "");
        const c = BASIN_COLORS[risk] ?? BASIN_COLORS.NORMAL;
        const selected = bid === selectedBasinId;
        return {
          color: selected ? "#38bdf8" : c.stroke,
          weight: selected ? 3 : 2.2,
          dashArray: "7 5",
          fillColor: c.fill,
          fillOpacity: selected ? Math.min(0.55, c.fillOpacity + 0.12) : c.fillOpacity,
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature?.properties ?? {};
        const bid = String(p.basin_id ?? "");
        const risk = String(p.risk_level ?? "NORMAL") as BasinRiskLevel;
        const badgeColor = BASIN_COLORS[risk]?.stroke ?? "#22c55e";
        basinLayerMapRef.current.set(bid, layer);

        const dsList = Array.isArray(p.downstream_stations)
          ? p.downstream_stations.join(", ")
          : "—";
        const ucList = Array.isArray(p.upper_catchment_stations)
          ? p.upper_catchment_stations.join(", ")
          : "—";
        const lagWin = Array.isArray(p.drainage_lag_window_hours)
          ? `${p.drainage_lag_window_hours[0]}–${p.drainage_lag_window_hours[1]} hrs`
          : `${p.drainage_lag_hours ?? 12} hrs`;

        const popupHtml = `
          <div style="min-width:230px;font-size:12px;line-height:1.45;">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px;">
              <strong style="font-size:13px;color:#f8fafc;">${p.name ?? bid}</strong>
              <span style="padding:2px 6px;border-radius:4px;font-family:monospace;font-size:10px;font-weight:700;border:1px solid ${badgeColor};color:${badgeColor};">${risk}</span>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 10px;font-family:monospace;font-size:11px;background:#020617;padding:6px 8px;border-radius:6px;border:1px solid #1e293b;">
              <div>Current Cover: <strong style="color:#f8fafc;">${Number(p.storm_coverage_pct ?? 0).toFixed(0)}%</strong></div>
              <div>+60m Cover: <strong style="color:#38bdf8;">${Number(p.projected_60m_pct ?? 0).toFixed(0)}%</strong></div>
              <div>24h Rain: <strong style="color:#f8fafc;">${Number(p.rainfall_24h_mm ?? 0).toFixed(1)} mm</strong></div>
              <div>Drainage Lag: <strong style="color:#fbbf24;">${lagWin}</strong></div>
            </div>
            <div style="margin-top:6px;font-size:11px;color:#cbd5e1;">
              <div><span style="color:#94a3b8;">Upper Catchment:</span> ${ucList}</div>
              <div style="margin-top:2px;"><span style="color:#94a3b8;">Monitor Downstream:</span> <strong style="color:#e2e8f0;">${dsList}</strong></div>
              ${
                p.upstream_surge
                  ? `<div style="margin-top:5px;padding:4px 6px;border-radius:4px;background:rgba(245,158,11,0.2);color:#fde68a;font-weight:600;">⚠ Upstream Surge Active</div>`
                  : ""
              }
            </div>
          </div>
        `;
        layer.bindPopup(popupHtml);
        layer.bindTooltip(`${p.name ?? bid} • ${risk} (${Number(p.rainfall_24h_mm ?? 0).toFixed(0)} mm/24h)`, {
          sticky: true,
        });
        layer.on("click", () => onSelectBasin(bid));
      },
    }).addTo(group);
  }, [nowcast?.river_basins_geojson, layers.floodBasins, selectedBasinId, onSelectBasin]);

  // Open popup when a basin is clicked from the side panel
  useEffect(() => {
    if (!selectedBasinId || !layers.floodBasins) return;
    const layer = basinLayerMapRef.current.get(selectedBasinId);
    if (layer && "openPopup" in layer && typeof (layer as L.Path).openPopup === "function") {
      (layer as L.Path).openPopup();
    }
  }, [selectedBasinId, layers.floodBasins]);

  // 5. Current Storm Masks
  useEffect(() => {
    const group = stormGroupRef.current;
    if (!group) return;
    group.clearLayers();
    if (!layers.stormMasks || !nowcast?.storm_cells_geojson) return;

    L.geoJSON(nowcast.storm_cells_geojson, {
      pane: "stormPane",
      style: (feature) => {
        const lbl = String(feature?.properties?.label ?? "monsoon_shield");
        const color = STORM_COLORS[lbl] ?? "#38bdf8";
        return {
          color,
          weight: 1.8,
          fillColor: color,
          fillOpacity: lbl === "deep_convective" ? 0.34 : 0.2,
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature?.properties ?? {};
        layer.bindTooltip(
          `<strong>${String(p.label ?? "storm").replace("_", " ").toUpperCase()}</strong> (${Math.round(
            Number(p.confidence ?? 0) * 100
          )}%)<br/>Motion: ${Number(p.speed_kmh ?? 0).toFixed(1)} km/h ${p.compass_dir ?? "STAT"}`,
          { sticky: true }
        );
      },
    }).addTo(group);
  }, [nowcast?.storm_cells_geojson, layers.stormMasks]);

  // 6. Projected Trajectories (+30m / +60m / +120m stepped-opacity dashed polygons)
  useEffect(() => {
    const group = trajectoryGroupRef.current;
    if (!group) return;
    group.clearLayers();
    if (!layers.projectedTrajectories || !nowcast?.projected_geojson) return;

    const horizons: { key: string; dash: string; opacity: number; fillOpacity: number; color: string }[] = [
      { key: "+120m", dash: "2 6", opacity: 0.45, fillOpacity: 0.05, color: "#f59e0b" },
      { key: "+60m", dash: "5 5", opacity: 0.68, fillOpacity: 0.1, color: "#fb923c" },
      { key: "+30m", dash: "8 4", opacity: 0.9, fillOpacity: 0.16, color: "#f43f5e" },
    ];

    for (const h of horizons) {
      const fc = nowcast.projected_geojson[h.key];
      if (!fc || !fc.features?.length) continue;
      L.geoJSON(fc, {
        pane: "trajectoryPane",
        style: () => ({
          color: h.color,
          weight: 1.5,
          dashArray: h.dash,
          opacity: h.opacity,
          fillColor: h.color,
          fillOpacity: h.fillOpacity,
        }),
        onEachFeature: (feature, layer) => {
          const p = feature?.properties ?? {};
          layer.bindTooltip(
            `Projected ${h.key}: ${String(p.label ?? "").replace("_", " ")} (${Number(p.speed_kmh ?? 0).toFixed(0)} km/h ${p.compass_dir ?? ""})`,
            { sticky: true }
          );
        },
      }).addTo(group);
    }
  }, [nowcast?.projected_geojson, layers.projectedTrajectories]);

  // 7. Storm Velocity Arrows (drawn from centroid along bearing_deg scaled by speed_kmh)
  useEffect(() => {
    const group = vectorGroupRef.current;
    if (!group) return;
    group.clearLayers();
    if (!layers.velocityArrows || !nowcast?.storm_cells?.length) return;

    for (const cell of nowcast.storm_cells) {
      if (cell.speed_kmh < 1.5) continue;
      // Scale arrow length from 24px (slow) to 56px (fast >= 80 km/h)
      const len = Math.round(24 + Math.min(1, cell.speed_kmh / 80) * 32);
      const box = len + 18;
      const mid = box / 2;
      const stroke = cell.label === "deep_convective" ? "#fb7185" : "#22d3ee";

      const svg = `
        <svg width="${box}" height="${box}" viewBox="0 0 ${box} ${box}" style="overflow:visible;">
          <g transform="rotate(${cell.bearing_deg.toFixed(1)} ${mid} ${mid})">
            <line x1="${mid}" y1="${mid}" x2="${mid}" y2="${mid - len}" stroke="${stroke}" stroke-width="2.4" stroke-linecap="round" />
            <polygon points="${mid},${mid - len - 6} ${mid - 5},${mid - len + 3} ${mid + 5},${mid - len + 3}" fill="${stroke}" />
            <circle cx="${mid}" cy="${mid}" r="2.5" fill="#f8fafc" />
          </g>
        </svg>
      `;

      const icon = L.divIcon({
        html: svg,
        className: "",
        iconSize: [box, box],
        iconAnchor: [mid, mid],
      });

      L.marker([cell.centroid_lat, cell.centroid_lon], {
        icon,
        pane: "vectorPane",
      })
        .bindTooltip(
          `<strong>${cell.speed_kmh.toFixed(1)} km/h ${cell.compass_dir}</strong> (${cell.bearing_deg.toFixed(0)}°)<br/>${cell.label.replace("_", " ")}`,
          { direction: "top" }
        )
        .addTo(group);
    }
  }, [nowcast?.storm_cells, layers.velocityArrows]);

  // 8. River Gauge Markers (Irrigation Dept ArcGIS / Fallback) with 24h SVG Sparkline Popups
  useEffect(() => {
    const group = gaugeGroupRef.current;
    if (!group) return;
    group.clearLayers();
    if (!layers.riverGauges || !nowcast?.gauges?.length) return;

    for (const g of nowcast.gauges) {
      const color = GAUGE_COLORS[g.status] ?? "#22c55e";
      const trendGlyph = g.trend === "RISING" ? "▲" : g.trend === "FALLING" ? "▼" : "•";
      const iconHtml = `
        <div style="display:flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:9999px;background:#020617;border:2.5px solid ${color};color:${color};font-size:10px;font-weight:900;box-shadow:0 0 10px rgba(0,0,0,0.8);">
          ${trendGlyph}
        </div>
      `;
      const icon = L.divIcon({
        html: iconHtml,
        className: "",
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      const sparkline = renderGaugeSparklineSvg(g.history_24h, g.alert_level_ft, color);
      const popupHtml = `
        <div style="min-width:225px;font-size:12px;line-height:1.4;">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;">
            <strong style="color:#f8fafc;">${g.station_name}</strong>
            <span style="font-family:monospace;font-size:10px;font-weight:700;color:${color};border:1px solid ${color};padding:1px 5px;border-radius:4px;">${g.status.replace("_", " ")}</span>
          </div>
          <div style="font-size:11px;color:#94a3b8;margin-bottom:5px;">${g.river_name} (${g.role}) • Updated ${formatAge(g.reading_timestamp)}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;font-family:monospace;font-size:11px;background:#020617;padding:6px 8px;border-radius:6px;border:1px solid #1e293b;">
            <div>Level: <strong style="color:#f8fafc;">${g.water_level_ft.toFixed(2)} ft</strong></div>
            <div>Rate: <strong style="color:${g.trend === "RISING" ? "#fbbf24" : "#94a3b8"};">${g.rate_of_rise_ft_hr >= 0 ? "+" : ""}${g.rate_of_rise_ft_hr.toFixed(2)} ft/h</strong></div>
            <div>Alert: <strong>${g.alert_level_ft !== null ? `${g.alert_level_ft} ft` : "—"}</strong></div>
            <div>Minor/Maj: <strong>${g.minor_flood_ft ?? "—"}/${g.major_flood_ft ?? "—"}</strong></div>
            <div style="grid-column:span 2;">24h Rain: <strong>${g.rainfall_24h_mm !== null ? `${g.rainfall_24h_mm.toFixed(1)} mm` : "—"}</strong></div>
          </div>
          <div style="margin-top:6px;">
            <div style="font-size:10px;color:#94a3b8;">24-Hour Water Level Trend (dashed = alert)</div>
            ${sparkline}
          </div>
        </div>
      `;

      L.marker([g.lat, g.lon], { icon, pane: "markerPaneCustom" })
        .bindPopup(popupHtml)
        .bindTooltip(
          `<strong>${g.station_name}</strong>: ${g.water_level_ft.toFixed(2)} ft (${g.trend}, ${g.status.replace("_", " ")})`,
          { direction: "top" }
        )
        .addTo(group);
    }
  }, [nowcast?.gauges, layers.riverGauges]);

  // 9. Open-Meteo Surface Stations
  useEffect(() => {
    const group = stationGroupRef.current;
    if (!group) return;
    group.clearLayers();
    if (!layers.surfaceStations || !nowcast?.surface_stations?.length) return;

    for (const st of nowcast.surface_stations) {
      L.circleMarker([st.lat, st.lon], {
        radius: 5,
        color: "#c4b5fd",
        weight: 1.5,
        fillColor: "#7c3aed",
        fillOpacity: 0.85,
        pane: "markerPaneCustom",
      })
        .bindTooltip(
          `<strong>${st.name}</strong><br/>Rain: ${st.precipitation_mm.toFixed(1)} mm | Gusts: ${st.wind_gusts_kmh.toFixed(0)} km/h<br/>Pressure: ${st.surface_pressure_hpa.toFixed(1)} hPa (3h Δ: ${st.pressure_drop_3h_hpa >= 0 ? "-" : "+"}${Math.abs(st.pressure_drop_3h_hpa).toFixed(1)})`,
          { direction: "top" }
        )
        .addTo(group);
    }
  }, [nowcast?.surface_stations, layers.surfaceStations]);

  return <div ref={containerRef} className="h-full w-full" aria-label="Sri Lanka Interactive Weather Nowcast Map" />;
}
