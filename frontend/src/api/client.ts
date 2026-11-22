import type {
  ActiveLearningDecisionResponse,
  ActiveLearningQueueResponse,
  ActiveLearningStats,
  AlertHistoryResponse,
  AlertStatusResponse,
  FrameMetadata,
  HourlyDigestTestResponse,
  NowcastPayload,
  SystemStatusResponse,
  TelegramTestResponse,
} from "../types/nowcast";

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = `HTTP ${res.status} on ${url}`;
    try {
      const errBody = (await res.json()) as { detail?: string };
      if (errBody?.detail) detail = errBody.detail;
    } catch {
      // keep HTTP status fallback
    }
    throw new Error(detail);
  }
  return (await res.json()) as T;
}

export const api = {
  getStatus: () => fetchJson<SystemStatusResponse>("/api/status"),
  getFrames: () => fetchJson<FrameMetadata[]>("/api/frames"),
  getDistrictsGeoJSON: () => fetchJson<GeoJSON.FeatureCollection>("/api/districts"),
  getSeaZonesGeoJSON: () => fetchJson<GeoJSON.FeatureCollection>("/api/sea-zones"),
  getLatestNowcast: () => fetchJson<NowcastPayload>("/api/nowcast/latest"),
  refreshNowcast: () => fetchJson<NowcastPayload>("/api/nowcast/refresh", { method: "POST" }),
  getActiveLearningStats: () => fetchJson<ActiveLearningStats>("/api/active-learning/stats"),
  getActiveLearningQueue: (status: "pending" | "approved" | "rejected" | "all" = "pending", limit = 20) =>
    fetchJson<ActiveLearningQueueResponse>(
      `/api/active-learning/queue?status=${encodeURIComponent(status)}&limit=${limit}`
    ),
  submitActiveLearningDecision: (
    sampleId: string,
    payload: { status: "approved" | "rejected"; updated_label_txt?: string }
  ) =>
    fetchJson<ActiveLearningDecisionResponse>(
      `/api/active-learning/${encodeURIComponent(sampleId)}/decision`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    ),
  getActiveLearningExportUrl: () => "/api/active-learning/export",
  getAlertStatus: () => fetchJson<AlertStatusResponse>("/api/alerts/status"),
  getAlertHistory: (limit = 10) =>
    fetchJson<AlertHistoryResponse>(`/api/alerts/history?limit=${limit}`),
  sendTelegramTestAlert: () =>
    fetchJson<TelegramTestResponse>("/api/alerts/telegram/test", { method: "POST" }),
  sendTelegramHourlySummary: () =>
    fetchJson<TelegramTestResponse>("/api/alerts/telegram/summary", { method: "POST" }),
  triggerHourlyDigestTest: (dispatch = true) =>
    fetchJson<HourlyDigestTestResponse>(
      `/api/alerts/hourly-digest/test?dispatch=${dispatch ? "true" : "false"}`,
      { method: "POST" }
    ),
};

export function createNowcastWebSocket(
  onUpdate: (payload: NowcastPayload) => void,
  onStatusChange: (connected: boolean) => void
): () => void {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/nowcast`;

  let ws: WebSocket | null = null;
  let closed = false;
  let reconnectTimer: number | null = null;
  let pingTimer: number | null = null;

  const connect = () => {
    if (closed) return;
    ws = new WebSocket(wsUrl);
    ws.onopen = () => {
      onStatusChange(true);
      pingTimer = window.setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send("ping");
        }
      }, 25000);
    };
    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg?.type === "nowcast_update" && msg.payload) {
          onUpdate(msg.payload as NowcastPayload);
        }
      } catch {
        // Ignore malformed frames
      }
    };
    ws.onclose = () => {
      onStatusChange(false);
      if (pingTimer !== null) window.clearInterval(pingTimer);
      if (!closed) {
        reconnectTimer = window.setTimeout(connect, 5000);
      }
    };
    ws.onerror = () => {
      ws?.close();
    };
  };

  connect();

  return () => {
    closed = true;
    if (reconnectTimer !== null) window.clearTimeout(reconnectTimer);
    if (pingTimer !== null) window.clearInterval(pingTimer);
    ws?.close();
  };
}
