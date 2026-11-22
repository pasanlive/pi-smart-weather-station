from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from api.websocket import ConnectionManager
from models import FrameMetadata, GaugesResponse, NowcastPayload, SystemStatusResponse
from nowcast.pipeline import NowcastPipeline


def create_api_router(pipeline: NowcastPipeline, ws_manager: ConnectionManager) -> APIRouter:
    router = APIRouter()

    @router.get("/api/status", response_model=SystemStatusResponse)
    async def get_status() -> SystemStatusResponse:
        return pipeline.get_status()

    @router.get("/api/frames", response_model=list[FrameMetadata])
    async def list_frames() -> list[FrameMetadata]:
        return pipeline.himawari.get_buffered_frames()

    @router.get("/api/frames/{filename}")
    async def get_frame_image(filename: str) -> FileResponse:
        # Strict path traversal protection
        if "/" in filename or "\\" in filename or ".." in filename or not filename.endswith(".png"):
            raise HTTPException(status_code=400, detail="Invalid frame filename")
        target = (pipeline.settings.frames_path / filename).resolve()
        frames_root = pipeline.settings.frames_path.resolve()
        if frames_root not in target.parents or not target.exists():
            raise HTTPException(status_code=404, detail="Frame not found")
        return FileResponse(
            target,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @router.get("/api/districts")
    async def get_districts_geojson() -> dict[str, Any]:
        path: Path = pipeline.settings.districts_geojson_path
        if not path.exists():
            raise HTTPException(status_code=404, detail="Districts GeoJSON missing")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.get("/api/sea-zones")
    async def get_sea_zones_geojson() -> dict[str, Any]:
        path: Path = pipeline.settings.sea_zones_geojson_path
        if not path.exists():
            raise HTTPException(status_code=404, detail="Sea zones GeoJSON missing")
        return json.loads(path.read_text(encoding="utf-8"))

    @router.get("/api/gauges", response_model=GaugesResponse)
    async def get_gauges() -> GaugesResponse:
        return pipeline.irrigation.get_gauges_response()

    @router.get("/api/nowcast/latest", response_model=NowcastPayload)
    async def get_latest_nowcast() -> NowcastPayload:
        if pipeline.latest_payload is None:
            return await pipeline.run_nowcast()
        return pipeline.latest_payload

    @router.post("/api/nowcast/refresh", response_model=NowcastPayload)
    async def refresh_nowcast() -> NowcastPayload:
        return await pipeline.refresh_all()

    @router.websocket("/ws/nowcast")
    async def nowcast_ws(websocket: WebSocket) -> None:
        await ws_manager.connect(websocket)
        try:
            if pipeline.latest_payload is not None:
                await websocket.send_json(
                    {"type": "nowcast_update", "payload": pipeline.latest_payload.model_dump()}
                )
            while True:
                # Keep connection alive; respond to client ping messages
                msg = await websocket.receive_text()
                if msg.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception:
            ws_manager.disconnect(websocket)

    return router
