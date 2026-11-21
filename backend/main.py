from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.active_learning import create_active_learning_router
from api.routes import create_api_router
from api.websocket import ConnectionManager
from config import settings
from nowcast.pipeline import NowcastPipeline
from storage.db import Database

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sl-nowcast")

ws_manager = ConnectionManager()
db = Database(settings.db_path)
pipeline = NowcastPipeline(settings=settings, db=db, broadcast_cb=ws_manager.broadcast)
scheduler = AsyncIOScheduler(timezone="UTC")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Kick off non-blocking startup bootstrap so API responds immediately while backfilling
    bootstrap_task = asyncio.create_task(pipeline.startup_bootstrap())

    scheduler.add_job(
        pipeline.poll_satellite_and_nowcast,
        trigger="interval",
        minutes=settings.satellite_poll_interval_min,
        id="himawari_poll",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        pipeline.poll_open_meteo,
        trigger="interval",
        minutes=settings.open_meteo_poll_interval_min,
        id="open_meteo_poll",
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        pipeline.poll_irrigation,
        trigger="interval",
        minutes=settings.irrigation_poll_interval_min,
        id="irrigation_arcgis_poll",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    logger.info(
        "Scheduler started: Himawari=%dm, Open-Meteo=%dm, Irrigation=%dm",
        settings.satellite_poll_interval_min,
        settings.open_meteo_poll_interval_min,
        settings.irrigation_poll_interval_min,
    )

    try:
        yield
    finally:
        scheduler.shutdown(wait=False)
        if not bootstrap_task.done():
            bootstrap_task.cancel()


app = FastAPI(
    title="Sri Lanka Edge-Native Weather Nowcasting & Flood Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(create_api_router(pipeline, ws_manager))
app.include_router(create_active_learning_router(pipeline.active_learning))

# Serve built React SPA on Raspberry Pi 5 when frontend/dist exists
dist_dir = settings.frontend_dist_path
if dist_dir.exists() and (dist_dir / "index.html").exists():
    assets_dir = dist_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_spa_root():
        return FileResponse(dist_dir / "index.html")
