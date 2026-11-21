from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from active_learning.pseudo_label import count_yolo_seg_classes, parse_yolo_seg, validate_yolo_seg
from active_learning.sampler import ActiveLearningSampler, _dir_size_bytes
from models import (
    ActiveLearningDecisionRequest,
    ActiveLearningDecisionResponse,
    ActiveLearningQueueResponse,
    ActiveLearningStats,
    ID_TO_STORM_CLASS,
)


def create_active_learning_router(sampler: ActiveLearningSampler) -> APIRouter:
    router = APIRouter(prefix="/api/active-learning", tags=["active-learning"])

    @router.get("/stats", response_model=ActiveLearningStats)
    async def get_stats() -> ActiveLearningStats:
        return sampler.get_stats()

    @router.get("/queue", response_model=ActiveLearningQueueResponse)
    async def get_queue(
        status: Literal["pending", "approved", "rejected", "all"] = Query("pending"),
        limit: int = Query(20, ge=1, le=200),
    ) -> ActiveLearningQueueResponse:
        rows = sampler.db.list_al_samples(status=status, limit=limit)
        samples = [sampler.hydrate_sample(r) for r in rows]
        return ActiveLearningQueueResponse(
            status_filter=status,
            count=len(samples),
            stats=sampler.get_stats(),
            samples=samples,
        )

    @router.get("/{sample_id}/frame")
    async def get_sample_frame(sample_id: str) -> FileResponse:
        if "/" in sample_id or "\\" in sample_id or ".." in sample_id:
            raise HTTPException(status_code=400, detail="Invalid sample_id")
        row = sampler.db.get_al_sample(sample_id)
        if not row:
            raise HTTPException(status_code=404, detail="Sample not found")

        frame_path = (Path(row["dir_path"]) / "frame.png").resolve()
        al_root = sampler.root_dir.resolve()
        if al_root not in frame_path.parents or not frame_path.exists():
            raise HTTPException(status_code=404, detail="Sample frame image not found")

        return FileResponse(
            frame_path,
            media_type="image/png",
            headers={"Cache-Control": "no-cache"},
        )

    @router.post("/{sample_id}/decision", response_model=ActiveLearningDecisionResponse)
    async def submit_decision(
        sample_id: str,
        body: ActiveLearningDecisionRequest,
    ) -> ActiveLearningDecisionResponse:
        row = sampler.db.get_al_sample(sample_id)
        if not row:
            raise HTTPException(status_code=404, detail="Sample not found")

        sample_dir = Path(row["dir_path"])
        label_path = sample_dir / "label.txt"
        meta_path = sample_dir / "meta.json"

        detection_count = int(row["detection_count"] or 0)
        label_updated = False
        if body.updated_label_txt is not None:
            valid, err = validate_yolo_seg(body.updated_label_txt)
            if not valid:
                raise HTTPException(status_code=422, detail=f"Invalid YOLOv8-seg label.txt: {err}")
            normalized_txt = body.updated_label_txt.strip()
            if normalized_txt:
                normalized_txt += "\n"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(normalized_txt, encoding="utf-8")
            detection_count = len(parse_yolo_seg(normalized_txt))
            label_updated = True

        reviewed_at = datetime.now(timezone.utc).isoformat()
        try:
            meta = json.loads(row["meta_json"]) if row.get("meta_json") else {}
        except Exception:
            meta = {}
        meta["status"] = body.status
        meta["reviewed_at"] = reviewed_at
        meta["detection_count"] = detection_count
        if label_updated:
            meta["human_edited_label"] = True

        meta_json = json.dumps(meta, indent=2)
        if sample_dir.exists():
            meta_path.write_text(meta_json, encoding="utf-8")

        size_bytes = _dir_size_bytes(sample_dir) if sample_dir.exists() else int(row["size_bytes"] or 0)
        updated_row = sampler.db.update_al_decision(
            sample_id=sample_id,
            status=body.status,
            reviewed_at=reviewed_at,
            detection_count=detection_count,
            size_bytes=size_bytes,
            meta_json=meta_json,
        )
        if updated_row is None:
            raise HTTPException(status_code=404, detail="Sample not found")

        return ActiveLearningDecisionResponse(
            sample=sampler.hydrate_sample(updated_row),
            stats=sampler.get_stats(),
        )

    @router.get("/export")
    async def export_approved_batch() -> StreamingResponse:
        approved_rows = sampler.db.list_approved_al_samples()
        if not approved_rows:
            raise HTTPException(
                status_code=400,
                detail="No approved active-learning samples available to export.",
            )

        now_utc = datetime.now(timezone.utc)
        date_slug = now_utc.strftime("%Y%m%d")
        zip_filename = f"active_learning_batch_{date_slug}.zip"

        buf = io.BytesIO()
        manifest_samples: list[dict] = []
        retained_frames: list[dict] = []

        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for row in approved_rows:
                sample_id = row["id"]
                sample_dir = Path(row["dir_path"])
                frame_path = sample_dir / "frame.png"
                label_path = sample_dir / "label.txt"
                if not frame_path.exists():
                    continue

                label_txt = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
                img_filename = f"{sample_id}.png"
                lbl_filename = f"{sample_id}.txt"

                zf.write(frame_path, arcname=f"images/{img_filename}")
                zf.writestr(f"labels/{lbl_filename}", label_txt)

                try:
                    meta = json.loads(row["meta_json"]) if row.get("meta_json") else {}
                except Exception:
                    meta = {}

                class_counts = count_yolo_seg_classes(label_txt)
                manifest_samples.append(
                    {
                        "id": sample_id,
                        "image": f"images/{img_filename}",
                        "label": f"labels/{lbl_filename}",
                        "frame_timestamp": row["frame_timestamp"],
                        "captured_at": row["captured_at"],
                        "reviewed_at": row.get("reviewed_at"),
                        "trigger_type": row["trigger_type"],
                        "trigger_detail": row["trigger_detail"],
                        "model_filename": row["model_filename"],
                        "class_counts": class_counts,
                        "meta": meta,
                    }
                )
                retained_frames.append(
                    {
                        "filename": img_filename,
                        "stem": sample_id,
                        "timestamp": row["frame_timestamp"],
                        "class_counts": class_counts,
                    }
                )

            if not manifest_samples:
                raise HTTPException(
                    status_code=400,
                    detail="Approved samples exist in DB but their frame.png files are missing on disk.",
                )

            manifest = {
                "schema_version": "1.0",
                "format": "yolov8-seg",
                "ml_pipeline_target": "3_prepare_dataset.py",
                "exported_at": now_utc.isoformat(),
                "batch_date": date_slug,
                "sample_count": len(manifest_samples),
                "image_size": [640, 640],
                "class_names": {str(k): v for k, v in sorted(ID_TO_STORM_CLASS.items())},
                "images_dir": "images",
                "labels_dir": "labels",
                "retained_frames": retained_frames,
                "samples": manifest_samples,
            }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))

        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{zip_filename}"',
                "Cache-Control": "no-store",
            },
        )

    return router
