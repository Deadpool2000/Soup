"""FastAPI application for Soup Web UI."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from pydantic import BaseModel as PydanticBaseModel

STATIC_DIR = Path(__file__).parent / "static"


class TrainRequest(PydanticBaseModel):
    """Request body for starting a training run."""
    config_yaml: str
    config_path: Optional[str] = None


class TrainStatus(PydanticBaseModel):
    """Current training process status."""
    running: bool
    pid: Optional[int] = None
    config_path: Optional[str] = None


class DataInspectRequest(PydanticBaseModel):
    """Request body for data inspection."""
    path: str
    limit: int = 50


# Global state for training process
_train_process: Optional[subprocess.Popen] = None
_train_config_path: Optional[str] = None
_train_lock = threading.Lock()


def create_app():
    """Create the Soup Web UI FastAPI application."""
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Soup Web UI", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Static files ---

    @app.get("/", response_class=HTMLResponse)
    def index():
        index_path = STATIC_DIR / "index.html"
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # --- Runs API ---

    @app.get("/api/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=500)):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            runs = tracker.list_runs(limit=limit)
            return {"runs": runs}
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            run = tracker.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            return run
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}/metrics")
    def get_run_metrics(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            run = tracker.get_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            metrics = tracker.get_metrics(run_id)
            return {"run_id": run_id, "metrics": metrics}
        finally:
            tracker.close()

    @app.delete("/api/runs/{run_id}")
    def delete_run(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            deleted = tracker.delete_run(run_id)
            if not deleted:
                raise HTTPException(status_code=404, detail="Run not found")
            return {"deleted": True, "run_id": run_id}
        finally:
            tracker.close()

    @app.get("/api/runs/{run_id}/eval")
    def get_run_eval(run_id: str):
        from soup_cli.experiment.tracker import ExperimentTracker

        tracker = ExperimentTracker()
        try:
            results = tracker.get_eval_results(run_id=run_id)
            return {"run_id": run_id, "eval_results": results}
        finally:
            tracker.close()

    # --- GPU / System Info ---

    @app.get("/api/system")
    def system_info():
        from soup_cli import __version__
        from soup_cli.utils.gpu import detect_device, get_gpu_info

        device, device_name = detect_device()
        gpu_info = get_gpu_info()
        return {
            "version": __version__,
            "device": device,
            "device_name": device_name,
            "gpu_info": gpu_info,
            "python_version": sys.version.split()[0],
        }

    # --- Templates ---

    @app.get("/api/templates")
    def list_templates():
        from soup_cli.config.schema import TEMPLATES

        return {"templates": {name: yaml_str for name, yaml_str in TEMPLATES.items()}}

    # --- Config Validation ---

    @app.post("/api/config/validate")
    def validate_config(body: dict):
        from soup_cli.config.loader import load_config_from_string

        yaml_str = body.get("yaml", "")
        if not yaml_str:
            raise HTTPException(status_code=400, detail="Empty config")
        try:
            config = load_config_from_string(yaml_str)
            return {"valid": True, "config": config.model_dump()}
        except Exception as exc:
            return {"valid": False, "error": str(exc)}

    # --- Training ---

    @app.post("/api/train/start")
    def start_training(req: TrainRequest):
        global _train_process, _train_config_path

        with _train_lock:
            if _train_process and _train_process.poll() is None:
                raise HTTPException(
                    status_code=409, detail="Training already in progress"
                )

            # Write config to temp file
            config_path = req.config_path or os.path.join(
                os.getcwd(), ".soup_ui_config.yaml"
            )
            with open(config_path, "w", encoding="utf-8") as fh:
                fh.write(req.config_yaml)

            _train_config_path = config_path
            _train_process = subprocess.Popen(
                [sys.executable, "-m", "soup_cli", "train", "--config", config_path, "--yes"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            return {"started": True, "pid": _train_process.pid, "config_path": config_path}

    @app.get("/api/train/status")
    def train_status():
        global _train_process
        with _train_lock:
            if _train_process is None:
                return TrainStatus(running=False)
            poll = _train_process.poll()
            if poll is None:
                return TrainStatus(
                    running=True,
                    pid=_train_process.pid,
                    config_path=_train_config_path,
                )
            return TrainStatus(running=False, pid=_train_process.pid)

    @app.post("/api/train/stop")
    def stop_training():
        global _train_process
        with _train_lock:
            if _train_process and _train_process.poll() is None:
                _train_process.terminate()
                return {"stopped": True}
            return {"stopped": False, "detail": "No training in progress"}

    # --- Data Inspection ---

    @app.post("/api/data/inspect")
    def inspect_data(req: DataInspectRequest):
        from soup_cli.data.loader import load_raw_data

        path = Path(req.path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")

        try:
            raw_data = load_raw_data(path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        total = len(raw_data)
        sample = raw_data[: req.limit]

        # Detect format
        from soup_cli.data.formats import detect_format

        fmt = detect_format(raw_data[:5]) if raw_data else "unknown"

        # Basic stats
        keys = set()
        for entry in sample:
            keys.update(entry.keys())

        return {
            "path": str(path),
            "total": total,
            "format": fmt,
            "keys": sorted(keys),
            "sample": sample,
        }

    # --- Health ---

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    return app
