"""PathSnap prediction API (Laptop 1).

Run from the repository root:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

Endpoints (contract in ui/README.md):
    GET  /health   service status
    GET  /trips    trip IDs the evaluator can pick
    POST /predict  {"trip_id": "..."} -> GPS points, predicted route, metrics

Trips are served from PATHSNAP_TRIPS_DIR if set, otherwise paths.eval_dir in
configs/config.yaml. If the evaluator folder doesn't exist yet, student_test
is served so the system can be tested. Ground truth (.route) is read only by
the scoring step and never returned.
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.inference.predict import predict_trip, score_prediction, to_response  # noqa: E402

CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
MODEL_PATH = Path(os.environ.get("PATHSNAP_MODEL", ROOT / "artifacts" / "final_model.joblib"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pathsnap.api")


def _trip_dirs(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_dir() and (p / f"{p.name}.track").is_file())


def resolve_trips_dir() -> Path:
    override = os.environ.get("PATHSNAP_TRIPS_DIR")
    if override:
        return Path(override).resolve()
    eval_dir = ROOT / CONFIG["paths"]["eval_dir"]
    if _trip_dirs(eval_dir):
        return eval_dir
    fallback = ROOT / CONFIG["paths"]["student_test_dir"]
    log.warning("Evaluator trips not found at %s; serving %s for testing.", eval_dir, fallback)
    return fallback


STATE: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run `python train_final_model.py` first.")
    STATE["model"] = joblib.load(MODEL_PATH)
    meta_path = MODEL_PATH.with_suffix(".json")
    STATE["model_info"] = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    STATE["trips_dir"] = resolve_trips_dir()
    log.info("Loaded model %s (%s)", MODEL_PATH.name, STATE["model_info"].get("model", "unknown"))
    log.info("Serving %d trips from %s", len(_trip_dirs(STATE["trips_dir"])), STATE["trips_dir"])
    yield
    STATE.clear()


app = FastAPI(title="PathSnap map-matching API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": STATE.get("model_info", {}).get("model", "unknown"),
        "trips_dir": str(STATE.get("trips_dir", "")),
    }


@app.get("/trips")
def trips() -> dict:
    items = []
    for trip_dir in _trip_dirs(STATE["trips_dir"]):
        with open(trip_dir / f"{trip_dir.name}.track", "rb") as handle:
            n_points = sum(1 for line in handle if line.strip())
        items.append({"trip_id": trip_dir.name, "n_points": n_points})
    return {"trips": items}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    # Only trip folders that are actually listed can be requested (no path tricks).
    available = {p.name: p for p in _trip_dirs(STATE["trips_dir"])}
    trip_dir = available.get(request.trip_id.strip())
    if trip_dir is None:
        raise HTTPException(status_code=404, detail=f"Unknown trip ID: {request.trip_id}")

    started = time.perf_counter()
    log.info("predict trip=%s: started", trip_dir.name)
    try:
        prediction = predict_trip(trip_dir, STATE["model"])
        metrics = score_prediction(prediction, trip_dir)
    except (ValueError, FileNotFoundError) as exc:
        log.warning("predict trip=%s: rejected: %s", trip_dir.name, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # report unexpected failures to the UI instead of a bare 500
        log.exception("predict trip=%s: failed", trip_dir.name)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    response = to_response(prediction, metrics)
    log.info(
        "predict trip=%s: points=%d arcs=%d accuracy=%s classifier_only=%s continuity=%.3f in %.1fs",
        trip_dir.name,
        metrics["n_points"],
        metrics["n_predicted_arcs"],
        f"{metrics['accuracy']:.3f}" if "accuracy" in metrics else "n/a",
        f"{metrics['classifier_only_accuracy']:.3f}" if "classifier_only_accuracy" in metrics else "n/a",
        metrics["route_continuity"],
        time.perf_counter() - started,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
