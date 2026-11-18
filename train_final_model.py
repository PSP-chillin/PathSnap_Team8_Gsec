"""Train the deployed candidate classifier and save it to artifacts/.

Uses the MLP, the model selected by student-test macro-F1 in
results/model_reports, with the same configuration (src.models.mlp.build_mlp)
and the same features as report.py. The final model is fitted on more evenly
spaced observations per trip than the comparison run, from the 80 training
trips only. Student-test trips are used only to report held-out metrics.

Run from the repository root:
    python train_final_model.py
"""

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
import yaml
from sklearn.metrics import f1_score, log_loss

from src.data.cleaning import clean_trip_with_stats
from src.data.loaders import load_trip
from src.data.validation import validate_trip
from src.features.labels import align_track_to_route, label_candidates
from src.inference.features import FEATURE_COLUMNS, build_trip_candidates, feature_matrix
from src.models.mlp import build_mlp

ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
REQUIRED = ("track", "nodes", "arcs", "route")


def trip_rows(trip_dir: Path, max_obs: int | None):
    """Labelled feature rows for one trip, or None if the trip can't be used."""
    trip = load_trip(trip_dir)
    if any(trip[name].empty for name in REQUIRED):
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trip, _ = clean_trip_with_stats(trip)
    if validate_trip(trip):
        return None
    tc = build_trip_candidates(trip["track"], trip["nodes"], trip["arcs"])
    if tc.candidates.empty:
        return None
    ref, _, _ = align_track_to_route(tc.px, tc.py, trip["route"]["arc_id"].to_numpy(), tc.geometry)
    labeled = label_candidates(tc.candidates, ref)
    n_obs = len(tc.gps)
    if max_obs is not None and n_obs > max_obs:
        keep = np.linspace(0, n_obs - 1, max_obs, dtype=int)
        labeled = labeled[labeled["obs_idx"].isin(keep)]
    if labeled.empty:
        return None
    return feature_matrix(labeled), labeled["label"].to_numpy(dtype=int), trip_dir.name


def build_split(split_dir: Path, max_obs: int | None, workers: int):
    trip_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
    X_parts, y_parts, used, skipped = [], [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for trip_dir, result in zip(trip_dirs, pool.map(trip_rows, trip_dirs, [max_obs] * len(trip_dirs))):
            if result is None:
                skipped.append(trip_dir.name)
                continue
            X_parts.append(result[0])
            y_parts.append(result[1])
            used.append(result[2])
    return np.vstack(X_parts), np.concatenate(y_parts), used, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-obs-per-trip", type=int, default=150)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "final_model.joblib")
    args = parser.parse_args()

    import report  # repo-root report.py: guard against the feature list drifting apart

    if tuple(report.FEATURE_COLUMNS) != FEATURE_COLUMNS:
        raise SystemExit("FEATURE_COLUMNS differ from report.py; update src/inference/features.py.")

    seed = int(CONFIG.get("random_seed", 42))
    train_dir = ROOT / CONFIG["paths"]["train_dir"]
    test_dir = ROOT / CONFIG["paths"]["student_test_dir"]

    started = time.perf_counter()
    print(f"Building training rows from {train_dir} ...", flush=True)
    X, y, used, skipped = build_split(train_dir, args.max_obs_per_trip, args.workers)
    print(f"  {len(used)} trips, {len(y):,} rows, positive rate {y.mean():.3f}, skipped {skipped}", flush=True)

    print(f"Building student-test rows from {test_dir} ...", flush=True)
    X_test, y_test, used_test, skipped_test = build_split(test_dir, args.max_obs_per_trip, args.workers)
    print(f"  {len(used_test)} trips, {len(y_test):,} rows", flush=True)

    model = build_mlp(random_state=seed)
    fit_started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ConvergenceWarning at max_iter=100, same as the comparison run
        model.fit(X, y)
    fit_seconds = time.perf_counter() - fit_started

    positive = list(model.classes_).index(1)
    proba = model.predict_proba(X_test)[:, positive]
    pred = (proba >= 0.5).astype(int)
    held_out = {
        "student_test_macro_f1": float(f1_score(y_test, pred, average="macro")),
        "student_test_log_loss": float(log_loss(y_test, model.predict_proba(X_test))),
    }
    print(f"  fit {fit_seconds:.1f}s | student-test macro-F1 {held_out['student_test_macro_f1']:.4f} "
          f"| log loss {held_out['student_test_log_loss']:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    metadata = {
        "model": "mlp",
        "selected_because": "highest student-test macro-F1 in results/model_reports/model_comparison.csv",
        "parameters": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in model.get_params().items()
                       if k.startswith("classifier__") and isinstance(v, (int, float, str, tuple, bool, type(None)))},
        "feature_columns": list(FEATURE_COLUMNS),
        "positive_class_index": positive,
        "max_obs_per_trip": args.max_obs_per_trip,
        "train_trips": used,
        "train_rows": int(len(y)),
        "train_positive_rate": float(y.mean()),
        "skipped_train_trips": skipped,
        "student_test_trips": used_test,
        "student_test_rows": int(len(y_test)),
        **held_out,
        "fit_seconds": round(fit_seconds, 2),
        "total_seconds": round(time.perf_counter() - started, 2),
        "random_state": seed,
        "sklearn_version": sklearn.__version__,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {args.out} and {args.out.with_suffix('.json').name} in {metadata['total_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
