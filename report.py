"""Build trip features and generate model comparison reports."""

import argparse
import importlib.util
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cleaning import clean_trip
from src.data.loaders import load_trip
from src.data.validation import validate_trip
from src.features.candidate_features import (
    Geometry,
    candidate_features,
    generate_candidates,
)
from src.features.gps_features import add_gps_features
from src.features.labels import align_track_to_route, label_candidates
from src.features.temporal_features import add_temporal_features
from src.models.registry import MODEL_BUILDERS
from src.models.report import generate_model_reports


ROOT = Path(__file__).resolve().parent
FEATURE_COLUMNS = (
    "perp_dist",
    "proj_frac",
    "dist_along",
    "dist_to_start",
    "dist_to_end",
    "signed_offset",
    "seg_length",
    "heading_cos",
    "is_opposite_dir",
    "is_endpoint_snap",
    "n_candidates",
    "dist_rank",
    "dist_minus_min",
    "heading_match_available",
)
EMPTY_REQUIRED_TABLES = ("track", "nodes", "arcs", "route")


@dataclass
class CandidateDataset:
    X: np.ndarray
    y: np.ndarray
    trip_groups: list[str]
    observation_ids: list[tuple[str, int]]
    candidate_coverage: float
    trip_count: int
    observation_count: int


def build_partition_dataset(partition_dir: str | Path) -> CandidateDataset:
    """Create labeled candidate rows from all trips in one data partition."""
    partition_dir = Path(partition_dir)
    if not partition_dir.is_dir():
        raise FileNotFoundError(f"Dataset partition not found: {partition_dir}")

    trip_dirs = sorted(path for path in partition_dir.iterdir() if path.is_dir())
    if not trip_dirs:
        raise ValueError(f"No trip directories found in {partition_dir}")

    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    trip_groups: list[str] = []
    observation_ids: list[tuple[str, int]] = []
    total_observations = 0
    matched_observations = 0
    used_trip_count = 0

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name
        trip = load_trip(trip_dir)
        empty_tables = [
            name for name in EMPTY_REQUIRED_TABLES if trip[name].empty
        ]
        if empty_tables:
            warnings.warn(
                f"Skipping trip {trip_id}: empty table(s): "
                f"{', '.join(empty_tables)}.",
                UserWarning,
                stacklevel=2,
            )
            continue

        trip = clean_trip(trip)
        issues = validate_trip(trip)
        if issues:
            raise ValueError(
                f"Trip {trip_id} failed validation after cleaning: "
                + "; ".join(issues)
            )

        gps = add_temporal_features(add_gps_features(trip["track"]))
        geometry = Geometry(trip["nodes"], trip["arcs"])
        px, py = geometry.project_track(gps)
        candidates = generate_candidates(px, py, geometry)
        total_observations += len(gps)
        if candidates.empty:
            warnings.warn(
                f"Skipping trip {trip_id}: no road candidates were generated.",
                UserWarning,
                stacklevel=2,
            )
            continue

        candidate_table = candidate_features(gps, px, py, candidates, geometry)
        reference_arcs, _, _ = align_track_to_route(
            px,
            py,
            trip["route"]["arc_id"].to_numpy(),
            geometry,
        )
        labeled = label_candidates(candidate_table, reference_arcs)
        matched_observations += int(
            labeled.groupby("obs_idx")["label"].max().sum()
        )

        labeled["heading_match_available"] = (
            labeled["heading_diff"].notna().astype(float)
        )
        X_trip = labeled.loc[:, list(FEATURE_COLUMNS)].copy()
        X_trip["heading_cos"] = X_trip["heading_cos"].fillna(0.0)
        X_trip["is_opposite_dir"] = X_trip["is_opposite_dir"].fillna(0.0)
        values = X_trip.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            invalid_columns = X_trip.columns[
                ~np.isfinite(values).all(axis=0)
            ].tolist()
            raise ValueError(
                f"Trip {trip_id} has non-finite model features in: "
                f"{', '.join(invalid_columns)}."
            )

        feature_rows.append(values)
        labels.append(labeled["label"].to_numpy(dtype=int))
        trip_groups.extend([trip_id] * len(labeled))
        observation_ids.extend(
            (trip_id, int(obs_idx)) for obs_idx in labeled["obs_idx"]
        )
        used_trip_count += 1

    if not feature_rows:
        raise ValueError(
            f"No labeled candidate rows could be built from {partition_dir}."
        )

    return CandidateDataset(
        X=np.concatenate(feature_rows, axis=0),
        y=np.concatenate(labels),
        trip_groups=trip_groups,
        observation_ids=observation_ids,
        candidate_coverage=(
            matched_observations / total_observations
            if total_observations
            else 0.0
        ),
        trip_count=used_trip_count,
        observation_count=total_observations,
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model comparison CSV and best-model Markdown."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "train",
        help="Training partition directory.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "student_test",
        help="Student-test partition directory used for model selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "model_reports",
        help="Directory for generated report files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_BUILDERS),
        default=None,
        help="Models to compare; defaults to installed models.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of trip-grouped cross-validation folds.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    model_names = args.models
    if model_names is None:
        model_names = list(MODEL_BUILDERS)
        if importlib.util.find_spec("lightgbm") is None:
            model_names.remove("lightgbm")
            print(
                "LightGBM is not installed in this Python environment; "
                "its report will be omitted. Install requirements to include it.",
                file=sys.stderr,
            )

    print(f"Building training features from {args.train_dir}...")
    training = build_partition_dataset(args.train_dir)
    print(f"Building validation features from {args.validation_dir}...")
    validation = build_partition_dataset(args.validation_dir)
    print(
        f"Training: {training.trip_count} trips, "
        f"{training.X.shape[0]} candidate rows, "
        f"{training.candidate_coverage:.1%} candidate coverage"
    )
    print(
        f"Validation: {validation.trip_count} trips, "
        f"{validation.X.shape[0]} candidate rows, "
        f"{validation.candidate_coverage:.1%} candidate coverage"
    )

    result = generate_model_reports(
        training.X,
        training.y,
        validation.X,
        validation.y,
        trip_groups=training.trip_groups,
        output_dir=args.output_dir,
        model_names=model_names,
        feature_names=FEATURE_COLUMNS,
        observation_ids=validation.observation_ids,
        train_candidate_coverage=training.candidate_coverage,
        validation_candidate_coverage=validation.candidate_coverage,
        n_splits=args.folds,
    )
    print(f"Best model by validation macro-F1: {result['best_model']}")
    print(f"Model comparison CSV: {result['csv_path']}")
    print(f"Best model Markdown: {result['markdown_path']}")


if __name__ == "__main__":
    main()
