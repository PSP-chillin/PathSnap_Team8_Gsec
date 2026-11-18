"""Per-trip candidate features shared by model training and the API.

This mirrors report.py's build_partition_dataset step for step (same feature
functions, same candidate settings, same NaN handling), so the deployed model
sees exactly the kind of feature rows it was selected and trained on.

Prediction never reads the .route file: `read_inputs` loads only the track,
nodes and arcs. The reference route is read separately, for scoring only.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders import _read_table
from src.features.candidate_features import Geometry, candidate_features, generate_candidates
from src.features.gps_features import add_gps_features
from src.features.temporal_features import add_temporal_features

# Must match report.py FEATURE_COLUMNS (checked in train_final_model.py).
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

# Same as generate_candidates' defaults used by report.py.
CANDIDATE_K = 10
CANDIDATE_RADIUS_M = 100.0

_INPUT_COLUMNS = {
    "track": ("longitude", "latitude", "time"),
    "nodes": ("longitude", "latitude"),
    "arcs": ("from_node", "to_node"),
}


def read_inputs(trip_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the prediction inputs (track, nodes, arcs). Never the route."""
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    tables = {}
    for suffix, columns in _INPUT_COLUMNS.items():
        path = trip_dir / f"{trip_id}.{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Trip file not found: {path.name}")
        tables[suffix] = _read_table(path, columns)
        if tables[suffix].empty:
            raise ValueError(f"Trip {trip_id} has an empty {suffix} file.")

    nodes, arcs = tables["nodes"], tables["arcs"]
    for column in ("from_node", "to_node"):
        ids = pd.to_numeric(arcs[column], errors="coerce")
        if ids.isna().any() or (ids < 0).any() or (ids >= len(nodes)).any():
            raise ValueError(f"Trip {trip_id}: arcs.{column} refers to nodes that don't exist.")
    if not np.isfinite(nodes[["longitude", "latitude"]].to_numpy(float)).all():
        raise ValueError(f"Trip {trip_id}: nodes contain non-numeric coordinates.")
    return tables


def read_reference_route(trip_dir: str | Path) -> np.ndarray | None:
    """Load the reference route for server-side scoring; None if not supplied."""
    trip_dir = Path(trip_dir)
    path = trip_dir / f"{trip_dir.name}.route"
    if not path.is_file():
        return None
    route = _read_table(path, ("arc_id",))
    if route.empty:
        return None
    return route["arc_id"].to_numpy(int)


def clean_track(track: pd.DataFrame) -> pd.DataFrame:
    """Same track cleaning as src.data.cleaning.clean_trip, without needing the route.

    Drops non-numeric, non-finite, out-of-range and duplicate observations and
    sorts by time. `orig_idx` keeps each point's row number in the .track file.
    """
    columns = ["longitude", "latitude", "time"]
    cleaned = track.copy()
    cleaned["orig_idx"] = np.arange(len(cleaned))
    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    valid = np.isfinite(cleaned[columns].to_numpy(dtype=float)).all(axis=1)
    valid &= cleaned["longitude"].between(-180, 180).to_numpy()
    valid &= cleaned["latitude"].between(-90, 90).to_numpy()
    cleaned = cleaned.loc[valid].drop_duplicates(subset=columns, keep="first")
    cleaned = cleaned.sort_values("time", kind="stable").reset_index(drop=True)
    if cleaned.empty:
        raise ValueError("No valid GPS observations remain after cleaning.")
    return cleaned


@dataclass
class TripCandidates:
    gps: pd.DataFrame          # cleaned track + GPS/temporal features, one row per observation
    geometry: Geometry         # projected (metre) road geometry
    px: np.ndarray             # projected GPS x (metres)
    py: np.ndarray             # projected GPS y (metres)
    candidates: pd.DataFrame   # one row per (obs_idx, arc_id) with candidate features


def build_trip_candidates(track: pd.DataFrame, nodes: pd.DataFrame, arcs: pd.DataFrame) -> TripCandidates:
    """Generate candidate road segments and their features for one cleaned track."""
    gps = add_temporal_features(add_gps_features(track))
    geometry = Geometry(nodes, arcs)
    px, py = geometry.project_track(gps)
    px, py = np.asarray(px, dtype=float), np.asarray(py, dtype=float)
    raw = generate_candidates(px, py, geometry, k=CANDIDATE_K, radius_m=CANDIDATE_RADIUS_M)
    if raw.empty:
        table = pd.DataFrame(columns=["obs_idx", "arc_id", *FEATURE_COLUMNS, "heading_diff"])
    else:
        table = candidate_features(gps, px, py, raw, geometry)
        table["heading_match_available"] = table["heading_diff"].notna().astype(float)
    return TripCandidates(gps=gps, geometry=geometry, px=px, py=py, candidates=table)


def feature_matrix(candidates: pd.DataFrame) -> np.ndarray:
    """Model input matrix, with the same NaN handling as report.py."""
    X = candidates.loc[:, list(FEATURE_COLUMNS)].copy()
    X["heading_cos"] = X["heading_cos"].fillna(0.0)
    X["is_opposite_dir"] = X["is_opposite_dir"].fillna(0.0)
    values = X.to_numpy(dtype=float)
    # report.py rejects non-finite rows; at prediction time we neutralise them instead of failing a trip.
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
