"""Basic cleaning for loaded PathSnap GPS tracks."""

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd


_TRACK_COLUMNS = ("longitude", "latitude", "time")
_REQUIRED_TABLE_COLUMNS = {
    "track": _TRACK_COLUMNS,
    "nodes": ("longitude", "latitude"),
    "arcs": ("from_node", "to_node"),
    "route": ("arc_id",),
}


def clean_trip(trip: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Remove invalid/duplicate GPS observations and sort by time.

    The nodes, arcs, and route tables are copied unchanged.
    """
    if any(not isinstance(frame, pd.DataFrame) for frame in trip.values()):
        raise ValueError("Every trip table must be a pandas DataFrame.")

    for table_name, columns in _REQUIRED_TABLE_COLUMNS.items():
        frame = trip.get(table_name)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"Trip is missing the '{table_name}' DataFrame.")
        if frame.empty:
            raise ValueError(f"Trip has an empty '{table_name}' table.")
        missing_columns = [column for column in columns if column not in frame]
        if missing_columns:
            raise ValueError(
                f"'{table_name}' is missing columns: {', '.join(missing_columns)}."
            )

    track = trip["track"].copy()

    for column in _TRACK_COLUMNS:
        track[column] = pd.to_numeric(track[column], errors="coerce")

    coordinates_and_time = track[list(_TRACK_COLUMNS)].to_numpy(dtype=float)
    valid = np.isfinite(coordinates_and_time).all(axis=1)
    valid &= track["longitude"].between(-180, 180).to_numpy()
    valid &= track["latitude"].between(-90, 90).to_numpy()

    invalid_count = int((~valid).sum())
    track = track.loc[valid].copy()
    before_deduplication = len(track)
    track = track.drop_duplicates(
        subset=list(_TRACK_COLUMNS),
        keep="first",
    )
    duplicate_count = before_deduplication - len(track)
    track = track.sort_values("time", kind="stable").reset_index(drop=True)

    if track.empty:
        raise ValueError("No valid GPS observations remain after cleaning.")
    if invalid_count or duplicate_count:
        warnings.warn(
            f"Removed {invalid_count} invalid and {duplicate_count} duplicate "
            "GPS observation(s).",
            UserWarning,
            stacklevel=2,
        )

    cleaned = {name: frame.copy() for name, frame in trip.items()}
    cleaned["track"] = track
    return cleaned


def clean_trips(
    trips: Mapping[str, Mapping[str, pd.DataFrame]],
) -> dict[str, dict[str, pd.DataFrame]]:
    """Clean trips, dropping any trip with an empty required table."""
    cleaned: dict[str, dict[str, pd.DataFrame]] = {}
    for trip_id, trip in trips.items():
        empty_tables = [
            table_name
            for table_name in _REQUIRED_TABLE_COLUMNS
            if isinstance(trip.get(table_name), pd.DataFrame)
            and trip[table_name].empty
        ]
        if empty_tables:
            warnings.warn(
                f"Dropping trip {trip_id}: empty table(s): "
                f"{', '.join(empty_tables)}.",
                UserWarning,
                stacklevel=2,
            )
            continue
        try:
            cleaned[trip_id] = clean_trip(trip)
        except ValueError as exc:
            raise ValueError(f"Cannot clean trip {trip_id}: {exc}") from exc

    if trips and not cleaned:
        raise ValueError("No trips remain after dropping trips with empty tables.")
    return cleaned
