"""Validation helpers for loaded PathSnap trips."""

from collections.abc import Mapping

import numpy as np
import pandas as pd


_REQUIRED_COLUMNS = {
    "track": ("longitude", "latitude", "time"),
    "nodes": ("longitude", "latitude"),
    "arcs": ("from_node", "to_node"),
    "route": ("arc_id",),
}
_COORDINATE_RANGES = {
    "longitude": (-180, 180),
    "latitude": (-90, 90),
}
_INTEGER_COLUMNS = {
    "arcs": ("from_node", "to_node"),
    "route": ("arc_id",),
}


def validate_trip(trip: Mapping[str, object]) -> list[str]:
    """Return validation issues found in one loaded trip; an empty list means valid."""
    issues = []
    numeric_columns: dict[str, dict[str, pd.Series]] = {}

    for table_name, columns in _REQUIRED_COLUMNS.items():
        frame = trip.get(table_name)
        if not isinstance(frame, pd.DataFrame):
            issues.append(f"Missing or invalid '{table_name}' table.")
            continue
        if frame.empty:
            issues.append(f"'{table_name}' table is empty.")
            continue

        missing_columns = [column for column in columns if column not in frame]
        if missing_columns:
            issues.append(
                f"'{table_name}' is missing columns: {', '.join(missing_columns)}."
            )
            continue

        numeric_columns[table_name] = {}
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = values.isna() | ~np.isfinite(values)
            if invalid.any():
                issues.append(
                    f"'{table_name}.{column}' has {int(invalid.sum())} "
                    "non-numeric or non-finite value(s)."
                )
            numeric_columns[table_name][column] = values

            if column in _COORDINATE_RANGES:
                minimum, maximum = _COORDINATE_RANGES[column]
                outside_range = values.notna() & ~values.between(minimum, maximum)
                if outside_range.any():
                    issues.append(
                        f"'{table_name}.{column}' has "
                        f"{int(outside_range.sum())} value(s) outside "
                        f"[{minimum}, {maximum}]."
                    )

    track_times = numeric_columns.get("track", {}).get("time")
    if track_times is not None:
        finite_times = track_times[np.isfinite(track_times)]
        if finite_times.diff().dropna().lt(0).any():
            issues.append("'track.time' must be in non-decreasing order.")

    for table_name, columns in _INTEGER_COLUMNS.items():
        for column in columns:
            values = numeric_columns.get(table_name, {}).get(column)
            if values is not None:
                fractional = values.notna() & values.mod(1).ne(0)
                if fractional.any():
                    issues.append(
                        f"'{table_name}.{column}' has "
                        f"{int(fractional.sum())} non-integer ID value(s)."
                    )

    nodes = trip.get("nodes")
    arcs = trip.get("arcs")
    if isinstance(nodes, pd.DataFrame) and isinstance(arcs, pd.DataFrame):
        for column in _INTEGER_COLUMNS["arcs"]:
            values = numeric_columns.get("arcs", {}).get(column)
            if values is not None:
                out_of_bounds = values.notna() & ~values.between(0, len(nodes) - 1)
                if out_of_bounds.any():
                    issues.append(
                        f"'arcs.{column}' has {int(out_of_bounds.sum())} "
                        "node ID value(s) outside the nodes table."
                    )

    route = trip.get("route")
    if isinstance(route, pd.DataFrame) and isinstance(arcs, pd.DataFrame):
        arc_ids = numeric_columns.get("route", {}).get("arc_id")
        if arc_ids is not None:
            out_of_bounds = arc_ids.notna() & ~arc_ids.between(0, len(arcs) - 1)
            if out_of_bounds.any():
                issues.append(
                    f"'route.arc_id' has {int(out_of_bounds.sum())} "
                    "arc ID value(s) outside the arcs table."
                )

    return issues


def validate_trips(
    trips: Mapping[str, Mapping[str, object]],
) -> dict[str, list[str]]:
    """Return issues keyed by trip ID; valid trips are omitted from the result."""
    return {
        trip_id: issues
        for trip_id, trip in trips.items()
        if (issues := validate_trip(trip))
    }
