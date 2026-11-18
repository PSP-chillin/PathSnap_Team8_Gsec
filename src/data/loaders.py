"""Load PathSnap trip files into pandas DataFrames."""

from pathlib import Path

import pandas as pd


_TABLE_COLUMNS = {
    "track": ("longitude", "latitude", "time"),
    "nodes": ("longitude", "latitude"),
    "arcs": ("from_node", "to_node"),
    "route": ("arc_id",),
}


def _read_table(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a headerless, comma- or whitespace-separated trip file."""
    frame = pd.read_csv(
        path,
        sep=r"\s+|,",
        engine="python",
        header=None,
        comment="#",
    )
    if frame.shape[1] != len(columns):
        raise ValueError(
            f"{path} has {frame.shape[1]} columns; expected {len(columns)}."
        )
    frame.columns = columns
    return frame


def load_trip(trip_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the .track, .nodes, .arcs, and .route files for one trip."""
    trip_dir = Path(trip_dir)
    if not trip_dir.is_dir():
        raise FileNotFoundError(f"Trip directory not found: {trip_dir}")

    trip_id = trip_dir.name
    tables = {}
    for suffix, columns in _TABLE_COLUMNS.items():
        path = trip_dir / f"{trip_id}.{suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"Trip file not found: {path}")
        tables[suffix] = _read_table(path, columns)

    return tables


def load_split(split_dir: str | Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Load every trip directory inside an existing dataset partition."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    trip_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
    if not trip_dirs:
        raise ValueError(f"No trip directories found in {split_dir}")

    return {trip_dir.name: load_trip(trip_dir) for trip_dir in trip_dirs}