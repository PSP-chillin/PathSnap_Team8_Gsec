"""Build features (+ training labels) for every trip in each labelled split."""
import sys
import warnings
from pathlib import Path

import pandas as pd

from src.data.loaders import load_trip          # change if your module path differs
from src.data.cleaning import clean_trip
from src.data.validation import validate_trip
from src.features.pipeline import build_trip_features
from src.features.labels import align_track_to_route, label_candidates

DATA = Path("data")
OUT = Path("features")
OUT.mkdir(exist_ok=True)

# output name -> folder under data/. Do NOT add the evaluator split.
SPLITS = {"train": "training", "student_test": "student_testing"}


def process_trip(trip_dir: Path):
    trip = load_trip(trip_dir)
    issues = validate_trip(trip)
    if issues:
        return None, issues
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trip = clean_trip(trip)
    gps, cand, trans, geom, px, py = build_trip_features(trip, k=10, radius_m=100.0)
    ref, _, dist = align_track_to_route(px, py, trip["route"]["arc_id"].to_numpy(), geom)
    cand = label_candidates(cand, ref)
    gps["ref_arc"], gps["ref_dist"] = ref, dist
    return (gps, cand, trans), []


def run_split(name: str, folder: str):
    acc = {"gps": [], "candidates": [], "transitions": []}
    skipped = {}
    trip_dirs = sorted(p for p in (DATA / folder).iterdir() if p.is_dir())
    for i, tdir in enumerate(trip_dirs, 1):
        tid = tdir.name
        try:
            result, issues = process_trip(tdir)
        except Exception as e:
            skipped[tid] = [repr(e)]
            print(f"FAILED {tid}: {e}", file=sys.stderr)
            continue
        if result is None:
            skipped[tid] = issues
            continue
        for key, df in zip(acc, result):
            df.insert(0, "trip_id", tid)
            acc[key].append(df)
        print(f"[{name}] {i}/{len(trip_dirs)} {tid}")

    for key, frames in acc.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(OUT / f"{name}_{key}.parquet")

    cand = pd.concat(acc["candidates"], ignore_index=True) if acc["candidates"] else None
    if cand is not None:
        per_obs = cand.groupby(["trip_id", "obs_idx"])["ref_in_candidates"].first()
        print(f"[{name}] trips ok={len(acc['gps'])} skipped={len(skipped)} "
              f"candidate_miss_rate={1 - per_obs.mean():.3%}")
    for tid, why in skipped.items():
        print(f"  skipped {tid}: {why}")


if __name__ == "__main__":
    for n, f in SPLITS.items():
        run_split(n, f)