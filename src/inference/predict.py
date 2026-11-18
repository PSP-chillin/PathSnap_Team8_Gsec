"""End-to-end prediction for one trip, plus server-side scoring.

predict_trip() uses only the track, nodes and arcs. score_prediction() is the
only place the reference route is read, and its result is reduced to metrics
before anything leaves the server.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.labels import align_track_to_route
from src.hmm.viterbi import HMMConfig, Observation, RoadGraph, collapse, continuity, fill_route, viterbi_decode
from src.inference.features import (
    TripCandidates,
    build_trip_candidates,
    clean_track,
    feature_matrix,
    read_inputs,
    read_reference_route,
)


@dataclass
class TripPrediction:
    trip_id: str
    gps: pd.DataFrame                 # cleaned track (longitude, latitude, time, orig_idx, ...)
    nodes: pd.DataFrame
    trip: TripCandidates
    graph: RoadGraph
    hmm_arcs: np.ndarray              # decoded arc per observation (-1 = none)
    classifier_arcs: np.ndarray       # highest-probability candidate per observation (-1 = none)
    route: list[int]                  # decoded route, gaps joined through the network
    chains: int
    inference_ms: float
    candidate_sets: list[set] = field(default_factory=list)


def _positive_index(model) -> int:
    classes = list(getattr(model, "classes_", [0, 1]))
    return classes.index(1) if 1 in classes else len(classes) - 1


def predict_trip(trip_dir: str | Path, model, config: HMMConfig = HMMConfig()) -> TripPrediction:
    started = time.perf_counter()
    trip_dir = Path(trip_dir)
    inputs = read_inputs(trip_dir)
    track = clean_track(inputs["track"])
    trip = build_trip_candidates(track, inputs["nodes"], inputs["arcs"])

    arc_table = trip.geometry.arc_table
    graph = RoadGraph(arc_table["from_node"].to_numpy(), arc_table["to_node"].to_numpy(),
                      arc_table["seg_length"].to_numpy())

    n_obs = len(trip.gps)
    observations: list[Observation | None] = [None] * n_obs
    classifier_arcs = np.full(n_obs, -1, dtype=int)
    candidate_sets: list[set] = [set() for _ in range(n_obs)]

    table = trip.candidates
    if not table.empty:
        proba = model.predict_proba(feature_matrix(table))[:, _positive_index(model)]
        table = table.assign(prob=proba).sort_values(["obs_idx", "perp_dist"], kind="stable")
        obs_idx = table["obs_idx"].to_numpy(dtype=int)
        arcs = table["arc_id"].to_numpy(dtype=int)
        along = table["dist_along"].to_numpy(dtype=float)
        probs = table["prob"].to_numpy(dtype=float)
        starts = np.flatnonzero(np.r_[True, obs_idx[1:] != obs_idx[:-1]])
        ends = np.r_[starts[1:], len(obs_idx)]
        for s, e in zip(starts, ends):
            t = int(obs_idx[s])
            p = np.clip(probs[s:e], config.min_prob, 1.0)
            observations[t] = Observation(arcs=arcs[s:e], along=along[s:e], log_emit=np.log(p))
            classifier_arcs[t] = int(arcs[s:e][np.argmax(p)])
            candidate_sets[t] = set(arcs[s:e].tolist())

    times = trip.gps["time"].to_numpy(dtype=float)
    hmm_arcs, chains = viterbi_decode(observations, trip.px, trip.py, times, graph, config)
    route = fill_route(collapse(hmm_arcs), graph, config)

    return TripPrediction(
        trip_id=trip_dir.name,
        gps=trip.gps,
        nodes=inputs["nodes"],
        trip=trip,
        graph=graph,
        hmm_arcs=hmm_arcs,
        classifier_arcs=classifier_arcs,
        route=route,
        chains=chains,
        inference_ms=(time.perf_counter() - started) * 1000.0,
        candidate_sets=candidate_sets,
    )


def _canonical_ids(graph: RoadGraph) -> np.ndarray:
    """Map duplicate arcs (same from/to nodes, identical geometry) to one id for scoring."""
    pairs = pd.DataFrame({"f": graph.from_nodes, "t": graph.to_nodes})
    return pairs.groupby(["f", "t"]).ngroup().to_numpy()


def score_prediction(prediction: TripPrediction, trip_dir: str | Path) -> dict:
    """Metrics against the reference route. Returns numbers only, never the route itself."""
    graph = prediction.graph
    metrics = {
        "n_points": int(len(prediction.gps)),
        "n_predicted_arcs": int(len(prediction.route)),
        "route_continuity": continuity(collapse(prediction.hmm_arcs), graph),
        "classifier_route_continuity": continuity(collapse(prediction.classifier_arcs), graph),
        "hmm_chains": int(prediction.chains),
    }
    route = read_reference_route(trip_dir)
    if route is None:
        metrics["note"] = "No reference route on the server for this trip, so accuracy was not computed."
        return metrics
    if route.min() < 0 or route.max() >= len(graph.from_nodes):
        metrics["note"] = "Reference route refers to arcs that don't exist; accuracy not computed."
        return metrics

    canon = _canonical_ids(graph)
    ref_arcs, _, _ = align_track_to_route(prediction.trip.px, prediction.trip.py, route, prediction.trip.geometry)
    ref_c = canon[np.asarray(ref_arcs, dtype=int)]

    def point_accuracy(pred_arcs: np.ndarray) -> float:
        ok = pred_arcs >= 0
        hits = np.zeros(len(pred_arcs), dtype=bool)
        hits[ok] = canon[pred_arcs[ok]] == ref_c[ok]
        return float(hits.mean())

    coverage = np.mean([
        bool({int(canon[a]) for a in cands} & {int(ref_c[t])}) for t, cands in enumerate(prediction.candidate_sets)
    ])

    lengths = graph.lengths
    pred_set = {int(canon[a]): lengths[a] for a in prediction.route}
    ref_set = {int(canon[a]): lengths[a] for a in route}
    shared = sum(length for key, length in pred_set.items() if key in ref_set)
    pred_total, ref_total = sum(pred_set.values()), sum(ref_set.values())

    metrics.update({
        "accuracy": point_accuracy(prediction.hmm_arcs),
        "classifier_only_accuracy": point_accuracy(prediction.classifier_arcs),
        "candidate_coverage": float(coverage),
        "route_precision": float(shared / pred_total) if pred_total > 0 else 0.0,
        "route_recall": float(shared / ref_total) if ref_total > 0 else 0.0,
    })
    return metrics


def to_response(prediction: TripPrediction, metrics: dict) -> dict:
    """JSON body for POST /predict (see ui/README.md). Coordinates as {lat, lon}."""
    gps = prediction.gps
    nodes = prediction.nodes
    lat = nodes["latitude"].to_numpy(dtype=float)
    lon = nodes["longitude"].to_numpy(dtype=float)
    graph = prediction.graph
    segments = [
        [
            {"lat": float(lat[graph.from_nodes[a]]), "lon": float(lon[graph.from_nodes[a]])},
            {"lat": float(lat[graph.to_nodes[a]]), "lon": float(lon[graph.to_nodes[a]])},
        ]
        for a in prediction.route
    ]
    return {
        "trip_id": prediction.trip_id,
        "gps_points": [
            {"lat": float(la), "lon": float(lo)}
            for la, lo in zip(gps["latitude"].to_numpy(dtype=float), gps["longitude"].to_numpy(dtype=float))
        ],
        "predicted_route": segments,
        "predicted_arc_ids": [int(a) for a in prediction.route],
        "metrics": metrics,
        "inference_time_ms": round(prediction.inference_ms),
    }
