"""Decode a trip using MLP candidate scores and directed road transitions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DecoderConfig:
    beta_m: float = 30.0
    gap_s: float = 5.0
    max_speed_mps: float = 40.0


def score_candidates(candidates, artifact, prepare=None):
    """Use the {'pipeline': ..., 'features': ...} bundle from tune_mlp.py.

    prepare, if supplied, must be the same row-preserving input preparation
    used in training outside the saved pipeline.
    """
    features = list(artifact["features"])
    forbidden = {"label", "ref_arc", "ref_dist", "ref_in_candidates"}
    if forbidden.intersection(features):
        raise ValueError("The model feature list contains reference-label columns.")

    out = candidates[["obs_idx", "arc_id"]].copy()
    if candidates.empty:
        out["mlp_score"] = np.array([], dtype=float)
        return out

    x = candidates[features].copy()
    if prepare is not None:
        x = prepare(x)

    model = artifact["pipeline"]
    positive = np.flatnonzero(np.asarray(model.classes_) == 1)
    if len(positive) != 1:
        raise ValueError("Expected the fitted MLP to have positive class 1.")

    out["mlp_score"] = model.predict_proba(x)[:, positive[0]]
    return out


def _transition_matrix(previous, current, edges, dt, config):
    """Return a matrix of log transition scores; -inf means disallowed."""
    matrix = np.full((len(previous), len(current)), -np.inf)
    prev_index = {arc: i for i, arc in enumerate(previous)}
    curr_index = {arc: j for j, arc in enumerate(current)}

    if edges is None:
        return matrix

    for edge in edges.itertuples(index=False):
        i = prev_index.get(edge.arc_from)
        j = curr_index.get(edge.arc_to)
        if i is None or j is None:
            continue

        road_m = edge.route_dist
        gps_m = edge.gc_dist  # GPS step distance from network_features.py
        if not (np.isfinite(road_m) and np.isfinite(gps_m)):
            continue
        if road_m < 0 or gps_m < 0 or road_m / dt > config.max_speed_mps:
            continue

        score = -abs(road_m - gps_m) / config.beta_m
        matrix[i, j] = max(matrix[i, j], score)

    return matrix


def decode_trip(gps, scored, transitions, config=DecoderConfig()):
    """Decode one full trip in cleaned GPS row order.

    gps: one row per observation with `time`; obs_idx is its zero-based position.
    scored: obs_idx, arc_id, mlp_score.
    transitions: obs_idx, arc_from, arc_to, route_dist, gc_dist.
        Transition obs_idx=i means a move from observation i to i+1.
    """
    settings = (config.beta_m, config.gap_s, config.max_speed_mps)
    if not all(np.isfinite(v) and v > 0 for v in settings):
        raise ValueError("Decoder settings must be finite and positive.")

    times = gps["time"].to_numpy(dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError("Pass cleaned GPS observations in timestamp order.")

    n = len(times)
    if scored.duplicated(["obs_idx", "arc_id"]).any():
        raise ValueError("Duplicate candidate arc for an observation.")

    obs_ids = scored["obs_idx"].to_numpy(dtype=float)
    if not (
        np.isfinite(obs_ids).all()
        and ((obs_ids >= 0) & (obs_ids < n) & (obs_ids % 1 == 0)).all()
    ):
        raise ValueError("Candidate obs_idx does not match the GPS row positions.")

    probs = scored["mlp_score"].to_numpy(dtype=float)
    if not (np.isfinite(probs).all() and ((probs >= 0) & (probs <= 1)).all()):
        raise ValueError("Expected finite MLP scores in [0, 1].")

    groups = {
        int(i): rows.sort_values("arc_id")
        for i, rows in scored.groupby("obs_idx", sort=False)
    }
    # The original transition generator can return an empty DataFrame
    # with no columns if there are no consecutive candidate pairs.
    edge_groups = (
        {}
        if transitions.empty
        else dict(tuple(transitions.groupby("obs_idx", sort=False)))
    )

    out = pd.DataFrame({
        "obs_idx": np.arange(n),
        "time": times,
        "mlp_arc": pd.array([pd.NA] * n, dtype="Int64"),
        "viterbi_arc": pd.array([pd.NA] * n, dtype="Int64"),
        "segment_id": pd.array([pd.NA] * n, dtype="Int64"),
        "break_reason": pd.array([pd.NA] * n, dtype="string"),
    })

    # Each history entry holds (obs_idx, candidate arc IDs, backpointers).
    history = []
    best = None
    segment = -1

    def finish_segment():
        if not history:
            return
        state = int(np.argmax(best))
        for i, arcs, back in reversed(history):
            out.at[i, "viterbi_arc"] = arcs[state]
            state = int(back[state])
        history.clear()

    for i in range(n):
        rows = groups.get(i)
        if rows is None:
            finish_segment()
            out.at[i, "break_reason"] = "no_candidates"
            continue

        arcs = rows["arc_id"].to_numpy(dtype=np.int64)
        p = rows["mlp_score"].to_numpy(dtype=float)
        out.at[i, "mlp_arc"] = arcs[np.argmax(p)]
        emission = np.log(np.clip(p, 1e-12, 1.0))

        reason = None
        if not history:
            reason = "trip_start" if i == 0 else "after_no_candidates"
        else:
            prev_i, prev_arcs, _ = history[-1]
            dt = times[i] - times[prev_i]
            if dt <= 0:
                reason = "nonpositive_dt"
            elif dt > config.gap_s:
                reason = "gps_gap"
            else:
                transition = _transition_matrix(
                    prev_arcs, arcs, edge_groups.get(prev_i), dt, config
                )
                # For each current candidate, pick its best predecessor.
                combined = best[:, None] + transition
                back = np.argmax(combined, axis=0)
                next_best = combined[back, np.arange(len(arcs))] + emission
                if np.isfinite(next_best).any():
                    best = next_best
                else:
                    reason = "no_feasible_transition"

        if reason is not None:
            finish_segment()
            segment += 1
            best = emission
            back = np.full(len(arcs), -1, dtype=int)
            out.at[i, "break_reason"] = reason

        history.append((i, arcs, back))
        out.at[i, "segment_id"] = segment

    finish_segment()
    return out