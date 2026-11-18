import numpy as np
import pandas as pd

from src.models.hmm_viterbi import decode_trip, score_candidates


def test_viterbi_prefers_connected_path_over_independent_top_pick():
    gps = pd.DataFrame({"time": [0.0, 1.0, 2.0]})
    scored = pd.DataFrame({
        "obs_idx": [0, 1, 1, 2],
        "arc_id": [10, 20, 99, 30],
        "mlp_score": [0.95, 0.65, 0.90, 0.95],
    })
    edges = pd.DataFrame({
        "obs_idx": [0, 0, 1],
        "arc_from": [10, 10, 20],
        "arc_to": [20, 99, 30],
        "route_dist": [10.0, np.inf, 10.0],
        "gc_dist": [10.0, 10.0, 10.0],
    })

    out = decode_trip(gps, scored, edges)

    assert out["mlp_arc"].tolist() == [10, 99, 30]
    assert out["viterbi_arc"].tolist() == [10, 20, 30]
    assert out["segment_id"].tolist() == [0, 0, 0]


def test_decoder_marks_gaps_missing_candidates_and_invalid_transitions():
    gps = pd.DataFrame({"time": [0.0, 8.0, 9.0, 10.0, 11.0, 11.0]})
    scored = pd.DataFrame({
        "obs_idx": [0, 1, 3, 4, 5],
        "arc_id": [10, 20, 30, 40, 50],
        "mlp_score": [0.9] * 5,
    })
    # Observation 3 -> 4 would require 100 m/s, so the path restarts.
    edges = pd.DataFrame({
        "obs_idx": [3],
        "arc_from": [30],
        "arc_to": [40],
        "route_dist": [100.0],
        "gc_dist": [10.0],
    })

    out = decode_trip(gps, scored, edges)

    assert out["break_reason"].tolist() == [
        "trip_start", "gps_gap", "no_candidates", "after_no_candidates",
        "no_feasible_transition", "nonpositive_dt",
    ]
    assert pd.isna(out.at[2, "viterbi_arc"])
    assert out["viterbi_arc"].dropna().tolist() == [10, 20, 30, 40, 50]


def test_scoring_uses_artifact_feature_order_and_positive_class():
    class FakePipeline:
        classes_ = np.array([0, 1])

        def predict_proba(self, x):
            assert list(x.columns) == ["heading_cos", "perp_dist"]
            return np.array([[0.2, 0.8]])

    candidates = pd.DataFrame({
        "obs_idx": [0], "arc_id": [10],
        "perp_dist": [4.0], "heading_cos": [0.9], "label": [1],
    })
    artifact = {
        "pipeline": FakePipeline(),
        "features": ["heading_cos", "perp_dist"],
    }

    scored = score_candidates(candidates, artifact)

    assert scored.to_dict("records") == [
        {"obs_idx": 0, "arc_id": 10, "mlp_score": 0.8}
    ]