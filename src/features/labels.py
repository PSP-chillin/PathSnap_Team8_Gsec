"""Training-only: per-observation reference arc by ordered alignment of the track to `.route`."""
import numpy as np

from .candidate_features import point_segment_geometry


def align_track_to_route(px, py, route_arcs, geom, clip_m=100.0):
    """Return (ref_arc_id, route_pos, dist) with route_pos non-decreasing."""
    px, py = np.asarray(px, float), np.asarray(py, float)
    route_values = np.asarray(route_arcs)
    if px.ndim != 1 or py.ndim != 1 or len(px) != len(py) or len(px) == 0:
        raise ValueError("px and py must be non-empty 1D arrays of equal length.")
    if route_values.ndim != 1 or len(route_values) == 0:
        raise ValueError("route_arcs must be a non-empty 1D sequence.")
    try:
        route_numbers = route_values.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("route_arcs must contain integer arc IDs.") from exc
    if (
        not np.isfinite(route_numbers).all()
        or not np.equal(route_numbers, np.floor(route_numbers)).all()
    ):
        raise ValueError("route_arcs must contain finite integer arc IDs.")
    if not np.isfinite(clip_m) or clip_m <= 0:
        raise ValueError("clip_m must be finite and positive.")

    route = route_numbers.astype(int)
    at = geom.arc_table
    if ((route < 0) | (route >= len(at))).any():
        raise ValueError("route_arcs contains an ID outside the arc table.")
    d = point_segment_geometry(
        px[:, None], py[:, None],
        *[at[c].to_numpy()[route][None] for c in ("x1", "y1", "x2", "y2")])[0]
    cost = np.minimum(d, clip_m)
    n, r = cost.shape
    dp = np.empty((n, r))
    back = np.zeros((n, r), int)
    dp[0] = cost[0]
    for i in range(1, n):
        prev = dp[i - 1]
        run = np.minimum.accumulate(prev)
        new_minimum = np.r_[True, prev[1:] <= run[:-1]]
        back[i] = np.maximum.accumulate(
            np.where(new_minimum, np.arange(r), 0)
        )
        dp[i] = cost[i] + run
    pos = np.zeros(n, int)
    pos[-1] = r - 1 - np.argmin(dp[-1, ::-1])
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos], pos, d[np.arange(n), pos]


def label_candidates(cand, ref_arc):
    out = cand.copy()
    out["label"] = (out["arc_id"].to_numpy() == np.asarray(ref_arc)[out["obs_idx"].to_numpy()]).astype(int)
    out["ref_in_candidates"] = out.groupby("obs_idx")["label"].transform("max")  # 0 = candidate miss
    return out