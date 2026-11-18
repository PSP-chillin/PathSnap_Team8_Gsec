"""Training-only: per-observation reference arc by ordered alignment of the track to `.route`."""
import numpy as np

from .candidate_features import point_segment_geometry


def align_track_to_route(px, py, route_arcs, geom, clip_m=100.0):
    """Return (ref_arc_id, route_pos, dist) with route_pos non-decreasing."""
    route = np.asarray(route_arcs, int)
    at = geom.arc_table
    d = point_segment_geometry(
        np.asarray(px)[:, None], np.asarray(py)[:, None],
        *[at[c].to_numpy()[route][None] for c in ("x1", "y1", "x2", "y2")])[0]
    cost = np.minimum(d, clip_m)
    n, r = cost.shape
    dp = np.empty((n, r)); back = np.zeros((n, r), int); dp[0] = cost[0]
    for i in range(1, n):
        prev = dp[i - 1]; run = np.minimum.accumulate(prev)
        back[i] = [np.flatnonzero(prev[:j + 1] == run[j])[0] for j in range(r)]
        dp[i] = cost[i] + run
    pos = np.zeros(n, int); pos[-1] = dp[-1].argmin()
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return route[pos], pos, d[np.arange(n), pos]


def label_candidates(cand, ref_arc):
    out = cand.copy()
    out["label"] = (out["arc_id"].to_numpy() == np.asarray(ref_arc)[out["obs_idx"].to_numpy()]).astype(int)
    out["ref_in_candidates"] = out.groupby("obs_idx")["label"].transform("max")  # 0 = candidate miss
    return out