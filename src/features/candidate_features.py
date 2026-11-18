"""Observation <-> candidate road-segment features.

A road segment (arc) is the straight, DIRECTED line from nodes[from_node] to nodes[to_node].
A candidate set is a DataFrame with columns [obs_idx, arc_id].
"""
import numpy as np
import pandas as pd

from .geo import angular_diff_deg, make_projector, xy_bearing_deg


class Geometry:
    """Projected (metre) geometry for one trip's nodes/arcs/track."""

    def __init__(self, nodes: pd.DataFrame, arcs: pd.DataFrame, ref=None):
        ref = ref or (nodes["longitude"].mean(), nodes["latitude"].mean())
        self.project = make_projector(*ref)
        nx, ny = self.project(nodes["longitude"], nodes["latitude"])
        f, t = arcs["from_node"].to_numpy(int), arcs["to_node"].to_numpy(int)
        self.x1, self.y1, self.x2, self.y2 = nx[f], ny[f], nx[t], ny[t]
        self.arc_table = build_arc_table(arcs, self.x1, self.y1, self.x2, self.y2)

    def project_track(self, track: pd.DataFrame):
        return self.project(track["longitude"], track["latitude"])


def build_arc_table(arcs, x1, y1, x2, y2) -> pd.DataFrame:
    dx, dy = x2 - x1, y2 - y1
    tab = pd.DataFrame({"arc_id": np.arange(len(arcs)), "from_node": arcs["from_node"].to_numpy(int),
                        "to_node": arcs["to_node"].to_numpy(int), "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    tab["seg_length"] = np.hypot(dx, dy)
    tab["seg_bearing"] = np.where(tab["seg_length"] > 0, xy_bearing_deg(dx, dy), np.nan)
    return tab.set_index("arc_id", drop=False)


def point_segment_geometry(px, py, x1, y1, x2, y2):
    """Vectorised projection of points onto segments (all arrays broadcastable).
    Returns perp_dist, frac (0..1, clamped), along (m from start), d_start, d_end, signed_offset."""
    dx, dy = x2 - x1, y2 - y1
    L2 = dx * dx + dy * dy
    L = np.sqrt(L2)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(L2 > 0, ((px - x1) * dx + (py - y1) * dy) / L2, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    cx, cy = x1 + frac * dx, y1 + frac * dy
    perp = np.hypot(px - cx, py - cy)
    cross = dx * (py - y1) - dy * (px - x1)
    with np.errstate(divide="ignore", invalid="ignore"):
        signed = np.where(L > 0, -cross / L, 0.0)  # >0: point is to the right of travel direction
    return perp, frac, frac * L, np.hypot(px - x1, py - y1), np.hypot(px - x2, py - y2), signed


def generate_candidates(px, py, geom: Geometry, k: int = 10, radius_m: float = 100.0, chunk: int = 256) -> pd.DataFrame:
    """k nearest arcs within radius_m of each observation (brute force, chunked)."""
    at = geom.arc_table
    rows = []
    for s in range(0, len(px), chunk):
        e = min(s + chunk, len(px))
        d = point_segment_geometry(px[s:e, None], py[s:e, None], at["x1"].to_numpy()[None], at["y1"].to_numpy()[None],
                                   at["x2"].to_numpy()[None], at["y2"].to_numpy()[None])[0]
        kk = min(k, d.shape[1])
        idx = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        for i in range(e - s):
            for j in idx[i]:
                if d[i, j] <= radius_m:
                    rows.append((s + i, int(j)))
    return pd.DataFrame(rows, columns=["obs_idx", "arc_id"], dtype=int)


def candidate_features(gps: pd.DataFrame, px, py, cands: pd.DataFrame, geom: Geometry) -> pd.DataFrame:
    """One row per (obs_idx, arc_id) with geometric match features.

    gps must contain `heading` (from gps_features). px, py: projected GPS metres.
    """
    c = cands.reset_index(drop=True).copy()
    a = geom.arc_table.loc[c["arc_id"].to_numpy()]
    o = c["obs_idx"].to_numpy()
    perp, frac, along, ds, de, signed = point_segment_geometry(
        px[o], py[o], a["x1"].to_numpy(), a["y1"].to_numpy(), a["x2"].to_numpy(), a["y2"].to_numpy())
    c["perp_dist"], c["proj_frac"], c["dist_along"] = perp, frac, along
    c["dist_to_start"], c["dist_to_end"], c["signed_offset"] = ds, de, signed
    c["seg_length"], c["seg_bearing"] = a["seg_length"].to_numpy(), a["seg_bearing"].to_numpy()
    c["heading"] = gps["heading"].to_numpy()[o]
    c["heading_diff"] = angular_diff_deg(c["heading"], c["seg_bearing"])   # NaN if heading unknown
    c["heading_cos"] = np.cos(np.radians(c["heading_diff"]))
    c["is_opposite_dir"] = (c["heading_diff"] > 90).astype(float).where(c["heading_diff"].notna())
    c["is_endpoint_snap"] = ((c["proj_frac"] <= 0) | (c["proj_frac"] >= 1)).astype(int)
    g = c.groupby("obs_idx")["perp_dist"]
    c["n_candidates"] = g.transform("size")
    c["dist_rank"] = g.rank(method="first")
    c["dist_minus_min"] = c["perp_dist"] - g.transform("min")
    return c