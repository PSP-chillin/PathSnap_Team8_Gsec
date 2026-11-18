"""Road-network features: adjacency, connectivity, shortest-path distance, transition feasibility."""
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MAX_SPEED_MPS = 40.0


class RoadNetwork:
    def __init__(self, arc_table: pd.DataFrame, n_nodes: int):
        self.at = arc_table
        self.n_nodes = n_nodes
        self.f = arc_table["from_node"].to_numpy(int)
        self.t = arc_table["to_node"].to_numpy(int)
        self.len = arc_table["seg_length"].to_numpy(float)
        self.out_adj = defaultdict(list)           # node -> [(to_node, length)]
        for fn, tn, ln in zip(self.f, self.t, self.len):
            self.out_adj[int(fn)].append((int(tn), float(ln)))
        self._cache = {}
        self.arc_pairs = set(zip(self.f.tolist(), self.t.tolist()))

    def node_distances(self, source: int, cutoff: float) -> dict:
        """Dijkstra from `source` over directed arcs; nodes farther than cutoff are omitted."""
        key = (source, cutoff)
        if key in self._cache:
            return self._cache[key]
        dist, pq = {source: 0.0}, [(0.0, source)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist.get(u, np.inf):
                continue
            for v, w in self.out_adj.get(u, ()):
                nd = d + w
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        self._cache[key] = dist
        return dist

    def route_distance(self, arc_a, along_a, arc_b, along_b, cutoff=5000.0) -> float:
        """Directed driving distance from a point on arc_a to a point on arc_b (inf if unreachable/too far)."""
        la, lb = self.len[arc_a], self.len[arc_b]
        if arc_a == arc_b and along_b >= along_a:
            distance = float(along_b - along_a)
            return distance if distance <= cutoff else np.inf
        rest_a = la - along_a
        if rest_a + along_b > cutoff:
            return np.inf
        d = self.node_distances(int(self.t[arc_a]), cutoff - rest_a - along_b).get(int(self.f[arc_b]))
        return np.inf if d is None else float(rest_a + d + along_b)


def arc_relation(net: RoadNetwork, a: int, b: int) -> dict:
    fa, ta, fb, tb = net.f[a], net.t[a], net.f[b], net.t[b]
    return {
        "same_arc": int(a == b),
        "is_successor": int(ta == fb and a != b),          # b continues directly after a
        "is_predecessor": int(tb == fa and a != b),
        "is_reverse_pair": int(ta == fb and fa == tb),     # opposite direction of same road
        "shares_node": int(len({fa, ta} & {fb, tb}) > 0),
    }


def local_arc_properties(arc_table: pd.DataFrame, n_nodes: int) -> pd.DataFrame:
    """Static per-arc network descriptors (indexed by arc_id)."""
    f, t = arc_table["from_node"].to_numpy(int), arc_table["to_node"].to_numpy(int)
    outdeg, indeg = np.bincount(f, minlength=n_nodes), np.bincount(t, minlength=n_nodes)
    pairs = set(zip(f.tolist(), t.tolist()))
    _, comp = connected_components(coo_matrix((np.ones(len(f)), (f, t)), shape=(n_nodes, n_nodes)), directed=False)
    comp_size = np.bincount(comp)
    p = pd.DataFrame({
        "from_in_degree": indeg[f], "from_out_degree": outdeg[f],
        "to_in_degree": indeg[t], "to_out_degree": outdeg[t],
        "has_reverse_arc": [int((b, a) in pairs) for a, b in zip(f, t)],
        "component_size": comp_size[comp[f]],
    }, index=arc_table.index)
    p["is_dead_end"] = (p["to_out_degree"] == 0).astype(int)
    p["to_is_intersection"] = ((p["to_out_degree"] + p["to_in_degree"]) > 2).astype(int)
    p["n_next_arcs"] = p["to_out_degree"]
    return p


def candidate_local_features(cf: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    """Attach static local road-network properties to a candidate feature table."""
    return cf.join(props, on="arc_id")


def transition_features(cf: pd.DataFrame, gps: pd.DataFrame, px, py, net: RoadNetwork,
                        max_speed=MAX_SPEED_MPS, min_cutoff=200.0, max_cutoff=5000.0) -> pd.DataFrame:
    """Features for every candidate pair (arc at obs i, arc at obs i+1).

    Columns: obs_idx, arc_from, arc_to, adjacency flags, route_dist (m, inf if infeasible),
    gc_dist (straight-line GPS step), route_minus_gc, route_gc_ratio, implied_speed, is_feasible.
    """
    by_obs = {k: g for k, g in cf.groupby("obs_idx")}
    dt = gps["time"].diff().shift(-1).to_numpy()
    rows = []
    for i in sorted(by_obs):
        if i + 1 not in by_obs:
            continue
        gc = float(np.hypot(px[i + 1] - px[i], py[i + 1] - py[i]))
        cutoff = float(np.clip(max_speed * (dt[i] if dt[i] > 0 else 1.0), min_cutoff, max_cutoff))
        for a in by_obs[i].itertuples():
            for b in by_obs[i + 1].itertuples():
                rd = net.route_distance(a.arc_id, a.dist_along, b.arc_id, b.dist_along, cutoff)
                rel = arc_relation(net, a.arc_id, b.arc_id)
                rows.append({"obs_idx": i, "arc_from": a.arc_id, "arc_to": b.arc_id, **rel,
                             "route_dist": rd, "gc_dist": gc, "dt": dt[i]})
    tr = pd.DataFrame(rows)
    if tr.empty:
        return tr
    fin = np.isfinite(tr["route_dist"])
    tr["is_reachable"] = fin.astype(int)
    tr["route_minus_gc"] = (tr["route_dist"] - tr["gc_dist"]).abs()
    tr["route_gc_ratio"] = tr["route_dist"] / tr["gc_dist"].where(tr["gc_dist"] > 1.0)
    tr["implied_speed"] = tr["route_dist"] / tr["dt"].where(tr["dt"] > 0)
    tr["is_feasible"] = (fin & (tr["implied_speed"].fillna(0) <= max_speed)).astype(int)
    return tr