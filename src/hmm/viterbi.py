"""HMM map matching with Viterbi decoding.

Hidden states: for each GPS observation, its candidate road segments (arcs).
Emission:      log P(candidate is the true segment) from the trained classifier.
Transition:    how well the network route distance between two consecutive
               candidates matches the straight-line GPS distance
               (Newson & Krumm style): log p = -|d_route - d_gps| / beta.
               If no route exists within the search limit, a fixed penalty is
               used instead of -inf, so decoding can recover from a bad step.
Gaps:          a time gap or GPS jump above the limits starts a new chain;
               observations with no candidates are skipped.
"""

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HMMConfig:
    beta_m: float = 10.0               # scale of the route-vs-GPS distance mismatch
    max_gap_s: float = 60.0            # a longer time gap breaks the chain
    max_step_m: float = 1000.0         # a longer GPS jump breaks the chain
    route_slack_m: float = 150.0       # network search radius beyond the GPS step
    disconnected_log_penalty: float = 50.0
    backtrack_tolerance_m: float = 15.0
    min_prob: float = 1e-6
    fill_search_m: float = 2000.0      # max network distance when joining route gaps


class RoadGraph:
    """Directed road network with cached, distance-bounded Dijkstra searches."""

    def __init__(self, from_nodes, to_nodes, lengths):
        self.from_nodes = np.asarray(from_nodes, dtype=int)
        self.to_nodes = np.asarray(to_nodes, dtype=int)
        self.lengths = np.asarray(lengths, dtype=float)
        self.out_arcs: dict[int, list[int]] = defaultdict(list)
        for arc, node in enumerate(self.from_nodes):
            self.out_arcs[int(node)].append(arc)
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def shortest_from(self, source: int, cutoff: float):
        """Distances (and last arc used) to every node within `cutoff` metres of `source`."""
        cached = self._cache.get(source)
        if cached is not None and cached[0] >= cutoff:
            return cached[1], cached[2]
        dist = {source: 0.0}
        via_arc: dict[int, int] = {}
        heap = [(0.0, source)]
        while heap:
            d, node = heapq.heappop(heap)
            if d > dist.get(node, math.inf):
                continue
            for arc in self.out_arcs.get(node, ()):
                nd = d + self.lengths[arc]
                target = int(self.to_nodes[arc])
                if nd <= cutoff and nd < dist.get(target, math.inf):
                    dist[target] = nd
                    via_arc[target] = arc
                    heapq.heappush(heap, (nd, target))
        self._cache[source] = (cutoff, dist, via_arc)
        return dist, via_arc

    def path_arcs(self, source: int, target: int, cutoff: float) -> list[int] | None:
        """Arcs of the shortest path from node `source` to node `target`, or None."""
        dist, via_arc = self.shortest_from(source, cutoff)
        if target not in dist:
            return None
        arcs = []
        node = target
        while node != source:
            arc = via_arc[node]
            arcs.append(arc)
            node = int(self.from_nodes[arc])
        arcs.reverse()
        return arcs

    def route_distance(self, arc_a: int, along_a: float, arc_b: int, along_b: float,
                       cutoff: float, tolerance: float) -> float | None:
        """Network distance from a point on arc_a to a point on arc_b, or None if unreachable."""
        if arc_a == arc_b:
            forward = along_b - along_a
            if forward >= -tolerance:
                return max(forward, 0.0)
        remaining = self.lengths[arc_a] - along_a
        dist, _ = self.shortest_from(int(self.to_nodes[arc_a]), cutoff)
        between = dist.get(int(self.from_nodes[arc_b]))
        if between is None:
            return None
        return remaining + between + along_b

    def connected(self, arc_a: int, arc_b: int) -> bool:
        return arc_a == arc_b or self.to_nodes[arc_a] == self.from_nodes[arc_b]


@dataclass
class Observation:
    arcs: np.ndarray       # candidate arc ids
    along: np.ndarray      # distance along each candidate arc (m)
    log_emit: np.ndarray   # log emission probability per candidate


def viterbi_decode(observations: list[Observation | None], px: np.ndarray, py: np.ndarray,
                   times: np.ndarray, graph: RoadGraph, config: HMMConfig = HMMConfig()):
    """Most likely candidate arc per observation (-1 where there were no candidates).

    Returns (best_arcs, chain_count).
    """
    n = len(observations)
    best = np.full(n, -1, dtype=int)
    chain: list[int] = []
    back: dict[int, np.ndarray | None] = {}
    scores = None
    chains = 0

    def flush():
        nonlocal chains
        if not chain:
            return
        chains += 1
        state = int(np.argmax(scores))
        for position in range(len(chain) - 1, -1, -1):
            t = chain[position]
            best[t] = int(observations[t].arcs[state])
            pointer = back[t]
            if pointer is not None:
                state = int(pointer[state])
        chain.clear()

    for t, obs in enumerate(observations):
        if obs is None or len(obs.arcs) == 0:
            continue  # no candidates: skip this point, keep the chain going
        prev = chain[-1] if chain else None
        if prev is not None:
            d_gps = float(math.hypot(px[t] - px[prev], py[t] - py[prev]))
            dt = float(times[t] - times[prev])
            if dt > config.max_gap_s or d_gps > config.max_step_m:
                flush()
                prev = None
        if prev is None:
            scores = obs.log_emit.astype(float).copy()
            back[t] = None
            chain.append(t)
            continue

        previous = observations[prev]
        cutoff = max(3.0 * d_gps, d_gps + config.route_slack_m)
        transition = np.empty((len(previous.arcs), len(obs.arcs)))
        for i, (arc_a, along_a) in enumerate(zip(previous.arcs, previous.along)):
            for j, (arc_b, along_b) in enumerate(zip(obs.arcs, obs.along)):
                d_route = graph.route_distance(int(arc_a), float(along_a), int(arc_b), float(along_b),
                                               cutoff, config.backtrack_tolerance_m)
                if d_route is None:
                    transition[i, j] = -config.disconnected_log_penalty
                else:
                    transition[i, j] = -abs(d_route - d_gps) / config.beta_m
        total = scores[:, None] + transition
        pointer = np.argmax(total, axis=0)
        scores = total[pointer, np.arange(len(obs.arcs))] + obs.log_emit
        back[t] = pointer
        chain.append(t)

    flush()
    return best, chains


def collapse(arcs) -> list[int]:
    """Arc sequence with consecutive repeats removed (ignores -1)."""
    sequence: list[int] = []
    for arc in arcs:
        if arc < 0:
            continue
        if not sequence or sequence[-1] != arc:
            sequence.append(int(arc))
    return sequence


def continuity(sequence: list[int], graph: RoadGraph) -> float:
    """Share of consecutive segment changes that are directly connected in the network."""
    if len(sequence) < 2:
        return 1.0
    links = [graph.connected(a, b) for a, b in zip(sequence[:-1], sequence[1:])]
    return float(np.mean(links))


def fill_route(sequence: list[int], graph: RoadGraph, config: HMMConfig = HMMConfig()) -> list[int]:
    """Join consecutive segments that don't touch with the shortest network path between them."""
    if not sequence:
        return []
    route = [sequence[0]]
    for arc in sequence[1:]:
        last = route[-1]
        if not graph.connected(last, arc):
            path = graph.path_arcs(int(graph.to_nodes[last]), int(graph.from_nodes[arc]), config.fill_search_m)
            if path:
                route.extend(a for a in path if a != route[-1])
        if route[-1] != arc:
            route.append(arc)
    return route
