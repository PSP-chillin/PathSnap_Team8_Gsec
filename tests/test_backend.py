import numpy as np

from src.hmm.viterbi import HMMConfig, Observation, RoadGraph, collapse, continuity, fill_route, viterbi_decode

# Straight road 0 -> 1 -> 2 -> 3 along the x axis (100 m arcs 0, 1, 2),
# plus a side road 1 -> 4 (arc 3) that is NOT connected to arc 2.
GRAPH_ARGS = ([0, 1, 2, 1], [1, 2, 3, 4], [100.0, 100.0, 100.0, 100.0])


def obs(arcs, along, probs):
    return Observation(np.array(arcs), np.array(along, dtype=float), np.log(np.array(probs, dtype=float)))


def test_viterbi_prefers_connected_route_over_noisy_emission():
    graph = RoadGraph(*GRAPH_ARGS)
    px = np.array([150.0, 160.0, 170.0])
    py = np.zeros(3)
    times = np.array([0.0, 1.0, 2.0])
    observations = [
        obs([1, 3], [50.0, 50.0], [0.9, 0.1]),
        obs([1, 3], [60.0, 60.0], [0.45, 0.55]),   # classifier slightly prefers the side road
        obs([1, 3], [70.0, 70.0], [0.9, 0.1]),
    ]
    best, chains = viterbi_decode(observations, px, py, times, graph)
    assert best.tolist() == [1, 1, 1]
    assert chains == 1


def test_gap_breaks_chain_and_missing_candidates_are_skipped():
    graph = RoadGraph(*GRAPH_ARGS)
    px = np.array([10.0, 20.0, 30.0, 40.0])
    py = np.zeros(4)
    times = np.array([0.0, 1.0, 500.0, 501.0])  # 499 s gap
    observations = [obs([0], [10.0], [0.9]), None, obs([0], [30.0], [0.9]), obs([0], [40.0], [0.9])]
    best, chains = viterbi_decode(observations, px, py, times, graph, HMMConfig(max_gap_s=60.0))
    assert best.tolist() == [0, -1, 0, 0]
    assert chains == 2


def test_fill_route_joins_gaps_and_continuity_counts_links():
    graph = RoadGraph(*GRAPH_ARGS)
    sequence = collapse([0, 0, -1, 2, 2])
    assert sequence == [0, 2]
    assert continuity(sequence, graph) == 0.0
    assert fill_route(sequence, graph) == [0, 1, 2]
    assert continuity([0, 1, 2], graph) == 1.0
