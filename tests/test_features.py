import numpy as np
import pandas as pd
import pytest

from src.features.candidate_features import Geometry, candidate_features, point_segment_geometry
from src.features.geo import angular_diff_deg, haversine_m, initial_bearing_deg
from src.features.gps_features import add_gps_features
from src.features.network_features import RoadNetwork, arc_relation, local_arc_properties, transition_features
from src.features.pipeline import build_trip_features
from src.features.temporal_features import add_temporal_features

M_PER_DEG = 111_194.9  # metres per degree latitude


def test_haversine_one_degree_lat():
    assert haversine_m(0, 0, 0, 1) == pytest.approx(M_PER_DEG, rel=1e-3)


def test_bearing_cardinals():
    assert initial_bearing_deg(0, 0, 0, 1) == pytest.approx(0)
    assert initial_bearing_deg(0, 0, 1, 0) == pytest.approx(90, abs=0.1)
    assert initial_bearing_deg(0, 0, 0, -1) == pytest.approx(180)
    assert initial_bearing_deg(0, 0, -1, 0) == pytest.approx(270, abs=0.1)


def test_angular_diff_wraps():
    assert angular_diff_deg(350, 10) == pytest.approx(20)
    assert angular_diff_deg(0, 180) == pytest.approx(180)
    assert np.isnan(angular_diff_deg(np.nan, 10))


def test_point_segment_perp_and_endpoints():
    perp, frac, along, ds, de, signed = point_segment_geometry(5.0, 3.0, 0.0, 0.0, 10.0, 0.0)
    assert (perp, frac, along, ds) == (3.0, 0.5, 5.0, pytest.approx(np.hypot(5, 3)))
    assert de == pytest.approx(np.hypot(5, 3)) and signed == pytest.approx(-3.0)  # left of eastbound
    perp, frac, *_ = point_segment_geometry(-4.0, 3.0, 0.0, 0.0, 10.0, 0.0)  # beyond start: clamped
    assert frac == 0 and perp == pytest.approx(5.0)


def _grid():
    # nodes 0-1-2 along east (0.001 deg lon apart ~ 110 m at equator), 3 north of 1
    nodes = pd.DataFrame({"longitude": [0, 0.001, 0.002, 0.001], "latitude": [0, 0, 0, 0.001]})
    arcs = pd.DataFrame({"from_node": [0, 1, 2, 1], "to_node": [1, 2, 1, 3]})
    return nodes, arcs


def test_arc_length_bearing_and_candidate_features():
    nodes, arcs = _grid()
    g = Geometry(nodes, arcs, ref=(0, 0))
    assert g.arc_table.loc[0, "seg_length"] == pytest.approx(111.19, rel=1e-2)
    assert g.arc_table.loc[0, "seg_bearing"] == pytest.approx(90, abs=0.1)
    assert g.arc_table.loc[3, "seg_bearing"] == pytest.approx(0, abs=0.1)
    gps = pd.DataFrame({"longitude": [0.0005], "latitude": [0.00002], "time": [0.0], "heading": [90.0]})
    px, py = g.project_track(gps)
    cf = candidate_features(gps, px, py, pd.DataFrame({"obs_idx": [0, 0], "arc_id": [0, 2]}), g)
    assert cf.loc[0, "perp_dist"] == pytest.approx(2.2, abs=0.1)
    assert cf.loc[0, "heading_diff"] == pytest.approx(0, abs=0.1)
    assert cf.loc[1, "heading_diff"] == pytest.approx(180, abs=0.1) and cf.loc[1, "is_opposite_dir"] == 1


def test_connectivity_and_route_distance():
    nodes, arcs = _grid()
    g = Geometry(nodes, arcs, ref=(0, 0))
    net = RoadNetwork(g.arc_table, 4)
    assert arc_relation(net, 0, 1)["is_successor"] == 1
    assert arc_relation(net, 1, 2)["is_reverse_pair"] == 1
    assert arc_relation(net, 0, 2)["is_successor"] == 0
    L = net.len
    assert net.route_distance(0, 20.0, 0, 50.0) == pytest.approx(30.0)
    assert net.route_distance(0, 20.0, 1, 10.0) == pytest.approx(L[0] - 20 + 10)
    assert net.route_distance(0, 20.0, 3, 10.0) == pytest.approx(L[0] - 20 + 10)   # via node 1
    assert net.route_distance(3, 0.0, 0, 0.0) == np.inf                           # dead end at node 3
    p = local_arc_properties(g.arc_table, 4)
    assert p.loc[3, "is_dead_end"] == 1 and p.loc[0, "to_is_intersection"] == 1 and p.loc[0, "has_reverse_arc"] == 0


def test_transition_feasibility():
    nodes, arcs = _grid()
    g = Geometry(nodes, arcs, ref=(0, 0))
    net = RoadNetwork(g.arc_table, 4)
    gps = add_gps_features(pd.DataFrame({"longitude": [0.0003, 0.0007], "latitude": [0, 0], "time": [0.0, 2.0]}))
    px, py = g.project_track(gps)
    cf = candidate_features(gps, px, py, pd.DataFrame({"obs_idx": [0, 1, 1], "arc_id": [0, 0, 3]}), g)
    tr = transition_features(cf, gps, px, py, net)
    same = tr[tr.arc_to == 0].iloc[0]
    assert same.route_dist == pytest.approx(44.5, rel=2e-2) and same.is_feasible == 1
    far = tr[tr.arc_to == 3].iloc[0]
    assert far.is_successor == 1 and far.route_dist > same.route_dist


def _track(times, lons, lats):
    return pd.DataFrame({"longitude": lons, "latitude": lats, "time": times})


def test_speed_heading_time():
    # 0.0001 deg lat ~ 11.12 m per second heading north
    trk = _track([0, 1, 2, 3], [0, 0, 0, 0], [0, 1e-4, 2e-4, 3e-4])
    f = add_temporal_features(add_gps_features(trk))
    assert f["speed"].iloc[1] == pytest.approx(11.12, rel=1e-2)
    assert f["dist_next"].iloc[0] == pytest.approx(f["dist_prev"].iloc[1])
    assert f["heading"].iloc[0] == pytest.approx(0) and np.isnan(f["dt_prev"].iloc[0])
    assert f["acceleration"].iloc[1] == pytest.approx(0, abs=1e-6)
    assert (f["dt_next"].iloc[:-1] == 1).all() and f["is_continuous"].iloc[1] == 1


def test_gap_and_turn_and_stationary():
    trk = _track([0, 1, 2, 10, 11], [0, 0, 1e-4, 1e-4, 1e-4], [0, 1e-4, 1e-4, 1e-4, 1e-4])
    f = add_temporal_features(add_gps_features(trk))
    assert f["heading_change"].iloc[1] == pytest.approx(90, abs=0.5)   # north then east
    assert f["is_gap_prev"].tolist() == [0, 0, 0, 1, 0] and f["segment_id"].tolist() == [0, 0, 0, 1, 1]
    assert f["is_continuous"].iloc[2] == 0
    s = add_gps_features(_track([0, 1, 2], [0, 0, 0], [0, 0, 0]))
    assert s["heading"].isna().all() and (s["is_stationary"].iloc[1:] == 1).all()


def test_duplicate_time_gives_nan_speed():
    f = add_gps_features(_track([0, 0, 1], [0, 1e-4, 2e-4], [0, 0, 0]))
    assert np.isnan(f["speed"].iloc[1])


def test_pipeline_end_to_end():
    nodes, arcs = _grid()
    trk = _track(range(5), [0.0002 + 0.0002 * i for i in range(5)], [1e-5] * 5)
    gps, cf, tr, *_ = build_trip_features({"track": trk, "nodes": nodes, "arcs": arcs})
    assert len(gps) == 5 and {"perp_dist", "heading_diff", "is_dead_end"} <= set(cf.columns)
    assert {"route_dist", "is_feasible"} <= set(tr.columns)