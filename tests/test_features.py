import unittest

import numpy as np
import pandas as pd

from src.features.candidate_features import (
    Geometry,
    candidate_features,
    generate_candidates,
    point_segment_geometry,
)
from src.features.geo import angular_diff_deg, haversine_m, initial_bearing_deg
from src.features.gps_features import add_gps_features
from src.features.labels import align_track_to_route, label_candidates
from src.features.network_features import (
    RoadNetwork,
    arc_relation,
    local_arc_properties,
    transition_features,
)
from src.features.pipeline import build_trip_features
from src.features.temporal_features import add_temporal_features


M_PER_DEG = 111_194.9


def _grid():
    nodes = pd.DataFrame(
        {
            "longitude": [0, 0.001, 0.002, 0.001],
            "latitude": [0, 0, 0, 0.001],
        }
    )
    arcs = pd.DataFrame(
        {"from_node": [0, 1, 2, 1], "to_node": [1, 2, 1, 3]}
    )
    return nodes, arcs


def _track(times, lons, lats):
    return pd.DataFrame({"longitude": lons, "latitude": lats, "time": times})


class FeatureTests(unittest.TestCase):
    def test_haversine_one_degree_latitude(self):
        distance = haversine_m(0, 0, 0, 1)
        self.assertAlmostEqual(distance, M_PER_DEG, delta=M_PER_DEG * 1e-3)

    def test_bearing_cardinals(self):
        self.assertAlmostEqual(initial_bearing_deg(0, 0, 0, 1), 0)
        self.assertAlmostEqual(initial_bearing_deg(0, 0, 1, 0), 90, delta=0.1)
        self.assertAlmostEqual(initial_bearing_deg(0, 0, 0, -1), 180)
        self.assertAlmostEqual(initial_bearing_deg(0, 0, -1, 0), 270, delta=0.1)

    def test_angular_difference_wraps(self):
        self.assertAlmostEqual(angular_diff_deg(350, 10), 20)
        self.assertAlmostEqual(angular_diff_deg(0, 180), 180)
        self.assertTrue(np.isnan(angular_diff_deg(np.nan, 10)))

    def test_point_segment_perpendicular_distance_and_endpoints(self):
        perp, frac, along, d_start, d_end, signed = point_segment_geometry(
            5.0, 3.0, 0.0, 0.0, 10.0, 0.0
        )
        np.testing.assert_allclose(
            [perp, frac, along, d_start],
            [3.0, 0.5, 5.0, np.hypot(5, 3)],
        )
        self.assertAlmostEqual(d_end, np.hypot(5, 3))
        self.assertAlmostEqual(signed, -3.0)

        perp, frac, *_ = point_segment_geometry(
            -4.0, 3.0, 0.0, 0.0, 10.0, 0.0
        )
        self.assertEqual(frac, 0)
        self.assertAlmostEqual(perp, 5.0)

    def test_arc_geometry_and_candidate_features(self):
        nodes, arcs = _grid()
        geometry = Geometry(nodes, arcs, ref=(0, 0))
        self.assertAlmostEqual(
            geometry.arc_table.loc[0, "seg_length"], 111.19, delta=1.1119
        )
        self.assertAlmostEqual(geometry.arc_table.loc[0, "seg_bearing"], 90, delta=0.1)
        self.assertAlmostEqual(geometry.arc_table.loc[3, "seg_bearing"], 0, delta=0.1)

        gps = pd.DataFrame(
            {
                "longitude": [0.0005],
                "latitude": [0.00002],
                "time": [0.0],
                "heading": [90.0],
            }
        )
        px, py = geometry.project_track(gps)
        candidates = pd.DataFrame({"obs_idx": [0, 0], "arc_id": [0, 2]})
        features = candidate_features(gps, px, py, candidates, geometry)
        self.assertAlmostEqual(features.loc[0, "perp_dist"], 2.2, delta=0.1)
        self.assertAlmostEqual(features.loc[0, "heading_diff"], 0, delta=0.1)
        self.assertAlmostEqual(features.loc[1, "heading_diff"], 180, delta=0.1)
        self.assertEqual(features.loc[1, "is_opposite_dir"], 1)

    def test_candidate_distance_ranks_ignore_input_order(self):
        nodes, arcs = _grid()
        geometry = Geometry(nodes, arcs, ref=(0, 0))
        gps = pd.DataFrame(
            {
                "longitude": [0.0005],
                "latitude": [0.00002],
                "time": [0.0],
                "heading": [90.0],
            }
        )
        px, py = geometry.project_track(gps)
        candidates = pd.DataFrame({"obs_idx": [0, 0], "arc_id": [3, 0]})

        features = candidate_features(gps, px, py, candidates, geometry)

        ranks = features.set_index("arc_id")["dist_rank"]
        self.assertEqual(ranks[0], 1)
        self.assertEqual(ranks[3], 2)

    def test_candidate_search_returns_nearest_arcs_within_radius(self):
        nodes, arcs = _grid()
        geometry = Geometry(nodes, arcs, ref=(0, 0))
        px, py = geometry.project_track(
            pd.DataFrame({"longitude": [0.0005], "latitude": [0.00002]})
        )

        candidates = generate_candidates(px, py, geometry, k=2, radius_m=120)

        distances = point_segment_geometry(
            px[0],
            py[0],
            geometry.arc_table["x1"].to_numpy(),
            geometry.arc_table["y1"].to_numpy(),
            geometry.arc_table["x2"].to_numpy(),
            geometry.arc_table["y2"].to_numpy(),
        )[0]
        expected = np.argsort(distances, kind="stable")[:2].tolist()
        self.assertEqual(candidates["arc_id"].tolist(), expected)

    def test_connectivity_and_route_distance(self):
        nodes, arcs = _grid()
        geometry = Geometry(nodes, arcs, ref=(0, 0))
        network = RoadNetwork(geometry.arc_table, 4)
        self.assertEqual(arc_relation(network, 0, 1)["is_successor"], 1)
        self.assertEqual(arc_relation(network, 1, 2)["is_reverse_pair"], 1)
        self.assertEqual(arc_relation(network, 0, 2)["is_successor"], 0)

        lengths = network.len
        self.assertAlmostEqual(network.route_distance(0, 20.0, 0, 50.0), 30.0)
        self.assertAlmostEqual(
            network.route_distance(0, 20.0, 1, 10.0), lengths[0] - 20 + 10
        )
        self.assertAlmostEqual(
            network.route_distance(0, 20.0, 3, 10.0), lengths[0] - 20 + 10
        )
        self.assertTrue(np.isinf(network.route_distance(3, 0.0, 0, 0.0)))
        self.assertTrue(
            np.isinf(network.route_distance(0, 0.0, 0, 100.0, cutoff=50.0))
        )

        properties = local_arc_properties(geometry.arc_table, 4)
        self.assertEqual(properties.loc[3, "is_dead_end"], 1)
        self.assertEqual(properties.loc[0, "to_is_intersection"], 1)
        self.assertEqual(properties.loc[0, "has_reverse_arc"], 0)

    def test_route_alignment_and_candidate_labels(self):
        nodes, arcs = _grid()
        geometry = Geometry(nodes, arcs, ref=(0, 0))
        track = pd.DataFrame(
            {
                "longitude": [0.0005, 0.0015],
                "latitude": [0, 0],
                "time": [0, 1],
            }
        )
        px, py = geometry.project_track(track)

        ref_arc, route_pos, distance = align_track_to_route(px, py, [0, 1], geometry)
        labeled = label_candidates(
            pd.DataFrame(
                {"obs_idx": [0, 0, 1, 1], "arc_id": [0, 3, 0, 1]}
            ),
            ref_arc,
        )

        self.assertEqual(ref_arc.tolist(), [0, 1])
        self.assertEqual(route_pos.tolist(), [0, 1])
        self.assertTrue((distance < 1).all())
        self.assertEqual(labeled["label"].tolist(), [1, 0, 0, 1])
        self.assertEqual(labeled["ref_in_candidates"].tolist(), [1, 1, 1, 1])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            align_track_to_route(px, py, [], geometry)

    def test_route_alignment_advances_across_clipped_cost_ties(self):
        nodes, arcs = _grid()
        geometry = Geometry(nodes, arcs, ref=(0, 0))
        track = pd.DataFrame(
            {
                "longitude": [0.0002, 0.0006, 0.0011, 0.0015, 0.0018],
                "latitude": [0, 0, 0, 0, 0],
            }
        )
        px, py = geometry.project_track(track)

        _, route_pos, distance = align_track_to_route(px, py, [0, 1], geometry)

        self.assertTrue((np.diff(route_pos) >= 0).all())
        self.assertEqual(route_pos[-1], 1)
        self.assertTrue((distance < 1).all())

    def test_transition_feasibility(self):
        nodes, arcs = _grid()
        geometry = Geometry(nodes, arcs, ref=(0, 0))
        network = RoadNetwork(geometry.arc_table, 4)
        gps = add_gps_features(
            pd.DataFrame(
                {
                    "longitude": [0.0003, 0.0007],
                    "latitude": [0, 0],
                    "time": [0.0, 2.0],
                }
            )
        )
        px, py = geometry.project_track(gps)
        candidates = pd.DataFrame(
            {"obs_idx": [0, 1, 1], "arc_id": [0, 0, 3]}
        )
        candidate_table = candidate_features(gps, px, py, candidates, geometry)
        transitions = transition_features(candidate_table, gps, px, py, network)

        same_arc = transitions[transitions.arc_to == 0].iloc[0]
        self.assertAlmostEqual(same_arc.route_dist, 44.5, delta=44.5 * 0.02)
        self.assertEqual(same_arc.is_feasible, 1)
        farther = transitions[transitions.arc_to == 3].iloc[0]
        self.assertEqual(farther.is_successor, 1)
        self.assertGreater(farther.route_dist, same_arc.route_dist)

    def test_speed_heading_and_time_features(self):
        track = _track(
            [0, 1, 2, 3],
            [0, 0, 0, 0],
            [0, 1e-4, 2e-4, 3e-4],
        )
        features = add_temporal_features(add_gps_features(track))

        self.assertAlmostEqual(features["speed"].iloc[1], 11.12, delta=0.1112)
        self.assertAlmostEqual(
            features["dist_next"].iloc[0], features["dist_prev"].iloc[1]
        )
        self.assertAlmostEqual(features["heading"].iloc[0], 0)
        self.assertTrue(np.isnan(features["dt_prev"].iloc[0]))
        self.assertAlmostEqual(features["acceleration"].iloc[1], 0, delta=1e-6)
        self.assertTrue((features["dt_next"].iloc[:-1] == 1).all())
        self.assertEqual(features["is_continuous"].iloc[1], 1)

    def test_gap_turn_and_stationary_features(self):
        track = _track(
            [0, 1, 2, 10, 11],
            [0, 0, 1e-4, 1e-4, 1e-4],
            [0, 1e-4, 1e-4, 1e-4, 1e-4],
        )
        features = add_temporal_features(add_gps_features(track))
        self.assertAlmostEqual(
            features["heading_change"].iloc[1], 90, delta=0.5
        )
        self.assertEqual(features["is_gap_prev"].tolist(), [0, 0, 0, 1, 0])
        self.assertEqual(features["segment_id"].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(features["is_continuous"].iloc[2], 0)

        stationary = add_gps_features(_track([0, 1, 2], [0, 0, 0], [0, 0, 0]))
        self.assertTrue(stationary["heading"].isna().all())
        self.assertTrue((stationary["is_stationary"].iloc[1:] == 1).all())

    def test_duplicate_time_gives_nan_speed(self):
        features = add_gps_features(
            _track([0, 0, 1], [0, 1e-4, 2e-4], [0, 0, 0])
        )
        self.assertTrue(np.isnan(features["speed"].iloc[1]))

    def test_feature_pipeline_end_to_end(self):
        nodes, arcs = _grid()
        track = _track(
            range(5),
            [0.0002 + 0.0002 * i for i in range(5)],
            [1e-5] * 5,
        )

        gps, candidates, transitions, *_ = build_trip_features(
            {"track": track, "nodes": nodes, "arcs": arcs}
        )

        self.assertEqual(len(gps), 5)
        self.assertTrue(
            {"perp_dist", "heading_diff", "is_dead_end"} <= set(candidates.columns)
        )
        self.assertTrue({"route_dist", "is_feasible"} <= set(transitions.columns))


if __name__ == "__main__":
    unittest.main()
