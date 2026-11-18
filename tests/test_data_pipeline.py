import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from src.data.cleaning import clean_trip, clean_trips
from src.data.loaders import load_split
from src.data.validation import validate_trip, validate_trips


def make_trip(track: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    if track is None:
        track = pd.DataFrame(
            {
                "longitude": [-122.1, -122.2],
                "latitude": [37.4, 37.5],
                "time": [1, 2],
            }
        )
    return {
        "track": track,
        "nodes": pd.DataFrame(
            {"longitude": [-122.1, -122.2], "latitude": [37.4, 37.5]}
        ),
        "arcs": pd.DataFrame({"from_node": [0], "to_node": [1]}),
        "route": pd.DataFrame({"arc_id": [0]}),
    }


class DataPipelineTests(unittest.TestCase):
    def test_load_split_reads_trip_files_into_named_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            split_dir = Path(temp_dir) / "train"
            trip_dir = split_dir / "00000001"
            trip_dir.mkdir(parents=True)
            (trip_dir / "00000001.track").write_text(
                "-122.1 37.4 1\n-122.2 37.5 2\n", encoding="utf-8"
            )
            (trip_dir / "00000001.nodes").write_text(
                "-122.1 37.4\n-122.2 37.5\n", encoding="utf-8"
            )
            (trip_dir / "00000001.arcs").write_text("0 1\n", encoding="utf-8")
            (trip_dir / "00000001.route").write_text("0\n", encoding="utf-8")

            trips = load_split(split_dir)

        self.assertEqual(list(trips), ["00000001"])
        self.assertEqual(
            list(trips["00000001"]["track"].columns),
            ["longitude", "latitude", "time"],
        )
        self.assertEqual(trips["00000001"]["track"].shape, (2, 3))

    def test_validate_trip_accepts_consistent_data(self) -> None:
        self.assertEqual(validate_trip(make_trip()), [])
        self.assertEqual(validate_trips({"trip-1": make_trip()}), {})

    def test_validate_trip_reports_invalid_coordinates_order_and_references(self) -> None:
        trip = make_trip(
            pd.DataFrame(
                {
                    "longitude": [-122.1, -122.2],
                    "latitude": [37.4, 95],
                    "time": [2, 1],
                }
            )
        )
        trip["arcs"] = pd.DataFrame({"from_node": [0], "to_node": [2]})
        trip["route"] = pd.DataFrame({"arc_id": [3]})

        issues = validate_trip(trip)

        self.assertTrue(any("latitude" in issue and "outside" in issue for issue in issues))
        self.assertIn("'track.time' must be in non-decreasing order.", issues)
        self.assertTrue(any("'arcs.to_node'" in issue for issue in issues))
        self.assertTrue(any("'route.arc_id'" in issue for issue in issues))

    def test_clean_trip_removes_bad_and_duplicate_points_and_sorts(self) -> None:
        track = pd.DataFrame(
            {
                "longitude": [-122.2, -122.2, 181, -122.1, -122.3],
                "latitude": [37.5, 37.5, 37.6, 37.4, 37.6],
                "time": [2, 2, 3, 1, float("nan")],
            }
        )
        trip = make_trip(track)
        original_nodes = trip["nodes"].copy()
        original_arcs = trip["arcs"].copy()
        original_route = trip["route"].copy()
        raw_issues = validate_trip(trip)
        self.assertIn("'track.time' must be in non-decreasing order.", raw_issues)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = clean_trip(trip)

        self.assertEqual(len(caught), 1)
        self.assertIn("Removed 2 invalid and 1 duplicate", str(caught[0].message))
        print("\nSample cleaning output:")
        print("Raw track:")
        print(trip["track"].to_string(index=False))
        print("Raw validation issues:")
        for issue in raw_issues:
            print(f"- {issue}")
        print(f"Cleaning summary: {caught[0].message}")
        print("Cleaned track:")
        print(cleaned["track"].to_string(index=False))
        print(f"Cleaned validation issues: {validate_trip(cleaned)}")

        self.assertEqual(cleaned["track"]["time"].tolist(), [1, 2])
        pd.testing.assert_frame_equal(cleaned["nodes"], original_nodes)
        pd.testing.assert_frame_equal(cleaned["arcs"], original_arcs)
        pd.testing.assert_frame_equal(cleaned["route"], original_route)
        self.assertEqual(len(trip["track"]), 5)

    def test_clean_trips_preserves_trip_ids_and_rejects_empty_clean_result(self) -> None:
        trips = {"trip-1": make_trip()}
        self.assertEqual(list(clean_trips(trips)), ["trip-1"])

        invalid_trip = make_trip(
            pd.DataFrame(
                {"longitude": [181], "latitude": [37.4], "time": [1]}
            )
        )
        with self.assertRaisesRegex(ValueError, "No valid GPS observations"):
            clean_trip(invalid_trip)


if __name__ == "__main__":
    unittest.main()
