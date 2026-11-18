import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from src.data.cleaning import clean_trip, clean_trip_with_stats, clean_trips
from src.data.loaders import load_split, load_trip
from src.data.report import (
    PartitionDataSummary,
    TripDataSummary,
    write_data_quality_report,
)
from src.data.validation import validate_trip, validate_trips


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = REPOSITORY_ROOT / "data" / "raw"
TRAIN_DIR = RAW_DATA_DIR / "train"
STUDENT_TEST_DIR = RAW_DATA_DIR / "student_test"
SAMPLE_TRIP_DIR = TRAIN_DIR / "00000001"


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
        self.assertEqual(cleaned["track"]["time"].tolist(), [1, 2])
        pd.testing.assert_frame_equal(cleaned["nodes"], original_nodes)
        pd.testing.assert_frame_equal(cleaned["arcs"], original_arcs)
        pd.testing.assert_frame_equal(cleaned["route"], original_route)
        self.assertEqual(len(trip["track"]), 5)

    def test_clean_trip_with_stats_reports_removed_track_rows(self) -> None:
        trip = make_trip(
            pd.DataFrame(
                {
                    "longitude": [-122.1, -122.1, 181],
                    "latitude": [37.4, 37.4, 37.5],
                    "time": [1, 1, 2],
                }
            )
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cleaned, stats = clean_trip_with_stats(trip)

        self.assertEqual(len(cleaned["track"]), 1)
        self.assertEqual(stats["invalid_track_rows"], 1)
        self.assertEqual(stats["duplicate_track_rows"], 1)

    def test_data_quality_report_writes_markdown_and_chart(self) -> None:
        trip = TripDataSummary(
            trip_id="trip-1",
            raw_table_rows={"track": 3, "nodes": 2, "arcs": 1, "route": 1},
            cleaned_table_rows={"track": 1, "nodes": 2, "arcs": 1, "route": 1},
            invalid_track_rows=1,
            duplicate_track_rows=1,
            raw_validation_issues=("'track.longitude' has invalid values.",),
            post_clean_validation_issues=(),
            status="Included in model dataset",
        )
        partition = PartitionDataSummary(
            name="train",
            source_dir=Path("train"),
            trips=[trip],
            candidate_rows=4,
            candidate_coverage=0.75,
            observation_count=1,
            sampled_observation_count=1,
            model_trip_count=1,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            paths = write_data_quality_report(
                [partition],
                output_dir,
                "All observations used.",
            )

            markdown = paths["markdown_path"].read_text(encoding="utf-8")
            chart = paths["chart_path"].read_bytes()

        self.assertIn("Data cleaning and validation report", markdown)
        self.assertIn("invalid values", markdown)
        self.assertIn("data_quality_summary.png", markdown)
        self.assertTrue(chart.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_clean_trips_preserves_trip_ids_and_rejects_empty_clean_result(self) -> None:
        trips = {"trip-1": make_trip()}
        self.assertEqual(list(clean_trips(trips)), ["trip-1"])

        empty_route_trip = make_trip()
        empty_route_trip["route"] = pd.DataFrame(columns=["arc_id"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = clean_trips(
                {"trip-1": make_trip(), "empty-trip": empty_route_trip}
            )
        self.assertEqual(list(cleaned), ["trip-1"])
        self.assertIn("Dropping trip empty-trip", str(caught[0].message))

        invalid_trip = make_trip(
            pd.DataFrame(
                {"longitude": [181], "latitude": [37.4], "time": [1]}
            )
        )
        with self.assertRaisesRegex(ValueError, "No valid GPS observations"):
            clean_trip(invalid_trip)

    def test_empty_file_is_validated_and_its_trip_is_dropped(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            trip_dir = Path(temp_dir) / "00000002"
            trip_dir.mkdir()
            (trip_dir / "00000002.track").write_text(
                "-122.1 37.4 1\n", encoding="utf-8"
            )
            (trip_dir / "00000002.nodes").write_text(
                "-122.1 37.4\n", encoding="utf-8"
            )
            (trip_dir / "00000002.arcs").write_text("0 0\n", encoding="utf-8")
            (trip_dir / "00000002.route").write_text("", encoding="utf-8")

            trip = load_trip(trip_dir)

        self.assertTrue(trip["route"].empty)
        self.assertTrue(
            any("route" in issue and "empty" in issue for issue in validate_trip(trip))
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = clean_trips({"00000002": trip, "valid-trip": make_trip()})
        self.assertEqual(list(cleaned), ["valid-trip"])
        self.assertIn("Dropping trip 00000002", str(caught[0].message))

    @unittest.skipUnless(
        TRAIN_DIR.is_dir() and STUDENT_TEST_DIR.is_dir(),
        "Real dataset partitions are not available.",
    )
    def test_real_partition_counts(self) -> None:
        train_count = sum(path.is_dir() for path in TRAIN_DIR.iterdir())
        student_test_count = sum(
            path.is_dir() for path in STUDENT_TEST_DIR.iterdir()
        )

        self.assertEqual(train_count, 80)
        self.assertEqual(student_test_count, 10)

    @unittest.skipUnless(
        SAMPLE_TRIP_DIR.is_dir(),
        "Real training trip data is not available.",
    )
    def test_real_training_trip_preview(self) -> None:
        trip = load_trip(SAMPLE_TRIP_DIR)
        raw_track_count = len(trip["track"])
        issues = validate_trip(trip)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cleaned = clean_trip(trip)

        self.assertGreater(raw_track_count, 0)
        self.assertGreater(len(cleaned["track"]), 0)
        self.assertEqual(raw_track_count, len(trip["track"]))

        print("\nReal dataset sample: train/00000001")
        for table_name, frame in trip.items():
            print(f"\n{table_name}: {frame.shape[0]} rows")
            print(frame.head(3).to_string(index=False))
        print(f"\nRaw validation issues: {issues or 'none'}")
        print(
            "Cleaning warnings:",
            [str(warning.message) for warning in caught] or "none",
        )
        print("\nCleaned track preview:")
        print(cleaned["track"].head(3).to_string(index=False))


if __name__ == "__main__":
    unittest.main()
