import tempfile
import unittest
from pathlib import Path

import numpy as np

from report import FEATURE_COLUMNS, build_partition_dataset


class ReportCliTests(unittest.TestCase):
    def test_build_partition_dataset_extracts_labeled_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            partition_dir = Path(temp_dir)
            trip_dir = partition_dir / "trip-1"
            trip_dir.mkdir()
            (trip_dir / "trip-1.nodes").write_text(
                "0 0\n0.001 0\n0.002 0\n0.001 0.001\n",
                encoding="utf-8",
            )
            (trip_dir / "trip-1.arcs").write_text(
                "0 1\n1 2\n2 1\n1 3\n",
                encoding="utf-8",
            )
            (trip_dir / "trip-1.track").write_text(
                "0.0002 0 0\n"
                "0.0006 0 1\n"
                "0.0011 0 2\n"
                "0.0015 0 3\n"
                "0.0018 0 4\n",
                encoding="utf-8",
            )
            (trip_dir / "trip-1.route").write_text("0\n1\n", encoding="utf-8")

            dataset = build_partition_dataset(partition_dir)

        self.assertEqual(dataset.trip_count, 1)
        self.assertEqual(dataset.observation_count, 5)
        self.assertEqual(dataset.X.shape[1], len(FEATURE_COLUMNS))
        self.assertEqual(len(dataset.X), len(dataset.y))
        self.assertEqual(len(dataset.trip_groups), len(dataset.y))
        self.assertTrue(np.isfinite(dataset.X).all())
        self.assertTrue((dataset.y == 1).any())
        self.assertGreater(dataset.candidate_coverage, 0)


if __name__ == "__main__":
    unittest.main()
