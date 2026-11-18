import unittest

import numpy as np
import optuna

from src.models.tune_mlp import (
    build_grouped_splits,
    score_predictions,
    trial_parameters,
)


class TuneMlpTests(unittest.TestCase):
    def test_grouped_splits_keep_each_trip_together(self) -> None:
        X = np.arange(64, dtype=float).reshape(32, 2)
        y = np.tile([0, 1], 16)
        trip_groups = np.repeat([f"trip-{index}" for index in range(8)], 4)

        splits = build_grouped_splits(
            X,
            y,
            trip_groups,
            n_splits=4,
            random_state=42,
        )

        self.assertEqual(len(splits), 4)
        for train_indices, validation_indices in splits:
            train_trips = set(trip_groups[train_indices])
            validation_trips = set(trip_groups[validation_indices])
            self.assertFalse(train_trips & validation_trips)

    def test_score_predictions_reports_candidate_and_observation_metrics(
        self,
    ) -> None:
        metrics = score_predictions(
            y_true=np.array([0, 1, 0, 1]),
            predictions=np.array([0, 1, 1, 1]),
            probabilities=np.array(
                [
                    [0.8, 0.2],
                    [0.2, 0.8],
                    [0.4, 0.6],
                    [0.3, 0.7],
                ]
            ),
            observation_ids=("obs-1", "obs-1", "obs-2", "obs-2"),
        )

        self.assertEqual(metrics["candidate_accuracy"], 0.75)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertGreater(metrics["macro_f1"], 0.0)
        self.assertGreater(metrics["candidate_log_loss"], 0.0)

    def test_trial_parameters_translate_architecture_to_mlp_shape(self) -> None:
        trial = optuna.trial.FixedTrial(
            {
                "architecture": "64_32",
                "activation": "relu",
                "alpha": 0.001,
                "learning_rate_init": 0.001,
                "batch_size": 128,
                "max_iter": 100,
            }
        )

        params = trial_parameters(trial)

        self.assertEqual(params["hidden_layer_sizes"], (64, 32))
        self.assertEqual(params["batch_size"], 128)
        self.assertTrue(params["early_stopping"])


if __name__ == "__main__":
    unittest.main()
