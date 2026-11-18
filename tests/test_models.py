import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.models.evaluate import evaluate_classifier
from src.models.registry import MODEL_BUILDERS, build_model_registry
from src.models.report import generate_model_reports
from src.models.train import cross_validate_models, train_model


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.0, 2.0],
                [1.0, 2.0],
                [0.0, 3.0],
                [1.0, 3.0],
            ]
        )
        self.y = np.array([0, 1] * 4)
        self.observation_ids = [
            "obs-1",
            "obs-1",
            "obs-2",
            "obs-2",
            "obs-3",
            "obs-3",
            "obs-4",
            "obs-4",
        ]

    def test_registry_builds_selected_models(self) -> None:
        models = build_model_registry(
            model_names=("logistic_regression", "random_forest", "knn", "mlp")
        )

        self.assertEqual(
            set(models),
            {"logistic_regression", "random_forest", "knn", "mlp"},
        )
        self.assertIn("lightgbm", MODEL_BUILDERS)
        self.assertIsInstance(models["random_forest"], RandomForestClassifier)
        self.assertTrue(
            all(
                isinstance(models[name], Pipeline)
                for name in ("logistic_regression", "knn", "mlp")
            )
        )
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            build_model_registry(model_names=("not_a_model",))

    def test_available_model_builders_fit_and_return_probabilities(self) -> None:
        models = build_model_registry(
            model_names=("logistic_regression", "random_forest", "knn", "mlp")
        )

        for model_name, model in models.items():
            with self.subTest(model=model_name):
                fitted_model = train_model(model, self.X, self.y)
                probabilities = fitted_model.predict_proba(self.X)
                self.assertEqual(probabilities.shape, (len(self.y), 2))
                self.assertTrue(np.isfinite(probabilities).all())

    def test_lightgbm_builder_fits_when_dependency_is_installed(self) -> None:
        try:
            model = build_model_registry(model_names=("lightgbm",))["lightgbm"]
        except ImportError as exc:
            self.skipTest(str(exc))

        fitted_model = train_model(model, self.X, self.y)
        probabilities = fitted_model.predict_proba(self.X)
        self.assertEqual(probabilities.shape, (len(self.y), 2))

    def test_report_generator_writes_comparison_csv_and_best_model_markdown(
        self,
    ) -> None:
        groups = [
            "trip-1",
            "trip-1",
            "trip-2",
            "trip-2",
            "trip-3",
            "trip-3",
            "trip-4",
            "trip-4",
        ]
        with tempfile.TemporaryDirectory() as output_dir:
            result = generate_model_reports(
                self.X,
                self.y,
                self.X,
                self.y,
                trip_groups=groups,
                output_dir=output_dir,
                model_names=("logistic_regression",),
                feature_names=("distance", "offset"),
                n_splits=2,
            )

            self.assertEqual(result["best_model"], "logistic_regression")
            with Path(result["csv_path"]).open(
                newline="", encoding="utf-8"
            ) as csv_file:
                report_row = next(csv.DictReader(csv_file))
            self.assertEqual(report_row["is_best"], "True")
            self.assertGreater(float(report_row["validation_macro_f1"]), 0.9)
            self.assertEqual(
                len(json.loads(report_row["confusion_matrix_json"])),
                2,
            )
            self.assertIn("distance", report_row["feature_importance_json"])
            markdown = Path(result["markdown_path"]).read_text(encoding="utf-8")
            self.assertIn("# Best model report: logistic_regression", markdown)
            self.assertIn("Cross-validation", markdown)

    def test_train_and_evaluate_returns_candidate_and_top1_metrics(self) -> None:
        model = train_model(
            build_model_registry(model_names=("logistic_regression",))[
                "logistic_regression"
            ],
            self.X,
            self.y,
        )

        metrics = evaluate_classifier(
            model, self.X, self.y, observation_ids=self.observation_ids
        )

        self.assertEqual(metrics["candidate_accuracy"], 1.0)
        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertGreater(metrics["inference_seconds"], 0.0)

    def test_cross_validation_keeps_trip_groups_and_aggregates_metrics(self) -> None:
        rng = np.random.default_rng(42)
        y = np.tile([0, 1, 0, 1], 8)
        X = np.column_stack(
            [y + rng.normal(0, 0.05, len(y)), rng.normal(size=len(y))]
        )
        groups = np.repeat([f"trip-{i}" for i in range(8)], 4)

        results = cross_validate_models(
            X,
            y,
            groups,
            model_names=("logistic_regression",),
            n_splits=4,
        )

        self.assertEqual(len(results["logistic_regression"]["folds"]), 4)
        self.assertIn(
            "candidate_accuracy",
            results["logistic_regression"]["mean"],
        )
        self.assertIn("fit_seconds", results["logistic_regression"]["std"])

    def test_cross_validation_rejects_misaligned_trip_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "one trip ID"):
            cross_validate_models(
                self.X,
                self.y,
                ["trip-1"],
                model_names=("logistic_regression",),
            )


if __name__ == "__main__":
    unittest.main()
