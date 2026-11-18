"""Tune the MLP on grouped training folds and evaluate on student-test data."""

import argparse
import json
from collections.abc import Hashable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import optuna
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.validation import check_X_y

from report import FEATURE_COLUMNS, build_partition_dataset
from .mlp import build_mlp


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED = 42
ARCHITECTURES = {
    "32": (32,),
    "64": (64,),
    "128": (128,),
    "64_32": (64, 32),
    "128_64": (128, 64),
}


def build_grouped_splits(
    X: ArrayLike,
    y: ArrayLike,
    trip_groups: Sequence[object],
    n_splits: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create stratified folds that keep every trip entirely in one fold."""
    X_checked, y_checked = check_X_y(X, y)
    groups = np.asarray(trip_groups)
    if groups.ndim != 1 or len(groups) != len(y_checked):
        raise ValueError("trip_groups must contain one trip ID per candidate row.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(np.unique(groups)) < n_splits:
        raise ValueError("n_splits cannot exceed the number of unique trips.")
    if not np.isin(y_checked, [0, 1]).all():
        raise ValueError("MLP tuning requires binary labels 0 and 1.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(X_checked, y_checked, groups))
    for train_indices, validation_indices in splits:
        train_trips = set(groups[train_indices])
        validation_trips = set(groups[validation_indices])
        if train_trips & validation_trips:
            raise RuntimeError("A trip was split across a training fold.")
        if not np.array_equal(
            np.unique(y_checked[train_indices]),
            np.array([0, 1]),
        ):
            raise ValueError("A training fold lacks one of the two label classes.")
    return splits


def score_predictions(
    y_true: ArrayLike,
    predictions: ArrayLike,
    probabilities: ArrayLike,
    observation_ids: Sequence[Hashable],
) -> dict[str, float]:
    """Score candidate classification and the top-ranked arc per observation."""
    y_checked = np.asarray(y_true)
    predictions_checked = np.asarray(predictions)
    probabilities_checked = np.asarray(probabilities, dtype=float)
    if y_checked.ndim != 1 or predictions_checked.shape != y_checked.shape:
        raise ValueError("Predictions must have one binary label per candidate.")
    if probabilities_checked.shape != (len(y_checked), 2):
        raise ValueError("Expected one probability column for each binary class.")
    if len(observation_ids) != len(y_checked):
        raise ValueError("observation_ids must have one value per candidate.")
    if not np.isin(y_checked, [0, 1]).all():
        raise ValueError("MLP tuning requires binary labels 0 and 1.")
    if not np.isfinite(probabilities_checked).all():
        raise ValueError("Predicted probabilities must be finite.")

    candidates_by_observation: dict[Hashable, list[int]] = {}
    for index, observation_id in enumerate(observation_ids):
        candidates_by_observation.setdefault(observation_id, []).append(index)
    if not candidates_by_observation:
        raise ValueError("Cannot score an empty candidate set.")

    top1_correct = 0
    for indices in candidates_by_observation.values():
        best_index = max(
            indices,
            key=lambda index: probabilities_checked[index, 1],
        )
        top1_correct += int(y_checked[best_index] == 1)

    return {
        "macro_f1": float(
            f1_score(
                y_checked,
                predictions_checked,
                labels=[0, 1],
                average="macro",
                zero_division=0,
            )
        ),
        "candidate_accuracy": float(
            accuracy_score(y_checked, predictions_checked)
        ),
        "candidate_log_loss": float(
            log_loss(y_checked, probabilities_checked, labels=[0, 1])
        ),
        "top1_accuracy": top1_correct / len(candidates_by_observation),
    }


def make_model(
    params: dict[str, Any] | None = None,
    random_state: int = DEFAULT_SEED,
):
    """Build the standard-scaled MLP, optionally overriding classifier settings."""
    model = build_mlp(random_state=random_state)
    if params:
        model.set_params(
            **{
                f"classifier__{name}": value
                for name, value in params.items()
            }
        )
    return model


def trial_parameters(trial: optuna.Trial) -> dict[str, Any]:
    """Sample a practical MLP architecture and optimizer configuration."""
    architecture = trial.suggest_categorical(
        "architecture",
        list(ARCHITECTURES),
    )
    return {
        "hidden_layer_sizes": ARCHITECTURES[architecture],
        "activation": trial.suggest_categorical(
            "activation",
            ["relu", "tanh"],
        ),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
        "learning_rate_init": trial.suggest_float(
            "learning_rate_init",
            1e-4,
            5e-3,
            log=True,
        ),
        "batch_size": trial.suggest_categorical(
            "batch_size",
            [64, 128, 256],
        ),
        "max_iter": trial.suggest_categorical("max_iter", [100, 200]),
        "early_stopping": True,
    }


def cross_validate_mlp(
    X: ArrayLike,
    y: ArrayLike,
    observation_ids: Sequence[Hashable],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    params: dict[str, Any] | None = None,
    random_state: int = DEFAULT_SEED,
    trial: optuna.Trial | None = None,
) -> dict[str, Any]:
    """Evaluate an MLP configuration using precomputed trip-grouped folds."""
    X_checked, y_checked = check_X_y(X, y)
    if len(observation_ids) != len(y_checked):
        raise ValueError("observation_ids must have one value per candidate.")
    if not splits:
        raise ValueError("At least one cross-validation fold is required.")

    fold_results = []
    fold_macro_f1 = []
    for fold_number, (train_indices, validation_indices) in enumerate(splits):
        if not np.array_equal(
            np.unique(y_checked[train_indices]),
            np.array([0, 1]),
        ):
            raise ValueError("A training fold lacks one of the two label classes.")

        model = make_model(params, random_state=random_state)
        started_at = perf_counter()
        model.fit(X_checked[train_indices], y_checked[train_indices])
        fit_seconds = perf_counter() - started_at
        probabilities = model.predict_proba(X_checked[validation_indices])
        metrics = score_predictions(
            y_checked[validation_indices],
            model.predict(X_checked[validation_indices]),
            probabilities,
            [observation_ids[index] for index in validation_indices],
        )
        metrics["fit_seconds"] = fit_seconds
        fold_results.append(metrics)
        fold_macro_f1.append(metrics["macro_f1"])

        if trial is not None:
            trial.report(
                float(np.mean(fold_macro_f1)),
                step=fold_number,
            )
            if trial.should_prune():
                raise optuna.TrialPruned()

    metric_names = fold_results[0].keys()
    return {
        "folds": fold_results,
        "mean": {
            metric: float(np.mean([fold[metric] for fold in fold_results]))
            for metric in metric_names
        },
        "std": {
            metric: float(np.std([fold[metric] for fold in fold_results]))
            for metric in metric_names
        },
    }


def _fit_and_score(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    observation_ids: Sequence[Hashable],
    params: dict[str, Any] | None,
    random_state: int,
) -> tuple[Any, dict[str, float]]:
    model = make_model(params, random_state=random_state)
    started_at = perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = perf_counter() - started_at
    metrics = score_predictions(
        y_validation,
        model.predict(X_validation),
        model.predict_proba(X_validation),
        observation_ids,
    )
    metrics["fit_seconds"] = fit_seconds
    return model, metrics


def _format_score(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _write_markdown(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    baseline_cv = report["baseline_cv"]
    tuned_cv = report["tuned_cv"]
    lines = [
        "# MLP hyperparameter tuning report",
        "",
        "Hyperparameters were selected using the mean macro-F1 from "
        "stratified, trip-grouped cross-validation on the training partition. "
        "The student-test partition was not used during the search.",
        "",
        "## Cross-validation results",
        "",
        "| Configuration | Mean macro-F1 | Std macro-F1 | Mean top-1 accuracy |",
        "| --- | ---: | ---: | ---: |",
        f"| Baseline MLP | {_format_score(baseline_cv['mean']['macro_f1'])} | "
        f"{_format_score(baseline_cv['std']['macro_f1'])} | "
        f"{_format_score(baseline_cv['mean']['top1_accuracy'])} |",
        f"| Tuned MLP | {_format_score(tuned_cv['mean']['macro_f1'])} | "
        f"{_format_score(tuned_cv['std']['macro_f1'])} | "
        f"{_format_score(tuned_cv['mean']['top1_accuracy'])} |",
        "",
        f"- Trials: {report['completed_trials']} completed, "
        f"{report['pruned_trials']} pruned, "
        f"{report['total_trials']} total.",
        f"- Search time: {report['search_seconds']:.2f} seconds",
        f"- Training trips: {report['train_trip_count']}",
        f"- Training candidate rows: {report['train_candidate_rows']}",
        f"- Training candidate coverage: "
        f"{report['train_candidate_coverage']:.2%}",
        f"- Sampling: {report['sampling_description']}",
    ]

    student_test = report.get("student_test")
    if student_test is not None:
        lines.extend(
            [
                "",
                "## Final student-test comparison",
                "",
                "These scores are a final comparison after the grouped-CV "
                "search; they did not choose the hyperparameters.",
                "",
                "| Configuration | Macro-F1 | Candidate accuracy | Top-1 accuracy | Log loss | Fit seconds |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in ("baseline", "tuned"):
            metrics = student_test[name]
            lines.append(
                f"| {name.title()} MLP | "
                f"{_format_score(metrics['macro_f1'])} | "
                f"{_format_score(metrics['candidate_accuracy'])} | "
                f"{_format_score(metrics['top1_accuracy'])} | "
                f"{_format_score(metrics['candidate_log_loss'])} | "
                f"{metrics['fit_seconds']:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Selected hyperparameters",
            "",
            "```json",
            json.dumps(report["tuned_params"], indent=2, sort_keys=True),
            "```",
            "",
            "## Run outputs",
            "",
            f"- Trial history: `{report['trials_csv']}`",
            f"- Machine-readable report: `{report['json_path']}`",
            "",
        ]
    )
    if report["model_path"] is not None:
        lines.insert(
            -1,
            f"- Fitted tuned model: `{report['model_path']}`",
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune the MLP with grouped training CV and compare on student-test."
        )
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "train",
        help="Training partition directory.",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=ROOT / "data" / "raw" / "student_test",
        help="Held-out student-test partition used only after tuning.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "mlp_tuning",
        help="Directory for tuning outputs.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of trip-grouped cross-validation folds.",
    )
    parser.add_argument(
        "--max-observations-per-trip",
        type=int,
        default=25,
        help="Observations retained per trip; 0 uses all observations.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional search timeout in seconds.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_SEED,
        help="Seed for data splits, Optuna, and MLP initialization.",
    )
    parser.add_argument(
        "--cv-only",
        action="store_true",
        help="Skip the student-test evaluation and final model fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    if args.trials < 1:
        raise ValueError("--trials must be at least 1.")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if args.max_observations_per_trip < 0:
        raise ValueError("--max-observations-per-trip cannot be negative.")
    if args.timeout is not None and args.timeout <= 0:
        raise ValueError("--timeout must be positive.")
    args.out.mkdir(parents=True, exist_ok=True)

    sample_limit = args.max_observations_per_trip or None
    sampling_description = (
        "All observations used."
        if sample_limit is None
        else (
            f"At most {sample_limit} evenly spaced observations per trip; "
            "all candidates for selected observations retained."
        )
    )
    training = build_partition_dataset(args.train_dir, sample_limit)
    splits = build_grouped_splits(
        training.X,
        training.y,
        training.trip_groups,
        args.folds,
        args.random_state,
    )

    baseline_cv = cross_validate_mlp(
        training.X,
        training.y,
        training.observation_ids,
        splits,
        random_state=args.random_state,
    )
    sampler = optuna.samplers.TPESampler(seed=args.random_state)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, args.trials),
        n_warmup_steps=1,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    started_at = perf_counter()
    study.optimize(
        lambda trial: cross_validate_mlp(
            training.X,
            training.y,
            training.observation_ids,
            splits,
            params=trial_parameters(trial),
            random_state=args.random_state,
            trial=trial,
        )["mean"]["macro_f1"],
        n_trials=args.trials,
        timeout=args.timeout,
        n_jobs=1,
        show_progress_bar=True,
    )
    search_seconds = perf_counter() - started_at
    if not study.best_trials:
        raise RuntimeError("The tuning search completed without a successful trial.")

    best_trial = study.best_trial
    best_params = trial_parameters(optuna.trial.FixedTrial(best_trial.params))
    tuned_cv = cross_validate_mlp(
        training.X,
        training.y,
        training.observation_ids,
        splits,
        params=best_params,
        random_state=args.random_state,
    )
    trials_csv = args.out / "trials.csv"
    study.trials_dataframe().to_csv(
        trials_csv,
        index=False,
        lineterminator="\n",
    )

    report: dict[str, Any] = {
        "selection_metric": "grouped_cv_macro_f1",
        "random_state": args.random_state,
        "fold_count": args.folds,
        "total_trials": len(study.trials),
        "completed_trials": sum(
            trial.state == optuna.trial.TrialState.COMPLETE
            for trial in study.trials
        ),
        "pruned_trials": sum(
            trial.state == optuna.trial.TrialState.PRUNED
            for trial in study.trials
        ),
        "search_seconds": search_seconds,
        "feature_columns": list(FEATURE_COLUMNS),
        "sampling_description": sampling_description,
        "train_trip_count": training.trip_count,
        "train_candidate_rows": len(training.y),
        "train_candidate_coverage": training.candidate_coverage,
        "baseline_params": {
            key: value
            for key, value in build_mlp(
                random_state=args.random_state
            ).named_steps["classifier"].get_params().items()
            if key
            in {
                "hidden_layer_sizes",
                "activation",
                "alpha",
                "learning_rate_init",
                "batch_size",
                "max_iter",
                "early_stopping",
            }
        },
        "tuned_params": best_params,
        "optuna_best_params": best_trial.params,
        "baseline_cv": baseline_cv,
        "tuned_cv": tuned_cv,
        "trials_csv": str(trials_csv),
        "model_path": None,
    }

    if not args.cv_only:
        validation = build_partition_dataset(
            args.validation_dir,
            sample_limit,
        )
        training_trip_ids = {summary.trip_id for summary in training.data_quality.trips}
        validation_trip_ids = {
            summary.trip_id for summary in validation.data_quality.trips
        }
        overlap = training_trip_ids & validation_trip_ids
        if overlap:
            raise ValueError(
                "Trip IDs overlap between training and student-test partitions: "
                + ", ".join(sorted(overlap))
            )

        _, baseline_metrics = _fit_and_score(
            training.X,
            training.y,
            validation.X,
            validation.y,
            validation.observation_ids,
            params=None,
            random_state=args.random_state,
        )
        tuned_model, tuned_metrics = _fit_and_score(
            training.X,
            training.y,
            validation.X,
            validation.y,
            validation.observation_ids,
            params=best_params,
            random_state=args.random_state,
        )
        model_path = args.out / "tuned_mlp.joblib"
        joblib.dump(
            {
                "pipeline": tuned_model,
                "feature_columns": list(FEATURE_COLUMNS),
                "parameters": best_params,
            },
            model_path,
        )
        report["student_test"] = {
            "baseline": baseline_metrics,
            "tuned": tuned_metrics,
            "candidate_rows": len(validation.y),
            "trip_count": validation.trip_count,
            "candidate_coverage": validation.candidate_coverage,
        }
        report["model_path"] = str(model_path)

    json_path = args.out / "tuning_report.json"
    report["json_path"] = str(json_path)
    json_path.write_text(
        json.dumps(report, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    markdown_path = args.out / "tuning_report.md"
    _write_markdown(report, markdown_path)

    print(f"Tuning report: {markdown_path}")
    print(f"Trial history: {trials_csv}")
    if report["model_path"] is not None:
        print(f"Tuned MLP model: {report['model_path']}")
    print(
        "Best grouped-CV macro-F1: "
        f"{tuned_cv['mean']['macro_f1']:.4f} "
        f"(baseline {baseline_cv['mean']['macro_f1']:.4f})"
    )


if __name__ == "__main__":
    main()
