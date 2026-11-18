"""Generate a comparison CSV and a detailed report for the best model."""

import csv
import json
from collections.abc import Hashable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_X_y

from .evaluate import evaluate_classifier
from .registry import build_model_registry
from .train import cross_validate_models, train_model


_CSV_COLUMNS = (
    "model",
    "is_best",
    "selection_metric",
    "validation_accuracy",
    "validation_balanced_accuracy",
    "validation_macro_precision",
    "validation_macro_recall",
    "validation_macro_f1",
    "validation_weighted_f1",
    "validation_log_loss",
    "validation_roc_auc",
    "validation_top1_accuracy",
    "cv_macro_f1_mean",
    "cv_macro_f1_std",
    "cv_candidate_accuracy_mean",
    "cv_candidate_accuracy_std",
    "cv_log_loss_mean",
    "cv_log_loss_std",
    "cv_fit_seconds_mean",
    "cv_fit_seconds_std",
    "cv_inference_seconds_mean",
    "cv_inference_seconds_std",
    "cv_fold_count",
    "random_state",
    "training_seconds",
    "inference_seconds",
    "train_rows",
    "validation_rows",
    "feature_count",
    "train_candidate_coverage",
    "validation_candidate_coverage",
    "sampling_description",
    "class_labels_json",
    "train_class_counts_json",
    "validation_class_counts_json",
    "confusion_matrix_json",
    "per_class_metrics_json",
    "parameters_json",
    "feature_importance_json",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseEstimator):
        return repr(value)
    return repr(value)


def _model_feature_importance(
    model: BaseEstimator,
    feature_names: Sequence[str],
) -> dict[str, float] | None:
    estimator = model.steps[-1][1] if isinstance(model, Pipeline) else model
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        importances = np.abs(coefficients).mean(axis=0)
    else:
        return None

    return {
        name: float(value)
        for name, value in sorted(
            zip(feature_names, importances),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def _markdown_report(
    model_name: str,
    record: dict[str, Any],
    cv_result: dict[str, Any],
    classes: np.ndarray,
) -> str:
    confusion = json.loads(record["confusion_matrix_json"])
    per_class = json.loads(record["per_class_metrics_json"])
    params = json.loads(record["parameters_json"])
    importance = json.loads(record["feature_importance_json"])
    train_class_counts = json.loads(record["train_class_counts_json"])
    validation_class_counts = json.loads(record["validation_class_counts_json"])
    train_coverage = record["train_candidate_coverage"]
    validation_coverage = record["validation_candidate_coverage"]
    train_coverage_text = "N/A" if train_coverage is None else f"{train_coverage:.2%}"
    validation_coverage_text = (
        "N/A" if validation_coverage is None else f"{validation_coverage:.2%}"
    )

    lines = [
        f"# Best model report: {model_name}",
        "",
        "Selected by validation macro-F1. Use the student-test partition for "
        "model comparison; do not use evaluator labels for model selection.",
        "",
        "## Validation metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric in (
        "validation_accuracy",
        "validation_balanced_accuracy",
        "validation_macro_precision",
        "validation_macro_recall",
        "validation_macro_f1",
        "validation_weighted_f1",
        "validation_log_loss",
        "validation_roc_auc",
        "validation_top1_accuracy",
        "training_seconds",
        "inference_seconds",
    ):
        value = record[metric]
        formatted = "N/A" if value is None else f"{value:.6f}"
        lines.append(f"| {metric} | {formatted} |")

    lines.extend(
        [
            "",
            "## Dataset summary",
            "",
            f"- Training candidate rows: {record['train_rows']}",
            f"- Validation candidate rows: {record['validation_rows']}",
            f"- Numeric features: {record['feature_count']}",
            f"- Training candidate coverage: {train_coverage_text}",
            f"- Validation candidate coverage: {validation_coverage_text}",
            f"- Sampling: {record['sampling_description']}",
            f"- Training label counts: `{train_class_counts}`",
            f"- Validation label counts: `{validation_class_counts}`",
            f"- Random state: {record['random_state']}",
            "",
            "## Cross-validation",
            "",
            "| Metric | Mean | Std |",
            "| --- | ---: | ---: |",
        ]
    )
    for metric in (
        "macro_f1",
        "candidate_accuracy",
        "log_loss",
        "fit_seconds",
        "inference_seconds",
    ):
        mean = cv_result["mean"].get(metric)
        std = cv_result["std"].get(metric)
        mean_text = "N/A" if mean is None else f"{mean:.6f}"
        std_text = "N/A" if std is None else f"{std:.6f}"
        lines.append(f"| {metric} | {mean_text} | {std_text} |")
    lines.extend(
        [
            "",
            "### Per-fold scores",
            "",
            "| Fold | Macro-F1 | Accuracy | Log loss | Fit seconds | Inference seconds |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for fold_number, fold in enumerate(cv_result["folds"], start=1):
        lines.append(
            f"| {fold_number} | {fold['macro_f1']:.6f} | "
            f"{fold['candidate_accuracy']:.6f} | {fold['log_loss']:.6f} | "
            f"{fold['fit_seconds']:.6f} | {fold['inference_seconds']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            f"Class order: `{classes.tolist()}`",
            "",
            "```json",
            json.dumps(confusion, indent=2),
            "```",
            "",
            "## Per-class metrics",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, values in per_class.items():
        if isinstance(values, dict) and "precision" in values:
            lines.append(
                f"| {label} | {values['precision']:.6f} | "
                f"{values['recall']:.6f} | {values['f1-score']:.6f} | "
                f"{values['support']} |"
            )

    lines.extend(
        [
            "",
            "## Model parameters",
            "",
            "```json",
            json.dumps(params, indent=2, sort_keys=True),
            "```",
            "",
            "## Feature importance",
            "",
        ]
    )
    if importance is None:
        lines.append("This estimator does not expose a supported importance measure.")
    else:
        lines.extend(
            [
                "| Feature | Importance |",
                "| --- | ---: |",
                *[
                    f"| {name} | {value:.6f} |"
                    for name, value in importance.items()
                ],
            ]
        )
    lines.extend(["", ""])
    return "\n".join(lines)


def generate_model_reports(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_validation: ArrayLike,
    y_validation: ArrayLike,
    trip_groups: Sequence[object],
    output_dir: str | Path = "results/model_reports",
    model_names: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    observation_ids: Sequence[Hashable] | None = None,
    train_candidate_coverage: float | None = None,
    validation_candidate_coverage: float | None = None,
    sampling_description: str = "No observation subsampling applied.",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train and compare models; write a CSV and a Markdown report for the best.

    The comparison data should be the student-test partition, not the
    evaluator partition. ``trip_groups`` must contain the trip ID for each
    training candidate row so cross-validation cannot split a trip across folds.
    """
    X_train_checked, y_train_checked = check_X_y(X_train, y_train)
    X_validation_checked, y_validation_checked = check_X_y(
        X_validation, y_validation
    )
    if X_train_checked.shape[1] != X_validation_checked.shape[1]:
        raise ValueError("Training and validation data must have the same features.")

    if feature_names is None:
        if hasattr(X_train, "columns"):
            names = [str(name) for name in X_train.columns]
        else:
            names = [f"feature_{index}" for index in range(X_train_checked.shape[1])]
    else:
        names = [str(name) for name in feature_names]
    if len(names) != X_train_checked.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match the feature count.")
    for coverage in (train_candidate_coverage, validation_candidate_coverage):
        if coverage is not None and (
            not np.isfinite(coverage) or not 0 <= coverage <= 1
        ):
            raise ValueError("Candidate coverage values must be between 0 and 1.")

    models = build_model_registry(random_state, model_names)
    selected_names = tuple(models)
    cv_results = cross_validate_models(
        X_train_checked,
        y_train_checked,
        groups=trip_groups,
        model_names=selected_names,
        n_splits=n_splits,
        random_state=random_state,
    )

    records: list[dict[str, Any]] = []
    model_classes: dict[str, np.ndarray] = {}
    for model_name, model in models.items():
        started_at = perf_counter()
        fitted_model = train_model(model, X_train_checked, y_train_checked)
        training_seconds = perf_counter() - started_at

        validation_metrics = evaluate_classifier(
            fitted_model,
            X_validation_checked,
            y_validation_checked,
            observation_ids=observation_ids,
        )
        predictions = fitted_model.predict(X_validation_checked)
        probabilities = fitted_model.predict_proba(X_validation_checked)
        classes = np.asarray(fitted_model.classes_)
        model_classes[model_name] = classes

        if len(np.unique(y_validation_checked)) == len(classes) and len(classes) > 1:
            if len(classes) == 2:
                roc_auc = float(roc_auc_score(y_validation_checked, probabilities[:, 1]))
            else:
                roc_auc = float(
                    roc_auc_score(
                        y_validation_checked,
                        probabilities,
                        labels=classes,
                        multi_class="ovr",
                        average="macro",
                    )
                )
        else:
            roc_auc = None

        cv_mean = cv_results[model_name]["mean"]
        cv_std = cv_results[model_name]["std"]
        train_labels, train_counts = np.unique(y_train_checked, return_counts=True)
        validation_labels, validation_counts = np.unique(
            y_validation_checked, return_counts=True
        )
        per_class = classification_report(
            y_validation_checked,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
        feature_importance = _model_feature_importance(fitted_model, names)
        records.append(
            {
                "model": model_name,
                "is_best": False,
                "selection_metric": "validation_macro_f1",
                "validation_accuracy": validation_metrics["candidate_accuracy"],
                "validation_balanced_accuracy": float(
                    balanced_accuracy_score(y_validation_checked, predictions)
                ),
                "validation_macro_precision": float(
                    precision_score(
                        y_validation_checked,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "validation_macro_recall": float(
                    recall_score(
                        y_validation_checked,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "validation_macro_f1": validation_metrics["macro_f1"],
                "validation_weighted_f1": float(
                    f1_score(
                        y_validation_checked,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "validation_log_loss": validation_metrics["log_loss"],
                "validation_roc_auc": roc_auc,
                "validation_top1_accuracy": validation_metrics["top1_accuracy"],
                "cv_macro_f1_mean": cv_mean.get("macro_f1"),
                "cv_macro_f1_std": cv_std.get("macro_f1"),
                "cv_candidate_accuracy_mean": cv_mean.get("candidate_accuracy"),
                "cv_candidate_accuracy_std": cv_std.get("candidate_accuracy"),
                "cv_log_loss_mean": cv_mean.get("log_loss"),
                "cv_log_loss_std": cv_std.get("log_loss"),
                "cv_fit_seconds_mean": cv_mean.get("fit_seconds"),
                "cv_fit_seconds_std": cv_std.get("fit_seconds"),
                "cv_inference_seconds_mean": cv_mean.get("inference_seconds"),
                "cv_inference_seconds_std": cv_std.get("inference_seconds"),
                "cv_fold_count": n_splits,
                "random_state": random_state,
                "training_seconds": training_seconds,
                "inference_seconds": validation_metrics["inference_seconds"],
                "train_rows": len(y_train_checked),
                "validation_rows": len(y_validation_checked),
                "feature_count": len(names),
                "train_candidate_coverage": train_candidate_coverage,
                "validation_candidate_coverage": validation_candidate_coverage,
                "sampling_description": sampling_description,
                "class_labels_json": json.dumps(classes.tolist(), default=_json_default),
                "train_class_counts_json": json.dumps(
                    {
                        str(label): int(count)
                        for label, count in zip(train_labels, train_counts)
                    }
                ),
                "validation_class_counts_json": json.dumps(
                    {
                        str(label): int(count)
                        for label, count in zip(validation_labels, validation_counts)
                    }
                ),
                "confusion_matrix_json": json.dumps(
                    confusion_matrix(
                        y_validation_checked,
                        predictions,
                        labels=classes,
                    ).tolist()
                ),
                "per_class_metrics_json": json.dumps(per_class, default=_json_default),
                "parameters_json": json.dumps(
                    fitted_model.get_params(deep=True),
                    default=_json_default,
                    sort_keys=True,
                ),
                "feature_importance_json": json.dumps(
                    feature_importance,
                    default=_json_default,
                    sort_keys=True,
                ),
            }
        )

    best_record = max(records, key=lambda record: record["validation_macro_f1"])
    best_record["is_best"] = True
    best_name = best_record["model"]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "model_comparison.csv"
    markdown_path = output_path / "best_model_report.md"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)

    markdown_path.write_text(
        _markdown_report(
            best_name,
            best_record,
            cv_results[best_name],
            model_classes[best_name],
        ),
        encoding="utf-8",
    )

    return {
        "best_model": best_name,
        "selection_metric": "validation_macro_f1",
        "csv_path": csv_path,
        "markdown_path": markdown_path,
        "reports": records,
    }
