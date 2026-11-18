"""Training helpers for caller-prepared candidate feature matrices."""

from collections.abc import Sequence
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.validation import check_X_y

from .evaluate import evaluate_classifier
from .registry import build_model_registry


def train_model(
    model: BaseEstimator,
    X: ArrayLike,
    y: ArrayLike,
) -> BaseEstimator:
    """Fit one model on candidate features and binary/multiclass labels."""
    X_checked, y_checked = check_X_y(X, y)
    return model.fit(X_checked, y_checked)


def cross_validate_models(
    X: ArrayLike,
    y: ArrayLike,
    groups: Sequence[object],
    model_names: Sequence[str] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict[str, dict[str, object]]:
    """Compare models with stratified folds that keep each trip in one fold."""
    X_checked, y_checked = check_X_y(X, y)
    groups_checked = np.asarray(groups)
    if groups_checked.ndim != 1 or len(groups_checked) != len(y_checked):
        raise ValueError("groups must have one trip ID per candidate row.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(np.unique(groups_checked)) < n_splits:
        raise ValueError("n_splits cannot exceed the number of unique trips.")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    results: dict[str, dict[str, object]] = {}
    models = build_model_registry(random_state, model_names)

    for model_name, model in models.items():
        fold_metrics: list[dict[str, float | None]] = []
        for train_indices, validation_indices in splitter.split(
            X_checked, y_checked, groups_checked
        ):
            fold_model = clone(model)
            started_at = perf_counter()
            fold_model.fit(X_checked[train_indices], y_checked[train_indices])
            fit_seconds = perf_counter() - started_at
            metrics = evaluate_classifier(
                fold_model,
                X_checked[validation_indices],
                y_checked[validation_indices],
            )
            metrics["fit_seconds"] = fit_seconds
            fold_metrics.append(metrics)

        metric_names = fold_metrics[0].keys()
        means = {
            metric: float(np.mean(values))
            for metric in metric_names
            if (values := [
                fold[metric]
                for fold in fold_metrics
                if fold[metric] is not None
            ])
        }
        standard_deviations = {
            metric: float(np.std(values))
            for metric in metric_names
            if (values := [
                fold[metric]
                for fold in fold_metrics
                if fold[metric] is not None
            ])
        }
        results[model_name] = {
            "folds": fold_metrics,
            "mean": means,
            "std": standard_deviations,
        }

    return results