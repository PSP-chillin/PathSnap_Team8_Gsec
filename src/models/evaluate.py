"""Evaluation metrics for candidate classifiers."""

from collections.abc import Hashable, Sequence
from time import perf_counter

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.utils.validation import check_X_y


def evaluate_classifier(
    model: BaseEstimator,
    X: ArrayLike,
    y_true: ArrayLike,
    observation_ids: Sequence[Hashable] | None = None,
) -> dict[str, float | None]:
    """Evaluate a fitted classifier on candidate rows.

    If observation IDs are supplied, top-1 accuracy measures whether the
    highest-scored candidate for each GPS observation is labeled as a match (1).
    """
    X_checked, y_checked = check_X_y(X, y_true)
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        raise TypeError("Model must be a fitted classifier with predict_proba().")

    if observation_ids is not None and len(observation_ids) != len(y_checked):
        raise ValueError("observation_ids must have one value per candidate row.")

    started_at = perf_counter()
    predictions = model.predict(X_checked)
    probabilities = model.predict_proba(X_checked)
    inference_seconds = perf_counter() - started_at

    classes = np.asarray(model.classes_)
    metrics: dict[str, float | None] = {
        "macro_f1": float(f1_score(y_checked, predictions, average="macro")),
        "candidate_accuracy": float(accuracy_score(y_checked, predictions)),
        "log_loss": float(log_loss(y_checked, probabilities, labels=classes)),
        "top1_accuracy": None,
        "inference_seconds": inference_seconds,
    }

    if observation_ids is not None:
        if not np.isin(y_checked, [0, 1]).all():
            raise ValueError(
                "Observation-level top-1 accuracy requires binary labels "
                "(1=matching candidate, 0=non-matching candidate)."
            )
        positive_columns = np.flatnonzero(classes == 1)
        if len(positive_columns) != 1:
            raise ValueError("Model does not provide probabilities for class 1.")

        positive_scores = probabilities[:, positive_columns[0]]
        candidates_by_observation: dict[Hashable, list[int]] = {}
        for index, observation_id in enumerate(observation_ids):
            candidates_by_observation.setdefault(observation_id, []).append(index)

        correct_top1 = []
        for indices in candidates_by_observation.values():
            best_index = max(indices, key=lambda index: positive_scores[index])
            correct_top1.append(y_checked[best_index] == 1)
        metrics["top1_accuracy"] = float(np.mean(correct_top1))

    return metrics