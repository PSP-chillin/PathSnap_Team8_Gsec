"""Registry for the model experiments used by PathSnap."""

from collections.abc import Sequence
from typing import Callable

from sklearn.base import BaseEstimator

from .knn import build_knn
from .lgbmmodel import build_lightgbm
from .logistic_regression import build_logistic_regression
from .mlp import build_mlp
from .randomforest import build_random_forest


MODEL_BUILDERS: dict[str, Callable[[int], BaseEstimator]] = {
    "logistic_regression": build_logistic_regression,
    "random_forest": build_random_forest,
    "knn": build_knn,
    "mlp": build_mlp,
    "lightgbm": build_lightgbm,
}


def build_model_registry(
    random_state: int = 42,
    model_names: Sequence[str] | None = None,
) -> dict[str, BaseEstimator]:
    """Build selected models; by default, build all five configured models."""
    names = tuple(MODEL_BUILDERS) if model_names is None else tuple(model_names)
    unknown_names = sorted(set(names) - MODEL_BUILDERS.keys())
    if unknown_names:
        raise ValueError(f"Unknown model name(s): {', '.join(unknown_names)}.")
    if not names:
        raise ValueError("Select at least one model.")

    return {
        name: MODEL_BUILDERS[name](random_state)
        for name in names
    }