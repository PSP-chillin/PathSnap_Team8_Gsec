"""LightGBM candidate-arc classifier."""

from sklearn.base import BaseEstimator


def build_lightgbm(random_state: int = 42) -> BaseEstimator:
    """Create a LightGBM classifier, loading the optional dependency on demand."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "The LightGBM model requires the 'lightgbm' package. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from exc

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )