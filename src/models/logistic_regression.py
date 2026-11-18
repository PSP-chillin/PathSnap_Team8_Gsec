"""Logistic-regression baseline for candidate-arc classification."""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_regression(random_state: int = 42) -> Pipeline:
    """Create a scaled logistic-regression classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )