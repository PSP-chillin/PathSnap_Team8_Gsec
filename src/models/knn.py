"""K-nearest-neighbours candidate-arc classifier."""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_knn(random_state: int = 42) -> Pipeline:
    """Create a scaled, distance-weighted KNN classifier."""
    del random_state
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    n_jobs=-1,
                ),
            ),
        ]
    )