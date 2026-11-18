"""Multilayer-perceptron candidate-arc classifier."""

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_mlp(random_state: int = 42) -> Pipeline:
    """Create a scaled, fully connected neural-network classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    max_iter=100,
                    random_state=random_state,
                ),
            ),
        ]
    )