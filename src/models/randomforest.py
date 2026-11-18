"""Random-forest candidate-arc classifier."""

from sklearn.ensemble import RandomForestClassifier


def build_random_forest(random_state: int = 42) -> RandomForestClassifier:
    """Create a reproducible random-forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )