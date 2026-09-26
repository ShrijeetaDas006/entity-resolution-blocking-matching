"""
THE DELIVERABLE Person C consumes: a trained model + an inference function
that scores any candidate pair.

Agreed contract: score_pair(row) -> float
    - row: a dict or pandas Series with the raw candidate-pair fields
      (name1, name2, addr1, addr2, country1, country2, postal1, postal2)
    - returns: a float match probability between 0 and 1

Also provides score_batch(df) for scoring many pairs at once (what C's
pipeline will actually call in practice, for speed).
"""

import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
from similarity_features import add_similarity_features
from exact_match_flags import add_exact_match_flags
from train_model import FEATURE_COLUMNS, load_model

# Load once at import time so repeated calls don't reload from disk
_MODEL = None
_THRESHOLD = 0.01  # update after running threshold_tuning.py


def _get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model()
    return _MODEL


def score_pair(row: dict) -> float:
    """Scores a single candidate pair. row must have the raw fields
    (name1, name2, addr1, addr2, country1, country2, postal1, postal2)."""
    df = pd.DataFrame([row])
    df = add_similarity_features(df)
    df = add_exact_match_flags(df)
    model = _get_model()
    return float(model.predict_proba(df[FEATURE_COLUMNS])[0, 1])


def score_batch(df: pd.DataFrame) -> pd.Series:
    """Scores many candidate pairs at once -- use this in the pipeline,
    not a loop of score_pair(), for speed."""
    df = add_similarity_features(df)
    df = add_exact_match_flags(df)
    model = _get_model()
    return pd.Series(model.predict_proba(df[FEATURE_COLUMNS])[:, 1], index=df.index)


def predict_match(row: dict, threshold: float = None) -> int:
    """Returns 1/0 using the tuned threshold (from threshold_tuning.py)."""
    t = threshold if threshold is not None else _THRESHOLD
    return int(score_pair(row) >= t)


if __name__ == "__main__":
    example_row = {
        "id1": "S1_0001", "id2": "S2_0001",
        "name1": "Acme Corp", "name2": "Acme Corporation",
        "addr1": "123 Main St", "addr2": "123 Main Street",
        "country1": "US", "country2": "US",
        "postal1": "10001", "postal2": "10001",
    }
    print("Note: run train_model.py first so trained_model.joblib exists.")
    print("score_pair result:", score_pair(example_row))
