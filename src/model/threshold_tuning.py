"""
Tunes the decision threshold specifically to maximize F0.5 (precision-weighted),
NOT F1 and NOT a naive 0.5 cutoff.

F-beta with beta=0.5 weights precision higher than recall -- appropriate when
false-positive matches (wrongly merging two different entities) are costlier
than false negatives (missing a real match).
"""

import numpy as np
from sklearn.metrics import fbeta_score, precision_score, recall_score


def find_best_threshold(y_true, y_proba, beta: float = 0.5,
                         thresholds=None) -> dict:
    """
    Sweeps thresholds and returns the one maximizing F-beta.
    Returns dict with best_threshold, best_f_beta, precision, recall at that point.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.00, 0.01)

    best = {"threshold": 0.5, "f_beta": -1, "precision": 0, "recall": 0}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        # skip degenerate thresholds where nothing is predicted positive
        if y_pred.sum() == 0:
            continue
        f = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        if f > best["f_beta"]:
            p = precision_score(y_true, y_pred, zero_division=0)
            r = recall_score(y_true, y_pred, zero_division=0)
            best = {"threshold": round(float(t), 2), "f_beta": round(float(f), 4),
                    "precision": round(float(p), 4), "recall": round(float(r), 4)}

    return best


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
    from make_mock_candidates import build_mock_dataset
    from similarity_features import add_similarity_features
    from exact_match_flags import add_exact_match_flags
    from split_and_balance import split_by_entity, add_sample_weights
    from train_model import train_lightgbm, FEATURE_COLUMNS

    df = build_mock_dataset(300)
    df = add_similarity_features(df)
    df = add_exact_match_flags(df)
    train_df, val_df = split_by_entity(df)
    train_df = add_sample_weights(train_df)

    model = train_lightgbm(train_df)
    val_proba = model.predict_proba(val_df[FEATURE_COLUMNS])[:, 1]

    result = find_best_threshold(val_df["label"].values, val_proba, beta=0.5)
    print("Best threshold result:", result)
