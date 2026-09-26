"""
Train/validation split done BY id1 (Source 1 entity), not by row/pair.
Splitting by pair would leak the same entity into both train and val
(since one entity can appear in multiple candidate pairs), inflating
validation scores.

Also handles class imbalance: most candidate pairs are non-matches.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def split_by_entity(df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42):
    """
    Splits so that every id1 (Source 1 entity) appears in only train or only val.
    Expects a column 'id1'.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=df["id1"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # sanity check: no id1 should appear in both
    overlap = set(train_df["id1"]) & set(val_df["id1"])
    assert not overlap, f"Leakage detected: {len(overlap)} id1 values in both splits"

    return train_df, val_df


def compute_class_weights(df: pd.DataFrame, label_col: str = "label") -> dict:
    """
    Returns {0: weight, 1: weight} inversely proportional to class frequency,
    for use as sample weights (e.g. in LightGBM's sample_weight param) instead
    of naive subsampling, which throws away data.
    """
    counts = df[label_col].value_counts()
    total = len(df)
    weights = {cls: total / (2 * count) for cls, count in counts.items()}
    return weights


def add_sample_weights(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    df = df.copy()
    weights = compute_class_weights(df, label_col)
    df["sample_weight"] = df[label_col].map(weights)
    return df


def subsample_majority_class(df: pd.DataFrame, label_col: str = "label",
                              ratio: float = 3.0, random_state: int = 42) -> pd.DataFrame:
    """
    Alternative to weighting: cap non-matches at `ratio` x the number of
    matches. Use ONE of weighting or subsampling, not both -- weighting is
    generally preferred since it keeps all the data.
    """
    matches = df[df[label_col] == 1]
    non_matches = df[df[label_col] == 0]
    n_keep = min(len(non_matches), int(len(matches) * ratio))
    non_matches_sampled = non_matches.sample(n=n_keep, random_state=random_state)
    return pd.concat([matches, non_matches_sampled]).sample(frac=1, random_state=random_state).reset_index(drop=True)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
    from make_mock_candidates import build_mock_dataset

    df = build_mock_dataset(300)
    train_df, val_df = split_by_entity(df)
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    print("Class weights:", compute_class_weights(train_df))
