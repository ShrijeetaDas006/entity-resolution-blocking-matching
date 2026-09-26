"""
Trains the matching classifier on the labeled candidate pairs.

Install first:
    pip install lightgbm scikit-learn joblib --break-system-packages

Model choice: LightGBM (MIT license, well under 8B params -- a gradient
boosted tree model is a few hundred KB to a few MB, nowhere near a
parameter-count concern). Logistic regression is included as a simpler
fallback/baseline.
"""

import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

FEATURE_COLUMNS = [
    "name_lev", "name_jw", "name_jaccard", "name_tfidf_cos",
    "addr_lev", "addr_jw", "addr_jaccard", "addr_tfidf_cos",
    "country_match", "postal_match", "first_token_match", "phonetic_match",
]


def train_lightgbm(train_df: pd.DataFrame, label_col: str = "label",
                    weight_col: str = "sample_weight") -> lgb.LGBMClassifier:
    X = train_df[FEATURE_COLUMNS]
    y = train_df[label_col]
    sample_weight = train_df[weight_col] if weight_col in train_df.columns else None

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
    )
    model.fit(X, y, sample_weight=sample_weight)
    return model


def train_logistic_regression(train_df: pd.DataFrame, label_col: str = "label") -> LogisticRegression:
    """Simple, interpretable baseline to sanity-check the LightGBM model against."""
    X = train_df[FEATURE_COLUMNS]
    y = train_df[label_col]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X, y)
    return model


def save_model(model, path: str = "src/model/trained_model.joblib"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str = "src/model/trained_model.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "features"))
    from make_mock_candidates import build_mock_dataset
    from similarity_features import add_similarity_features
    from exact_match_flags import add_exact_match_flags
    from split_and_balance import split_by_entity, add_sample_weights

    df = build_mock_dataset(300)
    df = add_similarity_features(df)
    df = add_exact_match_flags(df)
    train_df, val_df = split_by_entity(df)
    train_df = add_sample_weights(train_df)

    model = train_lightgbm(train_df)
    save_model(model)
    print("Training complete.")
