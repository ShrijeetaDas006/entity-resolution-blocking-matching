"""
Similarity feature functions for scoring a candidate pair's name and address.

Install first:
    pip install rapidfuzz scikit-learn pandas --break-system-packages
"""

from rapidfuzz.distance import Levenshtein, JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def levenshtein_ratio(a: str, b: str) -> float:
    """0-1 similarity; 1 = identical strings."""
    if not a or not b:
        return 0.0
    return Levenshtein.normalized_similarity(str(a), str(b))


def jaro_winkler_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return JaroWinkler.normalized_similarity(str(a), str(b))


def token_set_jaccard(a: str, b: str) -> float:
    """Set-based overlap of whitespace tokens -- robust to word-order swaps."""
    if not a or not b:
        return 0.0
    set_a = set(str(a).lower().split())
    set_b = set(str(b).lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def tfidf_cosine_batch(series_a: pd.Series, series_b: pd.Series) -> np.ndarray:
    """
    TF-IDF cosine similarity, computed pairwise (row i of A vs row i of B).
    Fit on the combined vocabulary of both columns so both sides share the
    same feature space.
    """
    combined = pd.concat([series_a.fillna(""), series_b.fillna("")], ignore_index=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    tfidf = vectorizer.fit_transform(combined)

    n = len(series_a)
    tfidf_a = tfidf[:n]
    tfidf_b = tfidf[n:]

    # row-wise cosine similarity (not full pairwise matrix)
    sims = np.array([
        cosine_similarity(tfidf_a[i], tfidf_b[i])[0, 0]
        for i in range(n)
    ])
    return sims


def add_similarity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: name1, name2, addr1, addr2
    Adds: name_lev, name_jw, name_jaccard, name_tfidf_cos,
          addr_lev, addr_jw, addr_jaccard, addr_tfidf_cos
    """
    df = df.copy()

    df["name_lev"] = df.apply(lambda r: levenshtein_ratio(r["name1"], r["name2"]), axis=1)
    df["name_jw"] = df.apply(lambda r: jaro_winkler_ratio(r["name1"], r["name2"]), axis=1)
    df["name_jaccard"] = df.apply(lambda r: token_set_jaccard(r["name1"], r["name2"]), axis=1)
    df["name_tfidf_cos"] = tfidf_cosine_batch(df["name1"], df["name2"])

    df["addr_lev"] = df.apply(lambda r: levenshtein_ratio(r["addr1"], r["addr2"]), axis=1)
    df["addr_jw"] = df.apply(lambda r: jaro_winkler_ratio(r["addr1"], r["addr2"]), axis=1)
    df["addr_jaccard"] = df.apply(lambda r: token_set_jaccard(r["addr1"], r["addr2"]), axis=1)
    df["addr_tfidf_cos"] = tfidf_cosine_batch(df["addr1"], df["addr2"])

    return df


if __name__ == "__main__":
    from make_mock_candidates import build_mock_dataset
    df = build_mock_dataset(50)
    df = add_similarity_features(df)
    print(df[["name1", "name2", "name_lev", "name_jw", "name_jaccard", "name_tfidf_cos"]].head())
