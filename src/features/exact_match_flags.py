"""
Exact-match boolean flags: country, postal code, first token, phonetic code.

Phonetic matching uses Soundex (via jellyfish) -- install:
    pip install jellyfish --break-system-packages
"""

import pandas as pd
import jellyfish


def first_token(s: str) -> str:
    if not s:
        return ""
    return str(s).strip().lower().split()[0] if str(s).strip() else ""


def soundex_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    try:
        return jellyfish.soundex(str(a)) == jellyfish.soundex(str(b))
    except Exception:
        return False


def add_exact_match_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expects columns: name1, name2, country1, country2, postal1, postal2
    Adds: country_match, postal_match, first_token_match, phonetic_match
    """
    df = df.copy()

    df["country_match"] = (
        df["country1"].astype(str).str.upper() == df["country2"].astype(str).str.upper()
    ).astype(int)

    df["postal_match"] = (
        df["postal1"].astype(str).str.strip() == df["postal2"].astype(str).str.strip()
    ).astype(int)

    df["first_token_match"] = df.apply(
        lambda r: int(first_token(r["name1"]) == first_token(r["name2"]) and first_token(r["name1"]) != ""),
        axis=1
    )

    df["phonetic_match"] = df.apply(
        lambda r: int(soundex_match(first_token(r["name1"]), first_token(r["name2"]))),
        axis=1
    )

    return df


if __name__ == "__main__":
    from make_mock_candidates import build_mock_dataset
    df = build_mock_dataset(20)
    df = add_exact_match_flags(df)
    print(df[["name1", "name2", "country_match", "postal_match",
              "first_token_match", "phonetic_match"]].head())
