import re
import pandas as pd

def clean_text(text: str) -> str:
    """Standardize input string by lowercasing and removing special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text)

def normalize_dataset(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Apply normalization across specified DataFrame columns."""
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_text)
    return df_clean
