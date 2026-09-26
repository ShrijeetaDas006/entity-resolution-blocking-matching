# Person B — Feature & Modeling Lead

## What's in here

| File | Purpose |
|---|---|
| `src/features/make_mock_candidates.py` | Generates a fake `candidate_pairs.tsv` so you can develop before Person A's real output exists |
| `src/features/similarity_features.py` | Levenshtein, Jaro-Winkler, token-set Jaccard, TF-IDF cosine (name + address) |
| `src/features/exact_match_flags.py` | Country match, postal match, first-token match, phonetic (Soundex) match |
| `src/model/split_and_balance.py` | Train/val split **by id1 (Source 1 entity)** to avoid leakage; class-weighting for imbalance |
| `src/model/train_model.py` | Trains the LightGBM classifier (+ logistic regression baseline) |
| `src/model/threshold_tuning.py` | Sweeps thresholds to maximize **F0.5**, not F1 or 0.5 |
| `src/model/inference.py` | **The deliverable**: `score_pair(row) -> float` and `score_batch(df)` for Person C to plug into the pipeline |

## Setup

```bash
pip install -r requirements_person_b.txt --break-system-packages
```

## Run order (with mock data, before Person A's real file exists)

```bash
cd src/features
python make_mock_candidates.py      # writes outputs/candidate_pairs_mock.tsv

cd ../model
python train_model.py               # trains + saves trained_model.joblib
python threshold_tuning.py          # prints best threshold for F0.5
python inference.py                 # sanity-checks score_pair() on one row
```

## Switching to Person A's real data

Once `outputs/candidate_pairs.tsv` exists (pushed by Person A), change the
data-loading line in `train_model.py`'s `__main__` block from
`build_mock_dataset(...)` to:

```python
df = pd.read_csv("../../outputs/candidate_pairs.tsv", sep="\t")
```

No other code changes needed — every downstream function (features, split,
training, inference) was built against the same agreed column schema, so it
works on real data exactly like it worked on mock data.

## The contract Person C depends on

```python
from src.model.inference import score_pair, score_batch

score_pair(row_dict)      # -> float probability, single pair
score_batch(dataframe)    # -> pandas Series of probabilities, many pairs
```

Update `_THRESHOLD` in `inference.py` after running `threshold_tuning.py`,
so `predict_match()` reflects the F0.5-optimal cutoff.

## Column schema (confirm this matches what the whole team agreed on)

`id1, id2, name1, name2, addr1, addr2, country1, country2, postal1, postal2, label`
