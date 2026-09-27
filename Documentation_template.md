# Entity Resolution Pipeline Technical Documentation

## 1. System Architecture
The end-to-end entity resolution workflow consists of five sequential modules:
1. **Data Ingestion:** Loads record datasets (`train_source1.tsv`, `train_source2.tsv`) and ground truth mappings.
2. **Data Normalization (`src/normalize.py`):** Cleans text attributes by lowercasing, stripping special characters, and standardizing whitespace.
3. **Blocking / Candidate Pair Generation (`src/blocking.py`):** Groups records by attribute keys to filter out obvious non-matches and output candidate pairs.
4. **Matching & Scoring (`src/model/`):** Evaluates candidate pairs using pairwise similarity scoring or classification models.
5. **Submission Export (`outputs/`):** Formats and writes finalized output files according to evaluation requirements.

## 2. Evaluation Framework
* **Primary Metric:** Macro-averaged $F_{0.5}$ score. $F_{0.5}$ weights precision higher than recall ($\beta = 0.5$) to ensure candidate pairs and final matches maintain low false-positive rates.
* **Secondary Metrics:** Precision, Recall, Pair Completeness, and Reduction Ratio (measuring candidate search space reduction).

## 3. Data & File Contracts
* Candidate output location: `outputs/candidate_pairs.tsv`
  * Required columns: `entity_id_1`, `entity_id_2`
* Matching output location: `outputs/matching_results.tsv`
  * Required columns: `entity_id_1`, `entity_id_2`, `match_score`