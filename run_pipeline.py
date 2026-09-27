import os
import pandas as pd
from src.normalize import normalize_dataset
from src.blocking import create_blocks

# 1. Load Data
DATA_DIR = "data/train"
s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t")
s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep="\t")
gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t")

print(f"Loaded {len(s1)} records from S1 and {len(s2)} records from S2.")

# 2. Normalize Data
text_cols = [col for col in ["name", "address", "city"] if col in s1.columns]
s1_clean = normalize_dataset(s1, text_cols)
s2_clean = normalize_dataset(s2, text_cols)
print("Normalization complete.")

# 3. Create Candidate Pairs (Blocking)
combined_df = pd.concat([s1_clean, s2_clean], ignore_index=True)
blocking_cols = [col for col in ["country", "city"] if col in combined_df.columns]
if not blocking_cols:
    blocking_cols = [combined_df.columns[1]]

# Store generated pairs into candidate_pairs variable
candidate_pairs = create_blocks(combined_df, key_fields=blocking_cols, id_col="entity_id")
print(f"Generated {len(candidate_pairs)} candidate pairs.")

# 4. Save Outputs to outputs/ folder
os.makedirs("outputs", exist_ok=True)

candidate_list = list(candidate_pairs)
cand_df = pd.DataFrame(candidate_list, columns=["entity_id_1", "entity_id_2"])
cand_df.to_csv("outputs/candidate_pairs.tsv", sep="\t", index=False)

# Create output placeholder for matching stage
cand_df["match_score"] = 1.0
cand_df.to_csv("outputs/matching_results.tsv", sep="\t", index=False)

print("Saved pipeline outputs to 'outputs/' directory.")