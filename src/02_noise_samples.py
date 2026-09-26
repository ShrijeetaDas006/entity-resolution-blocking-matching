import pandas as pd

s1 = pd.read_csv("data/train/train_source1.tsv", sep="\t")
s2 = pd.read_csv("data/train/train_source2.tsv", sep="\t")
s3 = pd.read_csv("data/train/train_source3.tsv", sep="\t")
gt = pd.read_csv("data/train/train_ground_truth.tsv", sep="\t")

gt["num_matches"] = gt["matched_entity_ids"].apply(
    lambda x: 0 if pd.isna(x) or x.strip()=="" else len(x.split(","))
)
samples = gt[gt["num_matches"] > 0].sample(40, random_state=42)

rows = []
for _, row in samples.iterrows():
    s1_id = row["source1_entity_id"]
    s1_row = s1[s1["entity_id"] == s1_id].iloc[0]
    for mid in row["matched_entity_ids"].split(","):
        src = s2 if mid.startswith("S2-") else s3
        match = src[src["entity_id"] == mid]
        if len(match) == 0:
            continue
        match = match.iloc[0]
        rows.append({
            "s1_id": s1_id, "s1_name": s1_row["business_name"],
            "s1_address": s1_row["business_address"],
            "match_id": mid, "match_name": match["business_name"],
            "match_address": match["business_address"],
        })

pd.DataFrame(rows).to_csv("outputs/noise_samples.csv", index=False)
print(f"Saved {len(rows)} pairs — open outputs/noise_samples.csv and read manually")