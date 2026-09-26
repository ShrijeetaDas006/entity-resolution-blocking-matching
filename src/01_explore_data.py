import pandas as pd

DATA_DIR = "data/train"

s1 = pd.read_csv(f"{DATA_DIR}/train_source1.tsv", sep="\t")
s2 = pd.read_csv(f"{DATA_DIR}/train_source2.tsv", sep="\t")
s3 = pd.read_csv(f"{DATA_DIR}/train_source3.tsv", sep="\t")
gt = pd.read_csv(f"{DATA_DIR}/train_ground_truth.tsv", sep="\t")

print("ROW COUNTS")
print(f"S1: {len(s1)}, S2: {len(s2)}, S3: {len(s3)}, GT: {len(gt)}")

print("\nNULL RATES (%)")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"\n{name}:\n{(df.isnull().mean()*100).round(2)}")

print("\nDUPLICATE entity_id CHECK")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"{name}: {df['entity_id'].duplicated().sum()} dupes")

def count_matches(x):
    if pd.isna(x) or x.strip() == "":
        return 0
    return len(x.split(","))

gt["num_matches"] = gt["matched_entity_ids"].apply(count_matches)
print("\nMATCH DISTRIBUTION")
print(gt["num_matches"].value_counts().sort_index())
print(f"Singletons: {(gt['num_matches']==0).mean()*100:.1f}%")

print("\nCOUNTRY DISTRIBUTION")
for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"\n{name}:\n{df['country'].value_counts()}")

# Also load and compare test set countries
test_s1 = pd.read_csv("data/test/test_source1.tsv", sep="\t")
print(f"\nTest S1 countries:\n{test_s1['country'].value_counts()}")