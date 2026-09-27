import os
import pandas as pd

def validate_submission_package():
    print("--- Running Submission Output Validation Checks ---")
    os.makedirs("outputs", exist_ok=True)
    
    for filename in ["candidate_pairs.tsv", "matching_results.tsv"]:
        path = os.path.join("outputs", filename)
        if os.path.exists(path):
            df = pd.read_csv(path, sep="\t")
            print(f"[PASS] outputs/{filename} exists ({len(df)} rows).")
        else:
            print(f"[FAIL] outputs/{filename} is missing!")

if __name__ == "__main__":
    validate_submission_package()