"""
Generates a small fake candidate_pairs.tsv so Person B can develop and test
feature code before Person A's real blocking output is ready.

Once A pushes the real candidate_pairs.tsv into /outputs, just point the
pipeline at that file instead -- no code changes needed downstream, since
everything is built against this same column schema.

AGREED SCHEMA (confirm this matches what the team agreed on):
    id1, id2        -> record IDs being compared
    name1, name2    -> entity names
    addr1, addr2    -> entity addresses
    country1, country2
    postal1, postal2
    label           -> 1 = true match, 0 = non-match (only present in train/val)
"""

import pandas as pd
import random

random.seed(42)

MOCK_ROWS = [
    # (name1, name2, addr1, addr2, country1, country2, postal1, postal2, label)
    ("Acme Corp", "Acme Corporation", "123 Main St", "123 Main Street",
     "US", "US", "10001", "10001", 1),
    ("Global Tech Ltd", "Global Technologies Limited", "45 Oak Ave", "45 Oak Avenue",
     "GB", "GB", "SW1A 1AA", "SW1A 1AA", 1),
    ("Nordic Fisheries AB", "Nordic Fisheries", "Hamngatan 12", "Hamngatan 12",
     "SE", "SE", "11147", "11147", 1),
    ("Sunrise Bakery", "Sunset Bakery", "9 Elm St", "22 Pine Rd",
     "US", "US", "60614", "60615", 0),
    ("Delta Airlines", "Delta Air Lines Inc", "PO Box 20706", "Hartsfield Airport",
     "US", "US", "30320", "30320", 1),
    ("Cafe Paris", "Cafe de Paris", "10 Rue de Rivoli", "10 Rue Rivoli",
     "FR", "FR", "75001", "75001", 1),
    ("Tokyo Imports", "Osaka Exports", "1-2 Shibuya", "3-4 Namba",
     "JP", "JP", "150-0002", "556-0011", 0),
    ("Berlin Motors GmbH", "Berlin Motors", "Alexanderplatz 1", "Alexanderplatz 1",
     "DE", "DE", "10178", "10178", 1),
    ("Maple Leaf Foods", "Maple Leaf Food Co", "500 King St W", "500 King Street West",
     "CA", "CA", "M5V 1L9", "M5V 1L9", 1),
    ("Riverdale Clinic", "Riverside Clinic", "12 River Rd", "88 Lake Rd",
     "US", "US", "94110", "94111", 0),
]

def build_mock_dataset(n_pairs: int = 200) -> pd.DataFrame:
    """Expands the hand-written examples above into a larger mock set by
    repeating/perturbing them, so downstream code (train/val split, class
    imbalance handling) has enough rows to run against realistically."""
    rows = []
    for i in range(n_pairs):
        base = random.choice(MOCK_ROWS)
        name1, name2, addr1, addr2, c1, c2, p1, p2, label = base
        rows.append({
            "id1": f"S1_{i:04d}",
            "id2": f"S2_{i:04d}",
            "name1": name1,
            "name2": name2,
            "addr1": addr1,
            "addr2": addr2,
            "country1": c1,
            "country2": c2,
            "postal1": p1,
            "postal2": p2,
            "label": label,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_mock_dataset()
    out_path = "outputs/candidate_pairs_mock.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Wrote {len(df)} mock candidate pairs to {out_path}")
