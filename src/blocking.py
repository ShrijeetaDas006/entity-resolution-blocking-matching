import pandas as pd

def generate_blocking_key(row, fields):
    keys = []
    for field in fields:
        val = str(row.get(field, "")).strip()
        if val:
            keys.append(val[:3])
    return "_".join(keys)

def create_blocks(df: pd.DataFrame, key_fields: list, id_col: str):
    df['blocking_key'] = df.apply(lambda r: generate_blocking_key(r, key_fields), axis=1)
    
    blocks = df.groupby('blocking_key')[id_col].apply(list).to_dict()
    
    candidate_pairs = set()
    for b_key, ids in blocks.items():
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    pair = tuple(sorted([ids[i], ids[j]]))
                    candidate_pairs.add(pair)
                    
    return candidate_pairs

if __name__ == "__main__":
    print("Blocking script ready.")