import re

LEGAL_SUFFIXES = {
    "corp": "corporation", "corporation": "corporation",
    "co": "company", "company": "company",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "pvt": "private", "private": "private",
    "llc": "llc", "llp": "llp",
}

ADDRESS_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard",
    "dr": "drive", "ln": "lane", "apt": "apartment", "no": "number",
}

def normalize_name(name):
    if not isinstance(name, str):
        return "", ""
    name = name.lower().strip()
    name = name.replace("&", " and ")
    name = re.sub(r"[^\w\s]", " ", name)
    tokens = name.split()
    suffix = ""
    core_tokens = []
    for t in tokens:
        if t in LEGAL_SUFFIXES:
            suffix = LEGAL_SUFFIXES[t]
        else:
            core_tokens.append(t)
    core = " ".join(sorted(core_tokens))
    return core, suffix

def normalize_address(address):
    if not isinstance(address, str):
        return {"landmark": "", "tokens": []}
    addr = address.lower().strip()
    landmark = ""
    landmark_match = re.search(r"near\s+([^,]+)", addr)
    if landmark_match:
        landmark = landmark_match.group(1).strip()
        addr = addr.replace(landmark_match.group(0), "")
    addr = re.sub(r"[^\w\s]", " ", addr)
    tokens = [ADDRESS_ABBR.get(t, t) for t in addr.split()]
    return {"landmark": landmark, "tokens": sorted(tokens)}