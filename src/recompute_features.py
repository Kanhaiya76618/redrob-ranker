"""
recompute_features.py — Recompute and overwrite features.parquet in artifacts.
Does not perform any slow embedding; finishes in seconds.
"""
from __future__ import annotations

import sys
import os
import pandas as pd
from datetime import date

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.parse import load_candidates, parse_date
from src.features import compute_features
from src.integrity import check_integrity

CANDIDATES = "/Users/kanhaiya_mehta/redrob-data/candidates.jsonl"
ARTIFACTS = "/Users/kanhaiya_mehta/redrob-data/artifacts"


def main():
    print("Loading candidates from:", CANDIDATES)
    cands = list(load_candidates(CANDIDATES))
    print(f"Loaded {len(cands)} candidates.")

    # 1) Derive reference date
    ref = date(2000, 1, 1)
    for c in cands:
        d = parse_date((c.get("redrob_signals", {}) or {}).get("last_active_date"))
        if d and d > ref:
            ref = d
    print("Reference date (max last_active):", ref)

    # 2) Compute features & integrity
    rows = []
    for c in cands:
        cid = c["candidate_id"]
        f = compute_features(c, ref)
        ig = check_integrity(c, ref)
        rows.append({
            "candidate_id": cid,
            **f,
            "impossibility_score": ig["impossibility_score"],
            "is_honeypot": ig["is_honeypot"],
            "stuffing_score": ig["stuffing_score"]
        })
    print("Computed features & integrity.")

    # 3) Overwrite features.parquet
    out_path = os.path.join(ARTIFACTS, "features.parquet")
    pd.DataFrame(rows).to_parquet(out_path, index=False)
    print("Overwrote features.parquet at:", out_path)


if __name__ == "__main__":
    main()
