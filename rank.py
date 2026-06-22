"""
rank.py — The scored step. Loads precomputed artifacts + candidates.jsonl and
writes the top-100 ranked CSV. Runs in seconds: no model, no network.

Tier 0  integrity gate     impossible profiles (honeypots) -> score 0
Tier 1  semantic recall    cosine(candidate, IDEAL) - cosine(candidate, ANTI)
Tier 2  rerank             core_fit + penalties, times behavioral availability

Output CSV columns: candidate_id, rank, score, reasoning   (exactly 100 rows)

Usage:
  python rank.py --candidates D:/redrob-data/.../candidates.jsonl \
                 --artifacts  D:/redrob-data/artifacts \
                 --out        submission.csv
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import pickle
import lightgbm as lgb
from reasoning import make_reasoning


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, "r", encoding="utf-8")


def fetch_records(path: str, want_ids: set) -> dict:
    """Stream the JSONL and keep only the records we need (the top 100)."""
    out = {}
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in want_ids:
                out[c["candidate_id"]] = c
                if len(out) == len(want_ids):
                    break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    # ---- load artifacts ----
    emb = np.load(os.path.join(args.artifacts, "embeddings.npy"))
    ids = np.load(os.path.join(args.artifacts, "ids.npy"), allow_pickle=True).astype(str)
    pz = np.load(os.path.join(args.artifacts, "profiles.npz"))
    ideal, anti = pz["ideal"], pz["anti"]
    feat = pd.read_parquet(os.path.join(args.artifacts, "features.parquet")).set_index("candidate_id")
    feat = feat.reindex(ids)  # align feature rows to embedding row order
    print(f"loaded {len(ids)} candidates, dim={emb.shape[1]}")

    # ---- Tier 1: contrastive semantic fit (vectors are L2-normalized) ----
    sem = (emb @ ideal.T).max(1) - (emb @ anti.T).max(1)
    sem_norm = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)

    feat["semantic_fit"] = sem_norm
    honey = feat["is_honeypot"].to_numpy().astype(bool)

    # ---- Tier 2: LightGBM Regressor prediction ----
    feature_cols = [
        "semantic_fit",
        "band_fit",
        "product_score",
        "location_fit",
        "notice_fit",
        "availability",
        "company_penalty",
        "domain_penalty",
        "country_penalty",
        "stuffing_score",
        "impossibility_score"
    ]
    X = feat[feature_cols]

    model_path = os.path.join(args.artifacts, "ranker_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    score = model.predict(X)
    score[honey] = 0.0                       # Tier 0 gate

    # ---- rank: sort by score desc, deterministic id tiebreak ----
    order = np.lexsort((ids, -score))[:args.topk]
    top_ids = ids[order]
    top_scores = np.round(score[order], 5)

    # ---- reasoning for the top K (pull their raw records) ----
    records = fetch_records(args.candidates, set(top_ids))
    rows = []
    for r, (cid, sc) in enumerate(zip(top_ids, top_scores), start=1):
        fr = feat.loc[cid].to_dict()
        reasoning = make_reasoning(records.get(cid, {}), fr)
        rows.append({"candidate_id": cid, "rank": r, "score": sc, "reasoning": reasoning})

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    hp = int(honey[order].sum())
    print(f"wrote {len(out)} rows to {args.out} | honeypots in top {args.topk}: {hp}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()