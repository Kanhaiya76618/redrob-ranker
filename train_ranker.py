"""
train_ranker.py — Offline script to train the LightGBM ranker model.
Generates heuristic target labels based on the JD rubric and trains an LGBMRegressor.
Saves the trained model to artifacts/ranker_model.pkl.
"""
from __future__ import annotations

import os
import sys
import pickle
import re
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parse import load_candidates

CANDIDATES = "/Users/kanhaiya_mehta/redrob-data/candidates.jsonl"
ARTIFACTS = "/Users/kanhaiya_mehta/redrob-data/artifacts"

_AI_TITLE = re.compile(r"\b(?:ml|ai|machine learning|data scien|nlp|research|recommend|search|applied|retrieval|rank)", re.I)
_STRONG_TITLE = re.compile(r"recommend|rank|retriev|search|nlp|applied ml|machine learning|relevance", re.I)


def silver_tier(row) -> int:
    """Heuristic relevance tier (0 to 5) from the JD rubric."""
    if row["is_honeypot"]:
        return 0                                   # impossible profile
    if row["consulting_only"] or row["stuffing_score"] >= 2 or row["domain_penalty"] < 0 or row["country_penalty"] < 0:
        return 1                                   # explicit do-not-want
    ai = bool(_AI_TITLE.search(str(row["title"])))
    strong = bool(_STRONG_TITLE.search(str(row["title"])))
    prod = row["product_score"] > 0
    in_band = 5 <= row["years"] <= 9
    ideal_band = 6 <= row["years"] <= 8
    active = row["availability"] >= 0.6
    if strong and prod and ideal_band and active:
        return 5
    if ai and prod and in_band:
        return 4
    if ai and (prod or in_band):
        return 3
    if ai:
        return 2
    return 1


def main():
    print("Loading precomputed artifacts...")
    emb = np.load(os.path.join(ARTIFACTS, "embeddings.npy"))
    ids = np.load(os.path.join(ARTIFACTS, "ids.npy"), allow_pickle=True).astype(str)
    pz = np.load(os.path.join(ARTIFACTS, "profiles.npz"))
    ideal, anti = pz["ideal"], pz["anti"]
    feat = pd.read_parquet(os.path.join(ARTIFACTS, "features.parquet")).set_index("candidate_id").reindex(ids)

    print("Extracting metadata for target labeling...")
    meta = {}
    with open(CANDIDATES, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["candidate_id"] in ids:
                p = c["profile"]
                meta[c["candidate_id"]] = (
                    p.get("current_title", ""),
                    p.get("years_of_experience", 0),
                    p.get("country", "")
                )
    
    feat["title"] = [meta.get(i, ("", 0, ""))[0] for i in ids]
    feat["years"] = [meta.get(i, ("", 0, ""))[1] for i in ids]
    feat["country"] = [meta.get(i, ("", 0, ""))[2] for i in ids]

    # Compute semantic fit feature
    print("Computing semantic fit...")
    sem = (emb @ ideal.T).max(1) - (emb @ anti.T).max(1)
    sem_norm = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
    feat["semantic_fit"] = sem_norm

    # Create target labels
    print("Generating heuristic target labels...")
    tiers = feat.apply(silver_tier, axis=1).to_numpy()
    
    # Map tiers to targets [0.0, 1.0]
    tier_map = {0: 0.0, 1: 0.1, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}
    targets = np.array([tier_map[t] for t in tiers])

    # Select features for LightGBM
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
    y = targets

    print(f"Training LightGBM Regressor on {X.shape[0]} candidates with {X.shape[1]} features...")
    # Train regressor
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    model.fit(X, y)

    # Save model
    model_path = os.path.join(ARTIFACTS, "ranker_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Trained LightGBM model saved to {model_path}")

    # Inspect feature importance
    importance = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for col, imp in feat_imp:
        print(f"  {col:20}: {imp}")


if __name__ == "__main__":
    main()
