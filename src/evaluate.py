"""
evaluate.py — Offline validation harness (no leaderboard, so we make our own).

We cannot see the real relevance tiers, so we build a transparent *silver*
ground truth from a rubric, then:
  1. score the ranking with the organizers' composite
     (0.50 NDCG@10 + 0.30 NDCG@50 + 0.15 MAP + 0.05 P@10),
  2. report objective trap counts in the top-K (honeypots, consulting-only,
     keyword-stuffers, non-AI titles, out-of-band, out-of-India),
  3. run ABLATIONS that justify each design choice by toggling it off.

IMPORTANT: silver labels overlap with the ranker's own signals, so absolute
NDCG is optimistic. Trust the *objective* numbers (honeypot rate, trap counts)
and the *relative* ablation deltas, not the absolute composite.

Usage:
  python evaluate.py --candidates <jsonl> --artifacts <dir>
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re

import numpy as np
import pandas as pd

_AI_TITLE = re.compile(r"\b(?:ml|ai|machine learning|data scien|nlp|research|"
                       r"recommend|search|applied|retrieval|rank)", re.I)
_STRONG_TITLE = re.compile(r"recommend|rank|retriev|search|nlp|applied ml|"
                           r"machine learning|relevance", re.I)


def _open(p):
    return gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, "r", encoding="utf-8")


# ---------- silver ground truth ----------
def silver_tier(row) -> int:
    """Transparent 0-5 relevance tier from the JD rubric."""
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


# ---------- metrics ----------
def _dcg(tiers):
    return sum((2 ** t - 1) / np.log2(i + 2) for i, t in enumerate(tiers))


def ndcg_at_k(ranked_tiers, all_tiers, k):
    idcg = _dcg(sorted(all_tiers, reverse=True)[:k])
    return _dcg(ranked_tiers[:k]) / idcg if idcg > 0 else 0.0


def average_precision(ranked_tiers, rel_threshold=3):
    hits, ap = 0, 0.0
    total_rel = sum(1 for t in ranked_tiers if t >= rel_threshold)
    for i, t in enumerate(ranked_tiers, start=1):
        if t >= rel_threshold:
            hits += 1
            ap += hits / i
    return ap / min(total_rel, len(ranked_tiers)) if total_rel else 0.0


def precision_at_k(ranked_tiers, k, rel_threshold=3):
    return sum(1 for t in ranked_tiers[:k] if t >= rel_threshold) / k


def composite(ranked_tiers, all_tiers):
    n10 = ndcg_at_k(ranked_tiers, all_tiers, 10)
    n50 = ndcg_at_k(ranked_tiers, all_tiers, 50)
    ap = average_precision(ranked_tiers[:100])
    p10 = precision_at_k(ranked_tiers, 10)
    return (0.50 * n10 + 0.30 * n50 + 0.15 * ap + 0.05 * p10,
            dict(ndcg10=n10, ndcg50=n50, map=ap, p10=p10))


# ---------- scoring (mirrors rank.py; config-driven for ablations) ----------
def score_all(sem_norm, F, cfg):
    core = (cfg["w_sem"] * sem_norm + cfg["w_prod"] * F["product_score"]
            + cfg["w_band"] * F["band_fit"] + cfg["w_loc"] * F["location_fit"]
            + cfg["w_notice"] * F["notice_fit"])
    pen = (F["company_penalty"] + F["domain_penalty"] + F["country_penalty"] - 0.10 * np.minimum(F["stuffing_score"], 4.0)
           if cfg.get("penalties", True) else 0.0)
    s = np.clip(core + pen, 0.0, None)
    if cfg["behavioral"]:
        s = s * (0.5 + 0.5 * F["availability"])      # softened down-weight
    if cfg["gate"]:
        s = np.where(F["is_honeypot"].astype(bool), 0.0, s)
    return s


FULL = dict(w_sem=.40, w_prod=.10, w_band=.35, w_loc=.05, w_notice=.10,
            behavioral=True, gate=True, penalties=True)
ABLATIONS = {
    "FULL (tuned)":          FULL,
    "no honeypot gate":      {**FULL, "gate": False},
    "no behavioral mult.":   {**FULL, "behavioral": False},
    "no trap penalties":     {**FULL, "penalties": False},
    "semantic only":         dict(w_sem=1., w_prod=0, w_band=0, w_loc=0, w_notice=0,
                                  behavioral=False, gate=False, penalties=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    emb = np.load(os.path.join(args.artifacts, "embeddings.npy"))
    ids = np.load(os.path.join(args.artifacts, "ids.npy"), allow_pickle=True).astype(str)
    pz = np.load(os.path.join(args.artifacts, "profiles.npz"))
    feat = pd.read_parquet(os.path.join(args.artifacts, "features.parquet")).set_index("candidate_id").reindex(ids)

    # pull title / years / country for silver labels + trap sets
    meta = {}
    want = set(ids)
    for c in _open(args.candidates):
        c = c.strip()
        if not c:
            continue
        c = json.loads(c)
        if c["candidate_id"] in want:
            p = c["profile"]
            meta[c["candidate_id"]] = (p.get("current_title", ""), p.get("years_of_experience", 0),
                                       p.get("country", ""))
    feat["title"] = [meta.get(i, ("", 0, ""))[0] for i in ids]
    feat["years"] = [meta.get(i, ("", 0, ""))[1] for i in ids]
    feat["country"] = [meta.get(i, ("", 0, ""))[2] for i in ids]

    tiers = feat.apply(silver_tier, axis=1).to_numpy()
    all_tiers = tiers.tolist()
    print(f"loaded {len(ids)} candidates | silver tier counts: "
          f"{dict(pd.Series(tiers).value_counts().sort_index())}")

    sem = (emb @ pz["ideal"].T).max(1) - (emb @ pz["anti"].T).max(1)
    sem_norm = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
    F = {k: feat[k].to_numpy() for k in
         ["product_score", "band_fit", "location_fit", "notice_fit",
          "company_penalty", "domain_penalty", "country_penalty", "stuffing_score", "availability", "is_honeypot"]}

    import pickle
    import lightgbm as lgb
    
    # Try to load LightGBM model
    model_path = os.path.join(args.artifacts, "ranker_model.pkl")
    lgb_scores = None
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        feat["semantic_fit"] = sem_norm
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
        lgb_scores = model.predict(feat[feature_cols])
        lgb_scores[F["is_honeypot"].astype(bool)] = 0.0

    print(f"\n{'config':22} {'composite':>10} {'ndcg10':>8} {'ndcg50':>8} "
          f"{'map':>6} {'p@10':>6} {'honeypots@100':>14}")
    print("-" * 80)
    for name, cfg in ABLATIONS.items():
        s = score_all(sem_norm, F, cfg)
        order = np.lexsort((ids, -s))[:100]
        rt = tiers[order].tolist()
        comp, m = composite(rt, all_tiers)
        hp = int(F["is_honeypot"][order].sum())
        print(f"{name:22} {comp:>10.4f} {m['ndcg10']:>8.4f} {m['ndcg50']:>8.4f} "
              f"{m['map']:>6.3f} {m['p10']:>6.2f} {hp:>14}")

    if lgb_scores is not None:
        order = np.lexsort((ids, -lgb_scores))[:100]
        rt = tiers[order].tolist()
        comp, m = composite(rt, all_tiers)
        hp = int(F["is_honeypot"][order].sum())
        print(f"{'LightGBM Regressor':22} {comp:>10.4f} {m['ndcg10']:>8.4f} {m['ndcg50']:>8.4f} "
              f"{m['map']:>6.3f} {m['p10']:>6.2f} {hp:>14}")

    # objective trap breakdown for the FULL model's top-K
    s = score_all(sem_norm, F, FULL)
    order = np.lexsort((ids, -s))[:100]
    sub = feat.iloc[order]
    def traps(d):
        return dict(honeypot=int(d["is_honeypot"].sum()),
                    consulting_only=int(d["consulting_only"].sum()),
                    keyword_stuffer=int((d["stuffing_score"] >= 2).sum()),
                    non_ai_title=int((~d["title"].str.contains(_AI_TITLE, na=False)).sum()),
                    out_of_band=int(((d["years"] < 5) | (d["years"] > 9)).sum()),
                    out_of_india=int((~d["country"].isin(["India", "in", "IN", ""])).sum()))
    print("\nobjective trap counts (FULL model):")
    print(f"  top 10 : {traps(sub.head(10))}")
    print(f"  top 100: {traps(sub)}")
    
    if lgb_scores is not None:
        lgb_order = np.lexsort((ids, -lgb_scores))[:100]
        lgb_sub = feat.iloc[lgb_order]
        print("\nobjective trap counts (LightGBM model):")
        print(f"  top 10 : {traps(lgb_sub.head(10))}")
        print(f"  top 100: {traps(lgb_sub)}")

    print("\nNOTE: silver NDCG is optimistic (labels overlap ranker signals). "
          "Trust honeypot rate, trap counts, and ablation deltas.")


if __name__ == "__main__":
    main()