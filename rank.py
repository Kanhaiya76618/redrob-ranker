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
import re

import numpy as np
import pandas as pd

# How the blended score is composed (documented + tunable in one place).
W_SEM, W_PROD, W_BAND, W_LOC, W_NOTICE = 0.40, 0.18, 0.24, 0.10, 0.08

_RETR_TERMS = [
    ("learning-to-rank", "learning to rank"), ("ranking systems", "ranking"),
    ("retrieval", "retriev"), ("search relevance", "search"),
    ("recommendation systems", "recommend"), ("embeddings", "embedding"),
    ("semantic matching", "semantic"), ("vector search", "vector"),
]


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


def evidence_phrase(c: dict):
    text = (c.get("profile", {}).get("summary", "") + " " +
            " ".join(h.get("description", "") for h in c.get("career_history", []) or [])).lower()
    for label, key in _RETR_TERMS:
        if key in text:
            return label
    return None


def make_reasoning(c: dict, fr: dict) -> str:
    """Grounded, varied, honest 1-2 sentence justification from real fields."""
    p = c.get("profile", {})
    yrs = p.get("years_of_experience")
    title = p.get("current_title", "professional")
    comp = p.get("current_company") or (c.get("career_history") or [{}])[0].get("company")
    lead = f"{yrs:.0f}y {title}" if isinstance(yrs, (int, float)) else title
    if comp:
        lead += f" at {comp}"

    ev = evidence_phrase(c)
    if ev and fr["product_score"] > 0:
        s = f"{lead}; career shows hands-on {ev} work at product companies"
    elif ev:
        s = f"{lead} with demonstrated {ev} experience"
    elif fr["product_score"] > 0:
        s = f"{lead} with product-company background"
    else:
        s = lead

    extra = []
    if fr["location_fit"] >= 0.85:
        extra.append("well-located for Pune/Noida")
    if fr["availability"] >= 0.7:
        extra.append("active and responsive on-platform")
    if extra:
        s += "; " + ", ".join(extra)
    s += "."

    concerns = []
    if fr["notice_fit"] <= 0.5:
        concerns.append("long notice period")
    if fr.get("consulting_only"):
        concerns.append("services-only background")
    if fr["availability"] < 0.4:
        concerns.append("limited recent activity")
    if fr["domain_penalty"] < 0:
        concerns.append("thin NLP/IR signal")
    if concerns:
        s += " Concern: " + ", ".join(concerns) + "."
    return s


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

    band = feat["band_fit"].to_numpy()
    prod = feat["product_score"].to_numpy()
    loc = feat["location_fit"].to_numpy()
    notice = feat["notice_fit"].to_numpy()
    avail = feat["availability"].to_numpy()
    comp_pen = feat["company_penalty"].to_numpy()
    dom_pen = feat["domain_penalty"].to_numpy()
    stuff = feat["stuffing_score"].to_numpy()
    honey = feat["is_honeypot"].to_numpy().astype(bool)

    # ---- Tier 2: blended score, behavioral multiplier, honeypot gate ----
    core = (W_SEM * sem_norm + W_PROD * prod + W_BAND * band +
            W_LOC * loc + W_NOTICE * notice)
    penalties = comp_pen + dom_pen - 0.10 * np.minimum(stuff, 4.0)
    # behavioral down-weight, softened: a dark candidate loses up to ~45%, not
    # 90% — honours the JD's availability signal without burying strong people.
    score = np.clip(core + penalties, 0.0, None) * (0.5 + 0.5 * avail)
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