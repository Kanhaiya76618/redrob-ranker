"""
embed.py — OFFLINE artifact builder. No time limit; this is NOT the scored step.

Turns the candidate pool into the arrays rank.py consumes:
  artifacts/embeddings.npy   (N, dim) float32, L2-normalized  -> cosine = dot
  artifacts/ids.npy          (N,) candidate_id, aligned to embeddings rows
  artifacts/profiles.npz     ideal (k,dim) + anti (m,dim) profile vectors
  artifacts/features.parquet candidate_id + deterministic features + integrity
  artifacts/meta.json        {model, dim, reference_date, n}

Runs identically on a PC or in Colab. Embeds with bge-small via fastembed
(quantized ONNX, CPU-friendly), so no GPU and no torch are required.

Usage:
  python src/embed.py --candidates C:/redrob-data/candidates.jsonl \
                      --artifacts  C:/redrob-data/artifacts
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date

import numpy as np

from parse import load_candidates, parse_date, candidate_text
from features import compute_features
from integrity import check_integrity
from jd_profiles import get_profiles

MODEL = "BAAI/bge-small-en-v1.5"


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True, help="path to candidates.jsonl[.gz]")
    ap.add_argument("--artifacts", default="artifacts", help="output dir")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--backend", choices=["fastembed", "st"], default="fastembed",
                    help="fastembed = quantized ONNX on CPU; "
                         "st = sentence-transformers on cuda/mps/cpu")
    args = ap.parse_args()
    os.makedirs(args.artifacts, exist_ok=True)

    # 1) Load candidates and derive a reference "today" = latest activity date.
    cands = list(load_candidates(args.candidates))
    print(f"loaded {len(cands)} candidates")
    ref = date(2000, 1, 1)
    for c in cands:
        d = parse_date((c.get("redrob_signals", {}) or {}).get("last_active_date"))
        if d and d > ref:
            ref = d
    print("reference date (max last_active):", ref)

    # 2) Deterministic features + integrity + the evidence-weighted text doc.
    ids, rows, texts = [], [], []
    for c in cands:
        cid = c["candidate_id"]
        f = compute_features(c, ref)
        ig = check_integrity(c, ref)
        ids.append(cid)
        rows.append({"candidate_id": cid, **f,
                     "impossibility_score": ig["impossibility_score"],
                     "is_honeypot": ig["is_honeypot"],
                     "stuffing_score": ig["stuffing_score"]})
        texts.append(candidate_text(c))
    print("features + integrity computed")

    # 3) Embed candidates (the slow part; minutes on CPU). Two interchangeable
    #    backends produce the same L2-normalized vectors:
    #      fastembed -> quantized ONNX on CPU, no torch needed
    #      st        -> sentence-transformers on cuda -> mps -> cpu
    prof = get_profiles()
    if args.backend == "fastembed":
        # Stream one vector at a time into a preallocated array with a live
        # progress bar, so memory stays flat and you can watch progress + ETA.
        from fastembed import TextEmbedding
        from tqdm import tqdm
        model = TextEmbedding(model_name=MODEL)
        print("embedding candidates (fastembed / CPU ONNX)...")
        gen = model.embed(texts, batch_size=args.batch)
        first = next(gen)
        dim = len(first)
        embs = np.zeros((len(texts), dim), dtype=np.float32)
        embs[0] = first
        for i, v in enumerate(tqdm(gen, total=len(texts) - 1, initial=1,
                                   desc="embedding", unit="cand"), start=1):
            embs[i] = v
        embs = l2_normalize(embs)
        ideal = l2_normalize(np.asarray(list(model.embed(prof["ideal"])), dtype=np.float32))
        anti = l2_normalize(np.asarray(list(model.embed(prof["anti"])), dtype=np.float32))
    else:  # st: sentence-transformers, GPU when available
        import torch
        from sentence_transformers import SentenceTransformer
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available()
                  else "cpu")
        print(f"embedding candidates (sentence-transformers on {device})...")
        model = SentenceTransformer(MODEL, device=device)
        embs = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
        ideal = model.encode(prof["ideal"], normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
        anti = model.encode(prof["anti"], normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)
    print("candidate embeddings:", embs.shape)
    print(f"profile vectors: ideal {ideal.shape}, anti {anti.shape}")

    # 5) Persist artifacts.
    np.save(os.path.join(args.artifacts, "embeddings.npy"), embs)
    np.save(os.path.join(args.artifacts, "ids.npy"), np.asarray(ids))
    np.savez(os.path.join(args.artifacts, "profiles.npz"), ideal=ideal, anti=anti)

    import pandas as pd
    pd.DataFrame(rows).to_parquet(
        os.path.join(args.artifacts, "features.parquet"), index=False)

    with open(os.path.join(args.artifacts, "meta.json"), "w") as fh:
        json.dump({"model": MODEL, "dim": int(embs.shape[1]),
                   "reference_date": ref.isoformat(), "n": len(ids)}, fh, indent=2)

    print(f"\nartifacts written to {args.artifacts}/")


if __name__ == "__main__":
    main()