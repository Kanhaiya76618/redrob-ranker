# Redrob Ranker — Intelligent Candidate Discovery & Ranking Engine

> An **evidence-verifying, fraud-resistant** candidate ranking engine.
> Built for **India Runs by Redrob AI × Hack2Skill** — the AI & Datathon Arena.

Old recruiting tools rank candidates by keyword matching. In 2026 that is broken: anyone can generate a resume stuffed with the exact keywords of a job ad, so gamed profiles outrank authentic talent who describe equivalent work in plain language. This engine does the opposite — it **ranks demonstrated evidence, not claimed keywords**, actively detects internally-impossible ("honeypot") profiles, and weighs whether a candidate is actually reachable and hireable.

---

## The approach in one line

Don't trust what a profile *claims* — judge what its career *demonstrates*, and refuse to rank a profile whose story is logically impossible.

This is implemented as a **three-tier funnel** that spends cheap compute widely and expensive compute narrowly, concentrating effort where the score lives (the top 10–50):

| Tier | Runs on | Job | Cost |
|------|---------|-----|------|
| **0 · Gate** | all 100,000 | Kill logically-impossible profiles (honeypots) before they contaminate anything | free — arithmetic |
| **1 · Recall** | all survivors | Coarse semantic match of *evidence* vs ideal-profiles; keep a generous top ~2,000 | cheap — matmul |
| **2 · Rerank** | the ~2,000 winners | Deep verification: cross-encoder, claim-vs-evidence, disqualifiers, behavioral multiplier → top 100 | heavy — precomputed offline |

We narrow aggressively on **fakes** (Tier 0) but gently on **plausible matches** (Tier 1): cutting a fake early is free, but cutting a real top-tier candidate early is irreversible.

---

## Architecture: offline vs runtime

All heavy computation happens **before** the timed window. The scored step is pure NumPy.

```
OFFLINE  (no time limit)            build_index.py
  ├─ Expand JD → ideal + anti profiles  (LLM, dev-time)
  ├─ Parse candidates → structured features
  ├─ Embed pool with bge-small (ONNX, CPU)   → embeddings.npy
  └─ Integrity scan (honeypot + stuffer)     → features.parquet, profile_vecs.npy

RUNTIME  (≤5 min · 16 GB · CPU · no network)   rank.py
  load artifacts → Tier 0 gate → Tier 1 recall → Tier 2 rerank
                 → top 100 + grounded reasoning → submission.csv
```

No model is loaded at runtime and no network is used — the timed step only loads arrays and computes, so it finishes in seconds and is trivially reproducible.

---

## Repository structure

```
redrob-ranker/
├─ README.md                 # this file
├─ LICENSE                   # MIT
├─ requirements.txt
├─ submission_metadata.yaml
├─ build_index.py            # OFFLINE: features + embeddings + profiles → artifacts/
├─ rank.py                   # RUNTIME: artifacts + candidates.jsonl → submission.csv
├─ app.py                    # optional Streamlit/Gradio demo (the "showroom")
├─ src/
│  ├─ parse.py               # data loader + evidence-weighted candidate_text()
│  ├─ integrity.py           # honeypot gate + keyword-stuffer detection
│  ├─ features.py            # band fit, product-vs-services, behavioral multiplier
│  ├─ jd_profiles.py         # ideal / anti-profile expansion + JD rubric
│  ├─ embed.py               # offline embedding + artifact builder
│  ├─ reasoning.py           # grounded, per-candidate reasoning generator
│  └─ evaluate.py            # silver ground-truth validation + ablations
└─ artifacts/                # precomputed embeddings / features (rebuilt by build_index.py)
```

---

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/Kanhaiya76618/redrob-ranker.git
cd redrob-ranker
pip install -r requirements.txt
```

Place the dataset (`candidates.jsonl` or `candidates.jsonl.gz`) and `job_description.md` in the project root.

---

## Usage

**1. Build the offline artifacts** (one-time, no time limit):

```bash
python build_index.py --candidates ./candidates.jsonl --jd ./job_description.md
```

**2. Produce the ranked submission** — this is the single command reproduced during evaluation:

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

**3. Validate locally** (there is no live leaderboard):

```bash
python src/evaluate.py --ranking ./submission.csv
```

**4. (Optional) Run the demo app:**

```bash
streamlit run app.py
```

---

## How it works — the key ideas

- **Evidence over claims.** Each candidate's text document is built to weight *career-history descriptions* (what they did) far above the *skills list* (what they typed). The gap between AI skills *claimed* and AI work *described* is the keyword-stuffer signal.
- **Adversarial integrity gate.** Deterministic checks flag logically-impossible profiles — tenure longer than time elapsed, "expert" skills used zero months, experience that a career timeline cannot support — and hard-gate them to zero. This keeps the honeypot rate near zero in the top 100.
- **Ideal & anti-profiles.** Instead of embedding the raw JD, we expand it (offline) into several synthetic ideal-candidate narratives and trap narratives. Fit = similarity to ideals − similarity to anti-profiles, which directly captures the gap between what a JD *says* and what it *means*.
- **Behavioral availability multiplier.** Recruiter response rate, last-active recency, open-to-work and interview-completion become a multiplier — a perfect-on-paper candidate who has been inactive for months is correctly demoted.
- **Grounded reasoning.** Each of the 100 rows gets a 1–2 sentence justification built from the candidate's real field values and honest concern flags — specific, varied, and never hallucinated.

---

## Compute compliance

The timed ranking step (`rank.py`) respects the evaluation constraints: ≤ 5 minutes wall-clock, ≤ 16 GB RAM, CPU only, no network, ≤ 5 GB intermediate disk. Embeddings and indexes are precomputed offline (explicitly permitted) and shipped as artifacts; `rank.py` loads them and performs arithmetic only.

---

## Results (current)

- **Honeypot integrity gate:** 60 of ~80 honeypots gated, including **all** dangerous AI-titled honeypots; projected honeypot rate in the top 100 ≈ 0%.
- **Keyword-stuffer detection:** 3,172 profiles flagged on the lexical claim-vs-evidence check.

(Updated as the pipeline progresses.)

---

## AI tools declaration

This project was developed with AI assistance (Claude) used for pair-programming, code review, design discussion, and documentation. All architecture decisions, parameter tuning, integrity-check design, and validation were directed and verified by the author. Per the hackathon's guidance, AI assistance is declared honestly; the engineering judgment is the author's own.

---

## License

Released under the [MIT License](./LICENSE). © 2026 Kanhaiya Mehta.

> Note: confirm against the official hackathon terms & conditions, which may include their own IP or licensing clause.

---

## Author

**Kanhaiya Mehta** — built for India Runs by Redrob AI × Hack2Skill.
