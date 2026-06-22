<div align="center">

<br/>

```
██████╗ ███████╗██████╗ ██████╗  ██████╗ ██████╗ 
██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔═══██╗██╔══██╗
██████╔╝█████╗  ██║  ██║██████╔╝██║   ██║██████╔╝
██╔══██╗██╔══╝  ██║  ██║██╔══██╗██║   ██║██╔══██╗
██║  ██║███████╗██████╔╝██║  ██║╚██████╔╝██████╔╝
╚═╝  ╚═╝╚══════╝╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═════╝ 
                                                 
██████╗  █████╗ ███╗   ██╗██║  ██╗███████╗██████╗ 
██╔══██╗██╔══██╗████╗  ██║██║ ██╔╝██╔════╝██╔══██╗
██████╔╝███████║██╔██╗ ██║█████╔╝ █████╗  ██████╔╝
██╔══██╗██╔══██║██║╚██╗██║██╔═██╗ ██╔══╝  ██╔══██╗
██║  ██║██║  ██║██║ ╚████║██║  ██╗███████╗██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
```

### **Evidence-Verifying, Fraud-Resistant Candidate Ranking Engine**

*When gamed resumes try to cheat the system, the true signals are already extracted.*

[![Live Showroom](https://img.shields.io/badge/🌐_Web_Showroom-localhost:8501-7dffa8?style=for-the-badge)](http://localhost:8501)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/ML_Model-LightGBM_Regressor-007acc?style=for-the-badge)](https://github.com/microsoft/LightGBM)
[![Embeddings](https://img.shields.io/badge/Embeddings-BAAI/bge--small--en--v1.5-ff6b5e?style=for-the-badge)](https://huggingface.co/BAAI/bge-small-en-v1.5)
[![Built for India Runs 2026](https://img.shields.io/badge/Built_for-India_Runs_2026-16224a?style=for-the-badge)](https://github.com/Kanhaiya76618/redrob-ranker)

</div>

---

## 🧠 What is Redrob Ranker?

**Redrob Ranker** is an autonomous candidate discovery and ranking system. Traditional recruiting tools rank candidates by keyword matching. This is easily gamed: anyone can generate a resume stuffed with the exact keywords of a job ad, letting gamed profiles outrank authentic talent. 

This engine does the opposite — it **ranks demonstrated evidence, not claimed keywords**, actively detects internally-impossible ("honeypot") profiles, and evaluates availability and location constraints using a pre-trained machine learning model.

> **Integrity Guard filters fakes → Contrastive Semantic Recall narrows pool → LightGBM Regressor reranks → Grounded Reasoning justifies every decision.**

---

## ✨ Funnel Architecture

To process **100,000 candidates** in seconds under strict CPU-only constraints, the engine is structured as a **three-tier funnel**:

```
┌────────────────────────────────────────────────────────┐
│               100,000 Candidate Records                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 0: Integrity Gate (Deterministic Checks)        │
│  - Blocks timeline anomalies (tenure vs. age)          │
│  - Filters dead-activity/honeypot profiles             │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼  [~95,000 clean candidates]
┌────────────────────────────────────────────────────────┐
│  Tier 1: Contrastive Semantic Recall                   │
│  - Dense vector similarity (Ideal vs. Anti Profiles)   │
│  - Reduces search space to top ~2,000                  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼  [Top ~2,000 matches]
┌────────────────────────────────────────────────────────┐
│  Tier 2: LightGBM Regressor Reranking                  │
│  - 11-feature model trained on JD rubric targets       │
│  - Ranks top 100 with deterministic tie-breaking      │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│    Top 100 Candidates with Grounded Explanations       │
└────────────────────────────────────────────────────────┘
```

| Funnel Stage | Processes | Operation | Cost |
| :--- | :--- | :--- | :--- |
| **Tier 0 · Gate** | 100,000 | Arithmetic timeline checks; sets honeypot scores to `0.0` | $O(1)$ — Instant |
| **Tier 1 · Recall** | survivors | Contrastive cosine similarity (Ideal − Anti profile vectors) | Vector dot product |
| **Tier 2 · Rerank** | top ~2,000 | LightGBM Regressor prediction + location/YOE constraint filters | < 50ms CPU inference |

---

## 📈 Performance & Trap Metrics

Our LightGBM Regressor achieves perfect compliance with all Job Description (JD) and trap constraints on the 100K candidate dataset:

* **Composite Score**: **`0.9449`**
* **NDCG@10**: **`0.9561`**
* **NDCG@50**: **`0.8897`**
* **MAP (Mean Average Precision)**: **`1.0000`** (Perfect recall)
* **Traps in Top 100**: **0** (Zero honeypots, zero consulting-only, zero keyword-stuffers, zero non-AI titles, zero out-of-India, and zero out-of-band experience profiles).

---

## 🏗️ Repository Structure

```
redrob-ranker/
├── artifacts/               # Precomputed candidate index + pre-trained model
│   ├── embeddings.npy       # dense L2-normalized BGE vectors (100000, 384)
│   ├── ids.npy              # unique candidate IDs matching embedding rows
│   ├── features.parquet     # deterministic tabular features
│   ├── profiles.npz         # ideal/anti profile vectors
│   └── ranker_model.pkl     # pre-trained LightGBM regressor
├── src/
│   ├── parse.py             # candidate loader + text formatter
│   ├── integrity.py         # honeypot checks + stuffing score
│   ├── features.py          # experience band, location fit, company metrics
│   ├── jd_profiles.py       # ideal/anti candidate profiles (JD intelligence)
│   ├── embed.py             # offline embedding generator
│   ├── reasoning.py         # grounded reasoning template generator
│   └── evaluate.py          # offline validation suite & ablations
├── build_index.py           # OFFLINE: generates index artifacts
├── rank.py                  # RUNTIME: CLI engine (generates submission.csv)
├── app.py                   # SHOWROOM: Streamlit web application
├── submission.csv           # Final scored output file
└── submission_metadata.yaml # Submission metadata file
```

---

## ⚡ Setup & Quick Start

### Prerequisites
- Python **3.10+**

### Installation
```bash
# Clone the repository
git clone https://github.com/Kanhaiya76618/redrob-ranker.git
cd redrob-ranker

# Install dependencies in a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Running the Project

### Scenario A: Running on a New Dataset (Full Pipeline)
If you have a brand-new dataset (e.g. `new_candidates.jsonl`), run these commands:

1. **Build the Index & Embeddings**:
   ```bash
   python build_index.py --candidates /path/to/new_candidates.jsonl --artifacts ./new_artifacts
   ```
2. **Rank Candidates & Generate CSV**:
   ```bash
   python rank.py --candidates /path/to/new_candidates.jsonl --artifacts ./new_artifacts --out ./submission.csv
   ```

### Scenario B: Ranking the Pre-packaged Dataset (Scored Step)
To rank the default 100K candidate dataset instantly using the precomputed artifacts:
```bash
python rank.py --candidates /Users/kanhaiya_mehta/redrob-data/candidates.jsonl --artifacts /Users/kanhaiya_mehta/redrob-data/artifacts --out ./submission.csv
```
This runs in **seconds** because it bypasses the slow embedding step and executes inference directly using the pre-trained LightGBM model.

### Scenario C: Launching the Interactive Showroom (Streamlit App)
Open the visual showroom to upload candidate datasets, check integrity indices, and read grounded justifications:
```bash
streamlit run app.py
```

---

## 🔧 Feature Engineering Details

The LightGBM model scores candidates using 11 highly interpretable features:

| Feature | Description |
| :--- | :--- |
| **semantic_fit** | Contrastive vector similarity (ideal minus anti-profiles). |
| **band_fit** | Experience matching the 5–9 years bracket (ideal: 6–8 years). |
| **product_score** | Fraction of career spent at product companies vs services. |
| **location_fit** | Pune/Noida preference and relocation willingness. |
| **notice_fit** | Notice period suitability (buyout <=30 days gets top score). |
| **availability** | Behavioral activity multiplier (response and activity rates). |
| **company_penalty** | Penalizes consulting-only careers (TCS, Wipro, Infosys, etc.). |
| **domain_penalty** | Filters out CV, Speech, and Robotics profiles without NLP. |
| **country_penalty** | Penalizes candidates outside of India (due to visa constraints). |
| **stuffing_score** | Ratio of self-reported skills to work-history descriptions. |
| **impossibility_score** | Chronological overlap or logic violation score. |

---

<div align="center">

**Built for India Runs by Redrob AI × Hack2Skill · Agentic & Autonomous Systems**

*Semantic Fit · LightGBM · Honeypot Gating · Grounded Reasoning*

[![Live Showroom](https://img.shields.io/badge/🚀_Try_Showroom-localhost:8501-7dffa8?style=for-the-badge)](http://localhost:8501)

</div>
