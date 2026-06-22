"""
app.py — Interactive candidate discovery and ranking showcase showroom.
"""
from __future__ import annotations

import os
import sys
import json
from datetime import date
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from parse import parse_date, candidate_text
from features import compute_features
from integrity import check_integrity
from reasoning import make_reasoning
from jd_profiles import get_profiles

# Set page config
st.set_page_config(
    page_title="Redrob Ranker — Showroom",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling matching the blueprint
st.markdown("""
<style>
    :root {
        --ink: #0a0d0c;
        --ink-2: #101513;
        --panel: #131a18;
        --panel-2: #172220;
        --line: #243330;
        --txt: #e8efe9;
        --txt-dim: #9bb0a8;
        --signal: #7dffa8;
        --signal-deep: #27c46a;
        --trap: #ff6b5e;
        --amber: #ffc24b;
        --blue: #7db4ff;
    }
    
    /* Main container styling */
    .stApp {
        background-color: var(--ink);
        color: var(--txt);
        font-family: 'Newsreader', Georgia, serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        font-family: 'Bricolage Grotesque', sans-serif;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    /* Custom Card container */
    .candidate-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s, border-color 0.2s;
    }
    .candidate-card:hover {
        transform: translateY(-2px);
        border-color: var(--signal);
    }
    
    /* Badges */
    .custom-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
        border: 1px solid var(--line);
        background: var(--panel-2);
        color: var(--txt-dim);
        display: inline-block;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    
    .badge-signal {
        color: var(--signal);
        border-color: rgba(125, 255, 168, 0.3);
        background: rgba(125, 255, 168, 0.05);
    }
    
    .badge-trap {
        color: var(--trap);
        border-color: rgba(255, 107, 94, 0.3);
        background: rgba(255, 107, 94, 0.05);
    }
    
    .badge-amber {
        color: var(--amber);
        border-color: rgba(255, 194, 75, 0.3);
        background: rgba(255, 194, 75, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# App Title & Brand
st.sidebar.markdown("<h2 style='color:#7dffa8;'>REDROB <b>RANKER</b></h2>", unsafe_allow_html=True)
st.sidebar.markdown("*Showroom / Candidate Discovery Sandbox*")
st.sidebar.divider()

# Input directories config
CANDIDATES_DEFAULT = "/Users/kanhaiya_mehta/redrob-data/candidates.jsonl"
ARTIFACTS_DIR = "/Users/kanhaiya_mehta/redrob-data/artifacts"

@st.cache_resource
def load_model_if_needed():
    """Load fastembed TextEmbedding to generate vectors on the fly for uploads."""
    from fastembed import TextEmbedding
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

@st.cache_data
def get_static_artifacts():
    """Load pre-computed profile vector ideals and anti vectors."""
    try:
        pz = np.load(os.path.join(ARTIFACTS_DIR, "profiles.npz"))
        return pz["ideal"], pz["anti"]
    except Exception as e:
        st.error(f"Failed to load profile vectors: {e}")
        return None, None

ideal_vecs, anti_vecs = get_static_artifacts()

# Sidebar Setup
st.sidebar.subheader("Configuration")
candidates_file = st.sidebar.file_uploader(
    "Upload Candidate Sample (.jsonl or .json)",
    type=["jsonl", "json"],
    help="Upload up to 100 candidate records in Redrob JSONL format. If none uploaded, default dataset will be used."
)

top_k = st.sidebar.slider("Number of Candidates to Output", 5, 100, 20)

st.sidebar.markdown("### JD Requirements Reference")
st.sidebar.markdown("- **Experience Required:** 5–9 years (Ideal: 6–8y)")
st.sidebar.markdown("- **Location Preferred:** Noida, Pune (willingness to relocate helps)")
st.sidebar.markdown("- **Disqualifier:** Consulting/Services-only career history")
st.sidebar.markdown("- **Target Skills:** Embedding, Retrieval, Semantic Search, Ranking")

# Main Content
st.markdown("<h1>Intelligent Candidate Discovery Showroom</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size:18px;color:#9bb0a8;'>Verify demonstrated evidence, detect honeypot fabrications, and evaluate availability signals.</p>", unsafe_allow_html=True)
st.divider()

# Core logic
def process_ranking(candidates_list):
    # Reference date extraction
    ref = date(2000, 1, 1)
    for c in candidates_list:
        d = parse_date((c.get("redrob_signals", {}) or {}).get("last_active_date"))
        if d and d > ref:
            ref = d
            
    # Compute features, integrity and construct texts
    ids, features_list, texts = [], [], []
    for c in candidates_list:
        cid = c["candidate_id"]
        f = compute_features(c, ref)
        ig = check_integrity(c, ref)
        ids.append(cid)
        features_list.append({
            "candidate_id": cid,
            **f,
            "impossibility_score": ig["impossibility_score"],
            "is_honeypot": ig["is_honeypot"],
            "stuffing_score": ig["stuffing_score"]
        })
        texts.append(candidate_text(c))
        
    df_feat = pd.DataFrame(features_list).set_index("candidate_id")
    
    # Semantic fit calculation (ONNX CPU embedding)
    st.info("Embedding candidates & calculating semantic fit...")
    model = load_model_if_needed()
    embs = np.asarray(list(model.embed(texts)), dtype=np.float32)
    # L2 normalize
    norms = np.linalg.norm(embs, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    embs = embs / norms
    
    sem = (embs @ ideal_vecs.T).max(1) - (embs @ anti_vecs.T).max(1)
    sem_norm = (sem - sem.min()) / (sem.max() - sem.min() + 1e-9)
    
    df_feat["semantic_fit"] = sem_norm
    honey = df_feat["is_honeypot"].to_numpy().astype(bool)
    
    # Try to load LightGBM model
    model_path = os.path.join(ARTIFACTS_DIR, "ranker_model.pkl")
    if os.path.exists(model_path):
        import pickle
        import lightgbm as lgb
        with open(model_path, "rb") as f:
            model = pickle.load(f)
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
        X = df_feat[feature_cols]
        score = model.predict(X)
    else:
        # Fallback to blended scoring
        band = df_feat["band_fit"].to_numpy()
        prod = df_feat["product_score"].to_numpy()
        loc = df_feat["location_fit"].to_numpy()
        notice = df_feat["notice_fit"].to_numpy()
        avail = df_feat["availability"].to_numpy()
        comp_pen = df_feat["company_penalty"].to_numpy()
        dom_pen = df_feat["domain_penalty"].to_numpy()
        country_pen = df_feat["country_penalty"].to_numpy()
        stuff = df_feat["stuffing_score"].to_numpy()
        
        W_SEM, W_PROD, W_BAND, W_LOC, W_NOTICE = 0.40, 0.10, 0.35, 0.05, 0.10
        core = (W_SEM * sem_norm + W_PROD * prod + W_BAND * band +
                W_LOC * loc + W_NOTICE * notice)
        penalties = comp_pen + dom_pen + country_pen - 0.10 * np.minimum(stuff, 4.0)
        score = np.clip(core + penalties, 0.0, None) * (0.5 + 0.5 * avail)
        
    score[honey] = 0.0
    
    # Sort and return
    order = np.lexsort((ids, -score))
    ranked_candidates = []
    for r, idx in enumerate(order, start=1):
        cid = ids[idx]
        cand_dict = candidates_list[idx]
        fr = df_feat.loc[cid].to_dict()
        ranked_candidates.append({
            "rank": r,
            "candidate_id": cid,
            "score": round(score[idx], 5),
            "semantic_score": round(sem_norm[idx], 3),
            "record": cand_dict,
            "features": fr,
            "reasoning": make_reasoning(cand_dict, fr)
        })
    return ranked_candidates

# Load dataset
candidates_pool = []
if candidates_file is not None:
    # Read uploaded candidates (capped at 100 for safety)
    file_contents = candidates_file.read().decode("utf-8")
    for line in file_contents.splitlines():
        line = line.strip()
        if line:
            try:
                candidates_pool.append(json.loads(line))
            except Exception:
                pass
    candidates_pool = candidates_pool[:100]
    st.sidebar.success(f"Successfully loaded {len(candidates_pool)} uploaded candidates.")
else:
    # Load top 50 sample from candidates.jsonl
    try:
        with open(CANDIDATES_DEFAULT, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 100:
                    break
                line = line.strip()
                if line:
                    candidates_pool.append(json.loads(line))
        st.sidebar.info(f"Using default sample ({len(candidates_pool)} candidates loaded).")
    except Exception as e:
        st.sidebar.error(f"Failed to load default candidates: {e}")

if candidates_pool:
    # Run the ranker
    results = process_ranking(candidates_pool)
    
    # Quick metrics summary cards
    hps = sum(1 for r in results if r["features"]["is_honeypot"])
    avg_score = np.mean([r["score"] for r in results])
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Candidates Evaluated", len(results))
    m2.metric("Average Score", f"{avg_score:.3f}")
    m3.metric("Fabricated profiles (Honeypots) gated", hps)
    
    st.divider()
    
    # Ranked List display
    st.subheader(f"Top {min(top_k, len(results))} Matches")
    
    for item in results[:top_k]:
        c = item["record"]
        p = c.get("profile", {})
        fr = item["features"]
        
        # Build candidate visual header
        yoe = p.get("years_of_experience", 0)
        title = p.get("current_title", "Software Engineer")
        company = p.get("current_company", "Product Company")
        loc = p.get("location", "India")
        
        with st.container():
            # Card styling
            st.markdown(f"""
            <div class="candidate-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                    <div>
                        <span style="font-size:20px; font-weight:bold; color:#e8efe9;">Rank {item['rank']} — {title} at {company}</span>
                        <span style="font-family: 'JetBrains Mono', monospace; font-size:12px; color:#65827a; margin-left:14px;">ID: {item['candidate_id']}</span>
                    </div>
                    <div style="font-size:24px; font-weight:800; color:#7dffa8;">{item['score']:.4f}</div>
                </div>
                <div style="margin-top: 10px; font-size:16px; color:#e8efe9; font-style:italic;">
                    "{item['reasoning']}"
                </div>
                <div style="margin-top:14px;">
                    <div class="custom-badge badge-signal">Experience: {yoe:.1f}y</div>
                    <div class="custom-badge badge-signal">Semantic Sim: {item['semantic_score']:.2f}</div>
                    <div class="custom-badge">Availability: {fr['availability']:.2f}</div>
                    <div class="custom-badge">Location Match: {fr['location_fit']:.2f} ({loc})</div>
                    <div class="custom-badge">Notice Match: {fr['notice_fit']:.2f} ({c.get('redrob_signals', {}).get('notice_period_days', 0)}d notice)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Details accordion
            with st.expander("Show detailed profile breakdown & career history"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("### Profile Summary")
                    st.markdown(p.get("summary", "No summary available."))
                    
                    st.markdown("### Stated Skills")
                    skills_list = [f"{s.get('name')} ({s.get('proficiency')})" for s in c.get("skills", [])]
                    st.write(", ".join(skills_list) if skills_list else "No skills listed.")
                    
                with col2:
                    st.markdown("### Integrity & Signal Verification")
                    st.write(f"- **Impossibility Index (Honeypot risk):** `{fr['impossibility_score']}`")
                    st.write(f"- **Keyword Stuffing Index (claims vs history):** `{fr['stuffing_score']}`")
                    
                    st.markdown("### Stated Career History Stints")
                    for role in c.get("career_history", []):
                        st.markdown(f"**{role.get('title')}** at *{role.get('company')}* ({role.get('start_date')} to {role.get('end_date') or 'Present'})")
                        st.markdown(f"<span style='font-size:14px;color:#9bb0a8;'>{role.get('description')}</span>", unsafe_allow_html=True)
                        
    # Allow download of ranked CSV
    csv_rows = []
    for r in results:
        csv_rows.append({
            "candidate_id": r["candidate_id"],
            "rank": r["rank"],
            "score": r["score"],
            "reasoning": r["reasoning"]
        })
    df_download = pd.DataFrame(csv_rows)
    csv_bytes = df_download.to_csv(index=False).encode('utf-8')
    
    st.sidebar.download_button(
        "Download Ranked CSV",
        data=csv_bytes,
        file_name="submission.csv",
        mime="text/csv"
    )
else:
    st.warning("Please upload a candidate JSONL dataset or make sure default candidates.jsonl is accessible.")
