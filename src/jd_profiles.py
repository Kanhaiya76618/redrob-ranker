"""
jd_profiles.py — JD intelligence: the scored rubric + ideal / anti profiles.

Two ideas live here:

1. RUBRIC — the JD's requirements distilled into weights. The deterministic
   parts (band, location, notice, consulting penalty, domain) are implemented
   in features.py; this dict documents the weighting and is the single place to
   tune it.

2. IDEAL / ANTI PROFILES — instead of embedding the raw job post (which floats
   keyword-stuffers and honeypots), we author several natural-language
   narratives of the *perfect* candidate and several of the *trap* candidates.
   Embedded once, offline. A candidate's semantic fit =
        max similarity to IDEAL profiles  -  max similarity to ANTI profiles
   This directly encodes the gap between what a profile says and what it proves.

The profiles are written the way real candidates describe themselves (summary +
career story), NOT as keyword lists, so they land near genuine talent in vector
space rather than near keyword-stuffers. They are static text => reproducible
with no network and no live model at any stage.
"""

# --- scored rubric (documentation + single source of weights) --------------
RUBRIC = {
    "must_have": {
        "embeddings_retrieval_in_production": 1.0,
        "vector_db_or_hybrid_search_ops": 1.0,
        "ranking_evaluation_frameworks": 0.9,   # NDCG / MRR / MAP, A/B testing
        "strong_python": 0.6,
    },
    "nice_to_have": {
        "llm_finetuning_lora_qlora": 0.4,
        "learning_to_rank": 0.4,
        "hr_tech_or_marketplace": 0.3,
        "distributed_systems_scale": 0.3,
        "open_source_contributions": 0.3,
    },
    "weights": {            # how the reranker blends the major signals
        "semantic_fit": 0.45,   # ideal - anti profile similarity
        "evidence": 0.20,       # built ranking/search/rec at a product company
        "band_fit": 0.12,
        "location_fit": 0.10,
        "notice_fit": 0.05,
        "nice_to_have": 0.08,
    },
    "multipliers": ["availability"],            # behavioral, in [0,1]
    "hard_penalties": [                          # JD "do NOT want" / disqualifiers
        "consulting_only", "research_only_no_production",
        "cv_speech_robotics_without_nlp", "keyword_stuffer",
    ],
}

# --- IDEAL candidate narratives -------------------------------------------
IDEAL_PROFILES = [
    # 1. core profile: production retrieval + ranking at a product company
    "Senior ML engineer with seven years of experience, four of them building "
    "production search and ranking systems at consumer product companies. Owned "
    "the embeddings-based retrieval stack end to end, handled embedding drift, "
    "index refresh and retrieval-quality regressions in production, and ran a "
    "hybrid dense-plus-lexical search service for real users. Defined offline "
    "evaluation with NDCG and MRR and correlated it to online A/B results. Strong "
    "Python; ships fast and iterates on real user metrics.",

    # 2. shipper-leaning, scrappy
    "Applied AI engineer who shipped a first recommendation ranker in a week, then "
    "improved it with embeddings and LLM-based re-ranking once real usage showed "
    "where to invest. Operated a vector database (Qdrant and FAISS) in production, "
    "managed reindexing and latency, and set up A/B testing and recruiter-style "
    "feedback loops. Comfortable owning the full intelligence layer of a product "
    "and mentoring a small team.",

    # 3. pre-LLM IR veteran (JD prizes this explicitly)
    "Search relevance and information-retrieval engineer with eight years of "
    "experience who worked on retrieval and learning-to-rank long before LLMs "
    "became fashionable. Built XGBoost and neural learning-to-rank models, tuned "
    "BM25 and hybrid retrieval at scale, and migrated the stack to dense "
    "embeddings and transformer re-rankers. Deep opinions on offline-to-online "
    "evaluation, defended with systems actually built and deployed.",

    # 4. recommendation / matching at a marketplace or fintech product company
    "Six years building candidate- and product-matching and recommendation "
    "systems at a marketplace product company. Designed the semantic matching and "
    "ranking pipeline that decides what users see, handled retrieval quality and "
    "index operations, and evaluated ranking rigorously with MAP and NDCG. "
    "Open-source contributions in the retrieval space and strong production Python.",
]

# --- ANTI profiles (the traps the JD explicitly rejects) -------------------
ANTI_PROFILES = [
    # keyword-stuffer: buzzwords as skills, no shipped systems / wrong role
    "Lists machine learning, deep learning, RAG, embeddings, vector databases, "
    "LLM fine-tuning and prompt engineering as skills, but the actual work "
    "experience is in marketing, content or general management with no shipped "
    "machine-learning systems and no production retrieval or ranking work.",

    # consulting lifer (JD: only ever at services firms = not a fit)
    "Entire career spent at large IT-services and consulting firms delivering "
    "client projects and managed services. Worked across many short engagements, "
    "no ownership of a product used by real end users, and no production machine "
    "learning, retrieval or ranking system built in-house.",

    # research-only academic (JD: pure research without production = disqualifier)
    "Academic researcher with a PhD and conference papers from university labs, "
    "focused on theory and benchmarks. No production deployment, no shipping to "
    "real users, no experience operating systems in production.",

    # CV / speech / robotics without NLP/IR (JD: would relearn fundamentals)
    "Computer vision and robotics specialist working on image processing, object "
    "detection, autonomous systems and sensor fusion. No natural-language, "
    "information-retrieval, search or ranking experience.",

    # framework-enthusiast / title-chaser (JD: not what we need)
    "Recent projects consist of LangChain tutorials and demos that call OpenAI "
    "APIs, with blog posts about the latest hot framework. Less than a year of "
    "such work, frequent job changes every eighteen months chasing senior titles, "
    "and no underlying retrieval, ranking or systems experience.",
]


def get_profiles() -> dict:
    """Return the ideal and anti narratives for offline embedding."""
    return {"ideal": IDEAL_PROFILES, "anti": ANTI_PROFILES}


if __name__ == "__main__":
    p = get_profiles()
    print(f"{len(p['ideal'])} ideal profiles, {len(p['anti'])} anti profiles")
    print(f"rubric weights: {RUBRIC['weights']}")