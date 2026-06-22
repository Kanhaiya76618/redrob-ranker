"""
reasoning.py — Grounded, varied, and honest reasoning generator for candidate fit.
"""
from __future__ import annotations

_RETR_TERMS = [
    ("learning-to-rank", "learning to rank"), ("ranking systems", "ranking"),
    ("retrieval", "retriev"), ("search relevance", "search"),
    ("recommendation systems", "recommend"), ("embeddings", "embedding"),
    ("semantic matching", "semantic"), ("vector search", "vector"),
]


def evidence_phrase(c: dict) -> str | None:
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
