"""
features.py — Deterministic, interpretable signals for the Tier-2 reranker.

All cheap arithmetic over structured fields. Two kinds of output:
  - scored features in roughly [-1, 1] that feed `core_fit`
  - an `availability` multiplier in [0, 1]: is this person actually reachable?

Every feature maps to an explicit line in the JD, so each number is defensible
in the Stage-5 interview.
"""
from __future__ import annotations

from datetime import date
import re

from parse import parse_date, months_between

_AI_TITLE = re.compile(r"\b(?:ml|ai|machine learning|data scien|nlp|research|recommend|search|applied|retrieval|rank)", re.I)


# --- company / domain knowledge -------------------------------------------
# JD explicitly does NOT want careers spent entirely at services/consulting.
SERVICES_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "tech mahindra", "hcl", "hcltech", "mindtree", "ltimindtree",
    "l&t infotech", "mphasis", "hexaware", "genpact", "dxc", "deloitte",
    "kpmg", "ernst", "pwc", "birlasoft", "nagarro", "cybage",
}
SERVICES_INDUSTRIES = {"it services", "consulting", "staffing", "outsourcing"}
PRODUCT_INDUSTRIES = {
    "software", "saas", "fintech", "e-commerce", "ai/ml", "edtech",
    "food delivery", "gaming", "healthtech", "developer tools", "cybersecurity",
}

# JD: Pune/Noida preferred; Hyderabad/Mumbai/Delhi-NCR/Bangalore welcome.
LOC_TOP = ("pune", "noida")
LOC_STRONG = ("hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
              "ghaziabad", "faridabad", "bengaluru", "bangalore")

# JD: CV/speech/robotics without NLP/IR would be "re-learning fundamentals".
CV_SPEECH_ROBOTICS = ("computer vision", "image processing", "speech", "asr",
                      "robotic", "lidar", "slam", "autonomous driving")
NLP_IR = ("nlp", "natural language", "retriev", "ranking", "search", "embedding",
          "recommend", "information retrieval", "language model", "llm", "bert")
RESEARCH_HINTS = ("university", "institute", "iit ", "phd", "postdoc",
                  "research lab", "laboratory", "academia")


def _is_services(company: str, industry: str) -> bool:
    c, i = (company or "").lower(), (industry or "").lower()
    return any(f in c for f in SERVICES_FIRMS) or any(s in i for s in SERVICES_INDUSTRIES)


def _is_product(industry: str) -> bool:
    return any(p in (industry or "").lower() for p in PRODUCT_INDUSTRIES)


def band_fit(yoe: float | None) -> float:
    """JD band is 5-9, ideal 6-8; flexible outside but tapering."""
    if not yoe:
        return 0.0
    if 6 <= yoe <= 8:
        return 1.0
    d = (6 - yoe) if yoe < 6 else (yoe - 8)
    return max(0.0, 1.0 - 0.18 * d)        # 5/9 -> .82, 4/10 -> .64


def company_signal(c: dict) -> dict:
    roles = c.get("career_history", []) or []
    n = len(roles)
    serv = sum(_is_services(r.get("company"), r.get("industry")) for r in roles)
    prod = sum(_is_product(r.get("industry")) for r in roles)
    consulting_only = n > 0 and serv == n           # JD explicit do-not-want
    cur_services = bool(roles) and _is_services(roles[0].get("company"),
                                                roles[0].get("industry"))
    product_score = prod / n if n else 0.0
    penalty = 0.0
    if consulting_only:
        penalty -= 0.60
    elif cur_services:
        penalty -= 0.10
    return {"product_score": round(product_score, 3),
            "consulting_only": consulting_only,
            "company_penalty": round(penalty, 3)}


def domain_penalty(c: dict) -> float:
    p = c.get("profile", {})
    text = " ".join([p.get("summary", ""), p.get("headline", "")] +
                    [r.get("description", "") + " " + r.get("title", "")
                     for r in c.get("career_history", []) or []]).lower()
    has_cv = any(t in text for t in CV_SPEECH_ROBOTICS)
    has_nlp = any(t in text for t in NLP_IR)
    has_research = any(t in text for t in RESEARCH_HINTS)
    has_product = any(_is_product(r.get("industry"))
                      for r in c.get("career_history", []) or [])
    pen = 0.0
    if has_cv and not has_nlp:          # CV/speech/robotics without NLP/IR
        pen -= 0.50
    if has_research and not has_product:  # pure research, no production
        pen -= 0.45
        
    # Non-AI title without NLP/IR keywords -> not a fit
    title = (p.get("current_title") or "").lower()
    is_ai = bool(_AI_TITLE.search(title))
    if not is_ai and not has_nlp:
        pen -= 0.50
        
    return round(pen, 3)


def location_fit(c: dict) -> float:
    p = c.get("profile", {})
    loc = (p.get("location") or "").lower()
    india = (p.get("country") or "").lower() in ("india", "in", "")
    relocate = bool((c.get("redrob_signals", {}) or {}).get("willing_to_relocate"))
    if not india:
        return 0.30 if relocate else 0.25          # JD: no visa sponsorship
    if any(t in loc for t in LOC_TOP):
        return 1.0
    if any(t in loc for t in LOC_STRONG):
        return 0.85
    return 0.80 if relocate else 0.55              # other India


def notice_fit(c: dict) -> float:
    nd = (c.get("redrob_signals", {}) or {}).get("notice_period_days")
    if nd is None:
        return 0.6
    if nd <= 30:                                    # JD can buy out <=30 days
        return 1.0
    if nd <= 60:
        return 0.70
    if nd <= 90:
        return 0.50
    return max(0.20, 1.0 - nd / 180.0)


def availability(c: dict, reference: date) -> float:
    """
    Behavioral multiplier [0.1, 1.0]. JD: a perfect-on-paper candidate who
    hasn't logged in for months with a 5% response rate is not actually
    available -- down-weight them.
    """
    s = c.get("redrob_signals", {}) or {}
    la = parse_date(s.get("last_active_date"))
    rec = max(0.0, 1.0 - months_between(la, reference) / 9.0) if la else 0.4
    resp = s.get("recruiter_response_rate")
    resp = 0.5 if resp is None else float(resp)
    icr = s.get("interview_completion_rate")
    icr = 0.7 if icr is None else float(icr)
    otw = 1.0 if s.get("open_to_work_flag") else 0.70
    score = 0.35 * rec + 0.30 * resp + 0.20 * icr + 0.15 * otw
    return round(min(1.0, max(0.1, score)), 3)


def country_penalty(c: dict) -> float:
    p = c.get("profile", {})
    country = (p.get("country") or "").lower()
    if country not in ("india", "in", ""):
        return -0.20
    return 0.0


def compute_features(c: dict, reference: date) -> dict:
    """One flat, interpretable feature dict per candidate."""
    p = c.get("profile", {})
    cs = company_signal(c)
    return {
        "band_fit": band_fit(p.get("years_of_experience")),
        "product_score": cs["product_score"],
        "consulting_only": cs["consulting_only"],
        "company_penalty": cs["company_penalty"],
        "domain_penalty": domain_penalty(c),
        "location_fit": location_fit(c),
        "notice_fit": notice_fit(c),
        "availability": availability(c, reference),
        "country_penalty": country_penalty(c),
    }


if __name__ == "__main__":
    import json
    ref = date(2026, 5, 27)
    for c in json.load(open("../sample_candidates.json"))[:8]:
        f = compute_features(c, ref)
        pr = c["profile"]
        print(f"{pr['current_title'][:26]:28} {pr['years_of_experience']:>4}y  "
              f"band={f['band_fit']:.2f} prod={f['product_score']:.2f} "
              f"loc={f['location_fit']:.2f} avail={f['availability']:.2f} "
              f"pen={f['company_penalty']+f['domain_penalty']:.2f}")