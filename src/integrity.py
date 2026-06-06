"""
integrity.py — Adversarial integrity checks.

The dataset deliberately contains ~80 "honeypot" candidates with internally
*impossible* profiles (e.g. 8 years at a company that can't have existed that
long; "expert" in 10 skills used 0 months). The ground truth forces them to
relevance tier 0, and any submission with >10% honeypots in its top 100 is
disqualified. It ALSO contains keyword-stuffers: profiles whose skills list is
crammed with target keywords that their actual career history does not support.

A naive embedding ranker floats both straight into the top 10, because surface
keyword density correlates with JD similarity. We refuse to rank on claims we
can't corroborate. This module returns, per candidate:

  - impossibility_score : evidence of an internally contradictory profile
  - is_honeypot         : hard gate (impossibility_score >= HONEYPOT_GATE)
  - stuffing_score      : claim-vs-evidence gap (lexical version here; the
                          embedding version is added in features.py)
  - reasons             : human-readable list of what fired (for explainability)

Every check is a *logical impossibility or strong contradiction*, not mere
oddness — we never want to gate a real-but-unusual candidate.
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from parse import parse_date, months_between

HONEYPOT_GATE = 3.0  # tuned below against the real population

# AI / retrieval / ranking vocabulary used to test "does the career text back
# up the AI skills the candidate claims?"
_AI_TERMS = re.compile(
    r"machine learning|deep learning|\bml\b|\bnlp\b|embedding|retriev|ranking|"
    r"recommend|search relevance|vector|transformer|llm|fine.?tun|pytorch|"
    r"tensorflow|bert|semantic|information retrieval|learning to rank|\brag\b",
    re.I,
)
_AI_SKILL = re.compile(
    r"machine learning|deep learning|\bnlp\b|llm|embedding|pytorch|tensorflow|"
    r"transformer|computer vision|data scien|recommend|retrieval|ranking|"
    r"reinforcement|generative|bert|hugging", re.I,
)


def check_integrity(c: dict, reference: date) -> dict:
    reasons: list[str] = []
    score = 0.0

    profile = c.get("profile", {})
    yoe = float(profile.get("years_of_experience", 0) or 0)
    hist = c.get("career_history", []) or []

    # ---- 1. Per-role temporal impossibilities -----------------------------
    role_months_sum = 0
    earliest_start: Optional[date] = None
    for h in hist:
        s = parse_date(h.get("start_date"))
        e = parse_date(h.get("end_date"))
        dur = int(h.get("duration_months", 0) or 0)
        role_months_sum += dur

        if s:
            earliest_start = s if earliest_start is None else min(earliest_start, s)
            # start in the future is impossible
            if s > reference:
                score += 2.0
                reasons.append("role start date is in the future")
            # claimed tenure longer than the time that has physically elapsed
            # since the role began -> "8 yrs at a company founded 3 yrs ago"
            elapsed = months_between(s, e or reference)
            if elapsed >= 0 and dur > elapsed + 3:
                score += 2.0
                reasons.append(
                    f"claims {dur}mo tenure but only {elapsed}mo elapsed since start"
                )
        if s and e and e < s:
            score += 2.0
            reasons.append("role end date precedes start date")

    # ---- 2. Stated experience not supported by the career timeline --------
    # You cannot have 16 years of experience if your entire career history
    # begins 5 years ago. Graded by the size of the gap: a 2-3y gap is plausible
    # (gaps / partial history); a 6y+ gap is a logical impossibility.
    if earliest_start is not None:
        timeline_years = months_between(earliest_start, reference) / 12.0
        gap = yoe - timeline_years
        if gap > 2.0:
            score += 1.0 + min(max(gap - 2.0, 0.0), 11.0) * 0.30
            reasons.append(
                f"claims {yoe:.0f}y experience but career timeline spans only "
                f"~{timeline_years:.1f}y"
            )

    # sum of role durations exceeding stated experience => fabricated / heavily
    # overlapping full-time roles. Graded by the size of the excess.
    if yoe > 0:
        excess_y = (role_months_sum - yoe * 12) / 12.0
        if excess_y > 1.5:
            score += min(excess_y, 8.0) * 0.45
            reasons.append("sum of role durations exceeds stated experience")

    # ---- 3. Skill claims with zero / contradicted backing -----------------
    skills = c.get("skills", []) or []
    expert_zero = 0
    for s in skills:
        prof = (s.get("proficiency") or "").lower()
        dur = s.get("duration_months", None)
        if prof in ("expert", "advanced") and dur is not None and dur == 0:
            expert_zero += 1
    if expert_zero >= 1:
        # one is suspicious; several is a textbook honeypot
        score += min(expert_zero, 4) * 1.0
        reasons.append(f"{expert_zero} expert/advanced skill(s) used 0 months")

    # assessment scores that contradict claimed mastery
    assess = (c.get("redrob_signals", {}) or {}).get("skill_assessment_scores", {}) or {}
    contradicted = 0
    for s in skills:
        prof = (s.get("proficiency") or "").lower()
        name = s.get("name")
        if prof == "expert" and name in assess and assess[name] < 35:
            contradicted += 1
    if contradicted >= 1:
        score += contradicted * 0.75
        reasons.append(f"{contradicted} 'expert' skill(s) with assessment < 35")

    is_honeypot = score >= HONEYPOT_GATE

    # ---- 4. Keyword-stuffing (lexical version) ----------------------------
    # Claims many AI skills, but the career descriptions never mention AI work.
    ai_skill_claims = sum(1 for s in skills if _AI_SKILL.search(s.get("name", "")))
    career_text = " ".join(h.get("description", "") + " " + h.get("title", "")
                            for h in hist)
    ai_evidence = bool(_AI_TERMS.search(career_text))
    title = (profile.get("current_title") or "").lower()
    title_is_ai = bool(re.search(r"ml|ai|machine learning|data scien|nlp|research",
                                 title))
    stuffing_score = 0.0
    if ai_skill_claims >= 3 and not ai_evidence:
        stuffing_score = min(ai_skill_claims / 3.0, 3.0)
        if not title_is_ai:
            stuffing_score += 1.0

    return {
        "impossibility_score": round(score, 2),
        "is_honeypot": is_honeypot,
        "stuffing_score": round(stuffing_score, 2),
        "reasons": reasons,
    }