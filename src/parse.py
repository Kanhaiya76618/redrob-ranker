"""
parse.py — Load and lightly normalize Redrob candidate records.

The dataset is a JSONL (one candidate per line). We keep parsing thin and
loss-less here: downstream modules (integrity, features, embedding) decide how
to interpret fields. The only normalization we do is robust date parsing.
"""
from __future__ import annotations

import gzip
import json
from datetime import date
from typing import Iterator, Optional


def _open(path: str):
    """Open plain .jsonl or gzipped .jsonl.gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_candidates(path: str) -> Iterator[dict]:
    """Yield candidate dicts one at a time (memory-friendly for 100K rows)."""
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_date(s: Optional[str]) -> Optional[date]:
    """Parse 'YYYY-MM-DD' (and tolerate 'YYYY-MM' / 'YYYY'). None on failure."""
    if not s or not isinstance(s, str):
        return None
    parts = s.strip().split("-")
    try:
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1)
        if len(parts) == 1 and parts[0]:
            return date(int(parts[0]), 1, 1)
    except (ValueError, TypeError):
        return None
    return None


def months_between(d1: date, d2: date) -> int:
    """Whole months from d1 to d2 (can be negative if d2 < d1)."""
    return (d2.year - d1.year) * 12 + (d2.month - d1.month)


def candidate_text(c: dict) -> str:
    """
    Build a single text document for a candidate, deliberately weighting what
    they DID (career-history descriptions) over what they merely LIST (skills).
    Used later for embedding. We repeat the headline/summary and career text so
    the document is dominated by demonstrated evidence, not keyword lists.
    """
    p = c.get("profile", {})
    parts = [p.get("headline", ""), p.get("summary", "")]
    for h in c.get("career_history", []):
        parts.append(f"{h.get('title','')} at {h.get('company','')} "
                     f"({h.get('industry','')}). {h.get('description','')}")
    return "  ".join(x for x in parts if x).strip()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data_candidates.jsonl"
    n = 0
    for c in load_candidates(path):
        n += 1
    print(f"Loaded {n} candidates from {path}")