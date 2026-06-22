"""
build_index.py — OFFLINE entry point. Distills candidates.jsonl into artifacts.
Usage:
  python build_index.py --candidates ./candidates.jsonl --jd ./job_description.md
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import embed


def main():
    ap = argparse.ArgumentParser(description="Build candidate database index artifacts.")
    ap.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    ap.add_argument("--jd", help="Path to job_description.md (optional, profiles are compiled in src/jd_profiles.py)")
    ap.add_argument("--artifacts", default="artifacts", help="Output directory for precomputed index artifacts")
    ap.add_argument("--backend", choices=["fastembed", "st"], default="fastembed", help="Embedding backend")
    ap.add_argument("--batch", type=int, default=256, help="Batch size for embedding model")
    args = ap.parse_args()

    # Align arguments to what src/embed.py expects
    sys.argv = [
        "src/embed.py",
        "--candidates", args.candidates,
        "--artifacts", args.artifacts,
        "--backend", args.backend,
        "--batch", str(args.batch)
    ]

    print("Running index builder pipeline...")
    embed.main()


if __name__ == "__main__":
    main()
