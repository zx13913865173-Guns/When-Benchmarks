#!/usr/bin/env python3
"""Adjudicate expert annotations into the ground-truth label file.

Rule (paper section 2.5): a style enters the ground truth when at least half of
the four experts include it; dominant/secondary order is taken from the majority
ordering. Pure-style controls receive the single control style.

Output columns: image_id, truth_styles (semicolon-separated), dominant, secondary
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

RATER_COLS = ["expert_1", "expert_2", "expert_3", "expert_4"]


def style_set(cell: str) -> set[str]:
    cell = (cell or "").strip().lower()
    if not cell or cell == "none":
        return set()
    return {s.strip() for s in cell.split(";") if s.strip()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--majority", type=int, default=2, help="minimum expert votes")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.annotations)))
    out_rows = []
    for r in rows:
        votes = Counter()
        first_votes = Counter()
        for c in RATER_COLS:
            labels = style_set(r.get(c, ""))
            votes.update(labels)
            if labels:
                first_votes.update([sorted(labels)[0]])
        kept = sorted([s for s, v in votes.items() if v >= args.majority])
        dom = first_votes.most_common(1)[0][0] if first_votes else ""
        sec = next((s for s in kept if s != dom), "")
        out_rows.append({
            "image_id": r["image_id"],
            "truth_styles": ";".join(kept),
            "dominant": dom,
            "secondary": sec,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "truth_styles", "dominant", "secondary"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"Ground truth for {len(out_rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
