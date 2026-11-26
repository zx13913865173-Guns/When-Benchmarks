#!/usr/bin/env python3
"""Fleiss' kappa for agreement among the four art experts (paper section 2.5).

The paper reports Fleiss kappa = 0.81 for the four raters. This script computes
the overall kappa on the per-image style-category ratings, plus a per-category
breakdown, from an annotation file with columns expert_1..expert_4.

Note: Fleiss' kappa assumes a fixed set of nominal categories. We treat each
mutually exclusive "primary style" judgement as one category; for multi-label
annotations we score the dominant (first-listed) style.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

RATER_COLS = ["expert_1", "expert_2", "expert_3", "expert_4"]


def fleiss_kappa(counts: list[list[int]]) -> float:
    """Fleiss' kappa from an N x k matrix of per-category rating counts."""
    n_subj = len(counts)
    n_raters = sum(counts[0])
    k = len(counts[0])

    p_j = [sum(row[j] for row in counts) / (n_subj * n_raters) for j in range(k)]
    P_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1)) for row in counts]
    P_bar = sum(P_i) / n_subj
    P_e = sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e)


def primary_label(cell: str) -> str:
    cell = (cell or "").strip().lower()
    if not cell or cell == "none":
        return "none"
    return cell.split(";")[0].strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.annotations)))

    # Build the N x k count matrix over the observed categories.
    categories = sorted({primary_label(r[c]) for r in rows for c in RATER_COLS})
    cat_index = {c: i for i, c in enumerate(categories)}
    counts = []
    for r in rows:
        row = [0] * len(categories)
        for c in RATER_COLS:
            row[cat_index[primary_label(r[c])]] += 1
        counts.append(row)

    kappa = fleiss_kappa(counts)
    print(f"Fleiss kappa = {kappa:.3f}  (paper reports 0.81)")

    # Per-subject agreement, for inspecting low-agreement images.
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "n_categories_used", "agreement_P_i"])
        for r, row in zip(rows, counts):
            n = sum(row)
            P_i = (sum(c * c for c in row) - n) / (n * (n - 1)) if n > 1 else 0.0
            w.writerow([r["image_id"], sum(1 for v in row if v), f"{P_i:.4f}"])
    print(f"Per-image agreement -> {args.out}")


if __name__ == "__main__":
    main()
