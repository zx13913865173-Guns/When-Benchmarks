#!/usr/bin/env python3
"""Style-level and pair-level proxy overlap-risk ratios and tertile assignment.

Definitions (paper section 2.6):

    ratio_style(style) = (n_high + n_borderline) / 75

    ratio_pair(pair)   = (ratio_style(A) + ratio_style(B)) / 2

The pair ratio is bounded in [0, 1] and is computable for every pair, including
those containing the prompt-conditioned ``minimalist`` style.

Tertiles: because no pair exceeds 30%, the 120 hybrid images are split into
low / medium / high strata of 40 images each (the 2 pairs with the highest ratios
form the high tertile, the 2 lowest the low tertile, the remaining 2 the medium).
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict

from audit_config import FUSION_PAIRS, STYLES, TRAIN_IMAGES_PER_STYLE


def style_ratios(verified: list[dict]) -> dict[str, float]:
    """(high + borderline) / 75 per style, counting one row per flagged image."""
    counts = defaultdict(Counter)
    for r in verified:
        counts[r["style"]][r["bucket"]] += 1
    ratios = {}
    for s in STYLES:
        n = counts[s]["high_similarity"] + counts[s]["borderline"]
        ratios[s] = n / TRAIN_IMAGES_PER_STYLE
    return ratios


def pair_ratios(s_ratios: dict[str, float]) -> dict[str, float]:
    return {pid: (s_ratios[a] + s_ratios[b]) / 2.0 for pid, (a, b) in FUSION_PAIRS.items()}


def assign_tertiles(pratios: dict[str, float]) -> dict[str, str]:
    ordered = sorted(pratios.items(), key=lambda kv: kv[1], reverse=True)
    tiers = {}
    for rank, (pid, _) in enumerate(ordered):
        tiers[pid] = "high" if rank < 2 else ("medium" if rank < 4 else "low")
    return tiers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verified", required=True, help="Stage 2 output CSV")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    verified = list(csv.DictReader(open(args.verified)))
    s_ratios = style_ratios(verified)
    p_ratios = pair_ratios(s_ratios)
    tiers = assign_tertiles(p_ratios)

    print("== style-level conservative proxy overlap ==")
    for s in STYLES:
        print(f"  {s:12s} {s_ratios[s]:.3f}")
    print("== pair-level ratio and tertile ==")
    rows = []
    for pid, v in sorted(p_ratios.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {pid:14s} {v:.3f}  [{tiers[pid]}]")
        rows.append({"pair": pid, "ratio": f"{v:.4f}", "tertile": tiers[pid]})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pair", "ratio", "tertile"])
        w.writeheader()
        w.writerows(rows)

    # Reference values reported in the paper (section 3.1)
    reference = {"ink_baroque": 0.227, "ink_cyber": 0.200, "ink_minimal": 0.160,
                 "cyber_baroque": 0.120, "ukiyo_baroque": 0.093, "ukiyo_minimal": 0.027}
    print("== deviation from paper (max abs) ==")
    print(" ", max(abs(p_ratios[k] - v) for k, v in reference.items()))


if __name__ == "__main__":
    main()
