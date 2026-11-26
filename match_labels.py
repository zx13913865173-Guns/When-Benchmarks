#!/usr/bin/env python3
"""Map generated prompts to style labels via a curated lexicon.

The interrogator emits free-form prompts; to score recall/precision we convert each
prompt into the set of styles it evokes. A style is "detected" when any of its
lexicon terms occurs in the prompt (case-insensitive, word-boundary aware).

The lexicon mirrors the style vocabulary the interrogator draws on (artists,
mediums, movements) restricted to the five styles under study.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

STYLE_LEXICON: dict[str, list[str]] = {
    "ink_wash": [
        "ink wash", "ink painting", "sumi", "sumi-e", "brush stroke", "brushstroke",
        "monochrome", "xuan paper", "rice paper", "black ink", "calligraphy",
    ],
    "ukiyo_e": [
        "ukiyo", "ukiyo-e", "ukiyoe", "woodblock", "wood block", "japanese print",
        "flat color", "flat colours", "flat colors",
    ],
    "cyberpunk": [
        "cyberpunk", "neon", "sci-fi", "scifi", "dystopian", "holographic",
        "futuristic city", "synthwave",
    ],
    "baroque": [
        "baroque", "chiaroscuro", "caravaggio", "rembrandt", "oil painting",
        "tenebrism", "ornate",
    ],
    "minimalist": [
        "minimalist", "minimalism", "minimal", "negative space", "simple composition",
        "clean lines",
    ],
}


def detect_styles(prompt: str, lexicon: dict[str, list[str]] = STYLE_LEXICON) -> list[str]:
    text = prompt.lower()
    hits = []
    for style, terms in lexicon.items():
        for t in terms:
            if re.search(r"\b" + re.escape(t) + r"\b", text):
                hits.append(style)
                break
    return hits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for r in csv.DictReader(open(args.prompts)):
        labels = detect_styles(r["prompt"])
        rows.append({"image_id": r["image_id"],
                     "pred_styles": ";".join(labels),
                     "n_pred": len(labels)})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "pred_styles", "n_pred"])
        w.writeheader()
        w.writerows(rows)
    print(f"Labelled {len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
