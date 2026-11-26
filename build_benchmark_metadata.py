#!/usr/bin/env python3
"""Build the 140-row benchmark manifest described in paper Table 1.

The manifest drives annotation, audit and evaluation. It is derived purely from
``fusion_config.json`` and ``seeds.json`` so it can be produced without a GPU.

Category semantics:
    asymmetric_primary          4 pairs x 2 ratios x 5 reps = 40
    asymmetric_generalization   2 pairs x 2 ratios x 5 reps = 20
    equal_weight_primary        4 pairs x 2 blocks x 5 reps = 40
    equal_weight_generalization 2 pairs x 2 blocks x 5 reps = 20
    pure_style_control          5 styles x 4 images         = 20
    total                                                     = 140

`dominant_style` / `secondary_style` are filled ONLY for asymmetric images.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "image_id", "category", "pair", "style_a", "style_b", "weight_a", "weight_b",
    "block", "rep", "seed", "dominant_style", "secondary_style", "file_name",
]


def build(config: dict, seeds_cfg: dict) -> list[dict]:
    base_seeds = seeds_cfg["base_seeds"]
    block_offset = seeds_cfg["block_seed_offset"]
    rows: list[dict] = []

    all_styles = list(config["taxonomy"]["2x2_cultural_origin_x_feature"].keys()) + \
        config["taxonomy"]["prompt_conditioned_no_culture"]

    def row(pair_id, style_a, style_b, wa, wb, block, rep, seed, category):
        dom = sec = ""
        if wa != wb:  # asymmetric only
            dom, sec = (style_a, style_b) if wa > wb else (style_b, style_a)
        fname = f"{pair_id}_{wa:.1f}-{wb:.1f}_b{block}_s{seed}.png"
        return dict(
            image_id=fname.replace(".png", ""), category=category, pair=pair_id,
            style_a=style_a, style_b=style_b, weight_a=f"{wa:.1f}", weight_b=f"{wb:.1f}",
            block=block, rep=rep, seed=seed, dominant_style=dom, secondary_style=sec,
            file_name=fname,
        )

    # asymmetric hybrids
    for pair_key, category in (("primary_pairs", "asymmetric_primary"),
                               ("generalization_pairs", "asymmetric_generalization")):
        for pair in config[pair_key]:
            for (wa, wb) in config["asymmetric_ratios"]:
                for rep, seed in enumerate(base_seeds):
                    rows.append(row(pair["id"], pair["style_a"], pair["style_b"],
                                    wa, wb, 0, rep, seed, category))

    # equal-weight hybrids, two blocks
    for pair_key, category in (("primary_pairs", "equal_weight_primary"),
                               ("generalization_pairs", "equal_weight_generalization")):
        for pair in config[pair_key]:
            for block in range(config["equal_weight"]["n_blocks"]):
                for rep, base_seed in enumerate(base_seeds):
                    seed = base_seed + block_offset * block
                    wa, wb = config["equal_weight"]["ratio"]
                    rows.append(row(pair["id"], pair["style_a"], pair["style_b"],
                                    wa, wb, block, rep, seed, category))

    # pure-style controls
    for style in all_styles:
        for rep in range(config["pure_style_controls"]["images_per_style"]):
            seed = base_seeds[rep]
            fname = f"pure_{style}_s{seed}.png"
            rows.append(dict(
                image_id=fname.replace(".png", ""), category="pure_style_control", pair="",
                style_a=style, style_b="", weight_a="1.0", weight_b="0.0", block="",
                rep=rep, seed=seed, dominant_style="", secondary_style="", file_name=fname,
            ))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="../02_hybrid_generation/fusion_config.json")
    ap.add_argument("--seeds", default="../02_hybrid_generation/seeds.json")
    ap.add_argument("--out", default="benchmark_metadata.csv")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    seeds_cfg = json.loads(Path(args.seeds).read_text())
    rows = build(config, seeds_cfg)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    n_asym = sum(1 for r in rows if r["category"].startswith("asymmetric"))
    n_eq = sum(1 for r in rows if r["category"].startswith("equal_weight"))
    n_pure = sum(1 for r in rows if r["category"] == "pure_style_control")
    print(f"total={total} asymmetric={n_asym} equal_weight={n_eq} pure={n_pure}")
    assert total == 140, total
    assert n_asym == 60, n_asym
    assert n_eq == 60, n_eq
    assert n_pure == 20, n_pure


if __name__ == "__main__":
    main()
