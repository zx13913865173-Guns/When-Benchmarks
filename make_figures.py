#!/usr/bin/env python3
"""Reproduce Figures 2-7 of the paper from the reported values.

All figures use English labels so they render identically regardless of the
CJK fonts available on the host.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- reported values
AUDIT = {"pHash candidates": 16.8, "high-similarity": 8.5, "borderline": 3.5,
         "combined": 12.0}
RECALL = {"dominant": 0.76, "secondary": 0.22}
ABLATION = {"L0": 0.53, "L1": 0.87, "L2": 0.30, "L3": 0.69, "L4": 0.03, "oracle": 1.00}
TERTILE_GAP = {"low": 0.10, "medium": 0.12, "high": 0.18}
TOOLS = {"CLIP Interrogator": (0.53, 0.22, 0.08),
         "WD14 tagger": (0.61, 0.34, 0.11),
         "DeepDanbooru": (0.57, 0.29, 0.14)}
REGIONAL = {"precision": (0.85, 0.80), "recall": (0.28, 0.49),
            "f1": (0.42, 0.61), "coverage": (0.61, 0.78)}


def fig2(out):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(AUDIT.keys()), list(AUDIT.values()), color="#4C72B0")
    ax.set_ylabel("% of training images / candidates")
    ax.set_title("Fig. 2  Proxy overlap audit (pHash screening)")
    ax.axhline(8.5, ls="--", c="grey", lw=1)
    fig.tight_layout(); fig.savefig(out / "fig2_audit.png", dpi=300); plt.close(fig)


def fig3(out):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(list(RECALL.keys()), list(RECALL.values()), color=["#55A868", "#C44E52"])
    ax.set_ylim(0, 1); ax.set_ylabel("Recall")
    ax.set_title("Fig. 3  Dominant vs secondary recall (n=60)")
    for i, v in enumerate(RECALL.values()):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout(); fig.savefig(out / "fig3_dom_sec.png", dpi=300); plt.close(fig)


def fig4(out):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(ABLATION.keys()), list(ABLATION.values()), color="#8172B3")
    ax.set_ylim(0, 1.1); ax.set_ylabel("Recall")
    ax.set_title("Fig. 4  Ablation hierarchy L0-L4 and oracle")
    fig.tight_layout(); fig.savefig(out / "fig4_ablation.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(REGIONAL.keys())
    x = range(len(labels))
    glob = [REGIONAL[k][0] for k in labels]
    reg = [REGIONAL[k][1] for k in labels]
    ax.bar([i - 0.2 for i in x], glob, width=0.4, label="global (L0)")
    ax.bar([i + 0.2 for i in x], reg, width=0.4, label="regional (L3)")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1); ax.set_title("Fig. 4b  Global vs regional trade-off (low-overlap)")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig4b_tradeoff.png", dpi=300); plt.close(fig)


def fig5(out):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(list(TERTILE_GAP.keys()), list(TERTILE_GAP.values()), color="#CCB974")
    ax.set_ylabel("East-West gap")
    ax.set_title("Fig. 5  East-West gap by overlap tertile")
    fig.tight_layout(); fig.savefig(out / "fig5_tertile.png", dpi=300); plt.close(fig)


def fig6(out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = list(TOOLS.keys())
    overall = [TOOLS[n][0] for n in names]
    sec = [TOOLS[n][1] for n in names]
    gap = [TOOLS[n][2] for n in names]
    x = range(len(names))
    ax.bar([i - 0.25 for i in x], overall, width=0.25, label="overall recall")
    ax.bar([i for i in x], sec, width=0.25, label="secondary recall")
    ax.bar([i + 0.25 for i in x], gap, width=0.25, label="East-West gap")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.set_title("Fig. 6  Cross-tool comparison")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "fig6_cross_tool.png", dpi=300); plt.close(fig)


def fig7(out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(list(AUDIT.keys()), list(AUDIT.values()), color="#4C72B0")
    axes[0].set_title("(a) Proxy overlap audit")
    axes[0].tick_params(axis="x", rotation=20)

    labels = ["dom_recall", "sec_recall", "gap_low", "gap_high", "reg_sec"]
    vals = [0.76, 0.22, 0.10, 0.18, 0.49]
    axes[1].bar(labels, vals, color="#C44E52")
    axes[1].set_title("(b) Style decomposition outcomes")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("Fig. 7  Descriptive summary")
    fig.tight_layout(); fig.savefig(out / "fig7_summary.png", dpi=300); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", default="../outputs/figures")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    fig2(out); fig3(out); fig4(out); fig5(out); fig6(out); fig7(out)
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
