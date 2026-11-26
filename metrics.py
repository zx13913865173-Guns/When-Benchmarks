#!/usr/bin/env python3
"""Multi-label style metrics against expert ground truth (paper section 2.7).

Reported quantities:
    - recall / precision / F1 / coverage  (multi-label, all 140 images)
    - dominant recall / secondary recall  (ONLY the 60 asymmetric images)
    - position-1 / position-2 recall      (the 60 equal-weight images, reported
                                           separately and never pooled with the
                                           dominant/secondary metrics)
    - ablation hierarchy L0-L4
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def prf(pred: set, truth: set) -> tuple[int, int, int]:
    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)
    return tp, fp, fn


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="match_labels.py output")
    ap.add_argument("--meta", required=True, help="benchmark_metadata.csv")
    ap.add_argument("--truth", required=True, help="ground_truth.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pred = {r["image_id"]: set(filter(None, r["pred_styles"].split(";")))
            for r in csv.DictReader(open(args.pred))}
    meta = {r["image_id"]: r for r in csv.DictReader(open(args.meta))}
    truth = {r["image_id"]: set(filter(None, r["truth_styles"].split(";")))
             for r in csv.DictReader(open(args.truth))}

    TP = FP = FN = 0
    cov_hit = cov_tot = 0
    dom_hit = dom_tot = sec_hit = sec_tot = 0
    pos = defaultdict(lambda: [0, 0])  # position index -> [hit, tot]

    rows = []
    for iid, preds in pred.items():
        m = meta.get(iid)
        if m is None or iid not in truth:
            continue
        t = truth[iid]
        tp, fp, fn = prf(preds, t)
        TP += tp; FP += fp; FN += fn
        cov_hit += len(t & preds); cov_tot += len(t)

        if m["category"].startswith("asymmetric"):
            if m["dominant_style"]:
                dom_tot += 1; dom_hit += int(m["dominant_style"] in preds)
            if m["secondary_style"]:
                sec_tot += 1; sec_hit += int(m["secondary_style"] in preds)
        elif m["category"].startswith("equal_weight"):
            for idx, st in enumerate((m["style_a"], m["style_b"])):
                pos[idx][1] += 1
                pos[idx][0] += int(st in preds)

        rows.append({"image_id": iid, "recall": f"{safe_div(tp, tp+fn):.4f}",
                     "precision": f"{safe_div(tp, tp+fp):.4f}",
                     "f1": f"{safe_div(2*tp, 2*tp+fp+fn):.4f}", "coverage": f"{safe_div(len(t & preds), len(t)):.4f}"},
                    )

    recap = {
        "recall": safe_div(TP, TP + FN),
        "precision": safe_div(TP, TP + FP),
        "f1": safe_div(2 * TP, 2 * TP + FP + FN),
        "coverage": safe_div(cov_hit, cov_tot),
        "dominant_recall": safe_div(dom_hit, dom_tot),
        "secondary_recall": safe_div(sec_hit, sec_tot),
        "position1_recall": safe_div(pos[0][0], pos[0][1]),
        "position2_recall": safe_div(pos[1][0], pos[1][1]),
    }

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in recap.items():
            w.writerow([k, f"{v:.4f}"])

    print("== summary (paper reference in parentheses) ==")
    print(f"  dominant_recall : {recap['dominant_recall']:.2f} (0.76)")
    print(f"  secondary_recall: {recap['secondary_recall']:.2f} (0.22)")
    print(f"  position1/2     : {recap['position1_recall']:.2f}/{recap['position2_recall']:.2f} (0.72/0.40)")


if __name__ == "__main__":
    main()
