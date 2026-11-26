#!/usr/bin/env python3
"""Stage 2 of the proxy overlap-risk audit: CLIP-embedding verification.

For every pHash candidate from Stage 1, compute the CLIP image embedding of the
training image and of the candidate corpus image, then take the cosine similarity.

Classification (paper section 2.6):
    cos >= 0.95            -> high-similarity candidate
    0.90 <= cos < 0.95     -> borderline similarity candidate
    cos <  0.90            -> clean

The backbone is the same ViT-L-14/openai model used by the default interrogator,
so that the verification lives in the same representation space as the tool under
study.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from audit_config import (
    CLIP_BATCH_SIZE,
    CLIP_BORDERLINE_LOW,
    CLIP_HIGH_SIMILARITY,
    CLIP_MODEL_NAME,
)

# Corpus thumbnails are resolved from a local mirror keyed by item_id.
# In the paper this was a locally cached LAION-2B-en subset (see 04 README).
CORPUS_IMAGE_ROOT = Path("../data/laion2b_en_thumbs")


def load_clip(model_name: str = CLIP_MODEL_NAME):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    model.eval().to("cuda")
    return model, preprocess


@torch.no_grad()
def embed(paths: list[Path], model, preprocess) -> np.ndarray:
    """Return L2-normalised CLIP image embeddings for a list of image paths."""
    out = []
    for i in range(0, len(paths), CLIP_BATCH_SIZE):
        batch = paths[i:i + CLIP_BATCH_SIZE]
        imgs = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to("cuda")
        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 512), dtype=np.float32)


def classify(cos: float) -> str:
    if cos >= CLIP_HIGH_SIMILARITY:
        return "high_similarity"
    if cos >= CLIP_BORDERLINE_LOW:
        return "borderline"
    return "clean"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="Stage 1 output CSV")
    ap.add_argument("--train_dir", required=True, help="dir with <style>/*.jpg")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cand = [r for r in csv.DictReader(open(args.candidates)) if r["is_candidate"] == "1"]

    model, preprocess = load_clip()

    rows = []
    for r in tqdm(cand, desc="verifying candidates"):
        train_path = Path(args.train_dir) / r["style"] / r["image"]
        corr_path = CORPUS_IMAGE_ROOT / f"{r['top_candidate']}.jpg"
        if not corr_path.exists():
            continue
        v1 = embed([train_path], model, preprocess)
        v2 = embed([corr_path], model, preprocess)
        cos = float(np.dot(v1[0], v2[0]))
        rows.append({
            "style": r["style"], "image": r["image"], "top_candidate": r["top_candidate"],
            "cosine": f"{cos:.4f}", "bucket": classify(cos),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_high = sum(r["bucket"] == "high_similarity" for r in rows)
    n_bord = sum(r["bucket"] == "borderline" for r in rows)
    n_clean = sum(r["bucket"] == "clean" for r in rows)
    print(f"high={n_high} borderline={n_bord} clean={n_clean} "
          f"(paper: 32 / 13 / 18 out of 63 candidates)")


if __name__ == "__main__":
    main()
