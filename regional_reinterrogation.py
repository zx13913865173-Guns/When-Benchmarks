#!/usr/bin/env python3
"""Regional (2x2 grid) re-interrogation.

Corresponds to paper section 3.6. Instead of interrogating the whole image once
(L0, global), split it into a 2x2 grid and interrogate each quadrant, then merge
the four prompts. This is the low-cost workflow modification whose effect on
secondary-style recall is reported as exploratory and partially confounded with
fusion pair.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from clip_interrogator import Config, Interrogator

CLIP_MODEL_NAME = "ViT-L-14/openai"


def quadrants(im: Image.Image) -> list[Image.Image]:
    w, h = im.size
    boxes = [(0, 0, w // 2, h // 2), (w // 2, 0, w, h // 2),
             (0, h // 2, w // 2, h), (w // 2, h // 2, w, h)]
    return [im.crop(b) for b in boxes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ci = Interrogator(Config(clip_model_name=CLIP_MODEL_NAME, quiet=True))

    rows = []
    for img in tqdm(sorted(Path(args.benchmark).glob("*.png")), desc="regional"):
        with Image.open(img) as im:
            im = im.convert("RGB")
            prompts = [ci.interrogate_classic(q) for q in quadrants(im)]
        rows.append({
            "image_id": img.stem,
            "q1": prompts[0], "q2": prompts[1], "q3": prompts[2], "q4": prompts[3],
            "merged": " , ".join(prompts),
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "q1", "q2", "q3", "q4", "merged"])
        w.writeheader()
        w.writerows(rows)
    print(f"Regional interrogation done for {len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
