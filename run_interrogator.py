#!/usr/bin/env python3
"""Run CLIP Interrogator 0.6.0 (ViT-L-14/openai, classic mode) over the benchmark.

Corresponds to paper section 2.4. The dual-path tool combines a BLIP content
caption with CLIP-based style scoring; ``interrogate_classic`` selects the classic
mode used in the paper.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from clip_interrogator import Config, Interrogator

CLIP_MODEL_NAME = "ViT-L-14/openai"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", required=True, help="dir containing the 140 PNGs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ci = Interrogator(Config(clip_model_name=CLIP_MODEL_NAME, quiet=True))

    images = sorted(Path(args.benchmark).glob("*.png"))
    rows = []
    for img in tqdm(images, desc="interrogating"):
        with Image.open(img) as im:
            prompt = ci.interrogate_classic(im.convert("RGB"))
        rows.append({"image_id": img.stem, "prompt": prompt})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "prompt"])
        w.writeheader()
        w.writerows(rows)
    print(f"Interrogated {len(rows)} images -> {args.out}")


if __name__ == "__main__":
    main()
