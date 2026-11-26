#!/usr/bin/env python3
"""Stage 1 of the proxy overlap-risk audit: pHash candidate screening.

Approach (paper section 2.6):
    - compute a 64-bit perceptual hash (pHash) for every one of the 375 LoRA
      training images and for the reference corpus (LAION-2B-en);
    - build a SimHash-style inverted bucket index by splitting each hash into
      ``HAMMING_THRESHOLD + 1`` bands (pigeonhole principle: two hashes within
      Hamming distance ``k`` must share at least one identical band);
    - probe the index for each training image and keep every corpus entry whose
      full Hamming distance is <= ``HAMMING_THRESHOLD`` as a candidate.

Two subcommands:
    build-index   build the bucket index from a JSONL of {item_id, phash_hex}
    screen        screen the training images against a prebuilt index

This stage is intentionally *candidate* screening: Stage 2 verifies each
candidate with CLIP embeddings.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import imagehash
from PIL import Image
from tqdm import tqdm

from audit_config import HAMMING_THRESHOLD, PHASH_HASH_SIZE, STYLES


def phash_int(path: Path, hash_size: int = PHASH_HASH_SIZE) -> int:
    """Return the pHash of an image as a Python int (bit length = hash_size**2)."""
    with Image.open(path) as im:
        return int(str(imagehash.phash(im.convert("RGB"), hash_size=hash_size)), 16)


def n_bands(threshold: int = HAMMING_THRESHOLD) -> int:
    """Number of bands guaranteeing no false negatives under pigeonhole."""
    return threshold + 1


def band_slices(total_bits: int, k: int) -> list[tuple[int, int]]:
    """Split ``total_bits`` bits into ``k`` contiguous bands (last band absorbs the remainder)."""
    width = total_bits // k
    bounds = [(i * width, (i + 1) * width) for i in range(k - 1)]
    bounds.append(((k - 1) * width, total_bits))
    return bounds


def band_keys(h: int, slices) -> list[tuple[int, int]]:
    """Band index -> band value pairs for an integer hash."""
    return [(i, (h >> s) & ((1 << (e - s)) - 1)) for i, (s, e) in enumerate(slices)]


def build_index(hashes: dict[str, int], total_bits: int = 64) -> dict:
    """Build the inverted bucket index. Returns {'slices':..., 'buckets': {band: {val: [ids]}}}."""
    slices = band_slices(total_bits, n_bands())
    buckets: dict[int, dict[int, list[str]]] = {i: {} for i in range(len(slices))}
    for item_id, h in tqdm(hashes.items(), desc="indexing corpus"):
        for band, val in band_keys(h, slices):
            buckets[band].setdefault(val, []).append(item_id)
    return {"slices": slices, "buckets": buckets}


def probe(query_hash: int, index: dict, hashes: dict[str, int], threshold: int = HAMMING_THRESHOLD):
    """Return [(item_id, hamming)] candidates sharing >=1 band with the query hash."""
    slices = index["slices"]
    seen: dict[str, int] = {}
    for band, val in band_keys(query_hash, slices):
        for item_id in index["buckets"][band].get(val, []):
            if item_id in seen:
                continue
            d = bin(query_hash ^ hashes[item_id]).count("1")
            if d <= threshold:
                seen[item_id] = d
    return sorted(seen.items(), key=lambda kv: kv[1])


def cmd_build_index(args) -> None:
    hashes: dict[str, int] = {}
    with open(args.laion_index) as f:
        for line in f:
            rec = json.loads(line)
            hashes[rec["item_id"]] = int(rec["phash_hex"], 16)
    index = build_index(hashes)
    with open(args.index_out, "wb") as f:
        pickle.dump({"index": index, "hashes": hashes}, f)
    print(f"Indexed {len(hashes)} corpus items -> {args.index_out}")


def cmd_screen(args) -> None:
    with open(args.index, "rb") as f:
        payload = pickle.load(f)
    index, corpus_hashes = payload["index"], payload["hashes"]

    rows = []
    for style in STYLES:
        for img in sorted((Path(args.train_dir) / style).glob("*.jpg")):
            h = phash_int(img)
            cands = probe(h, index, corpus_hashes)
            rows.append({
                "style": style,
                "image": img.name,
                "phash_hex": f"{h:016x}",
                "n_candidates": len(cands),
                "top_candidate": cands[0][0] if cands else "",
                "top_hamming": cands[0][1] if cands else "",
                "is_candidate": int(bool(cands)),
            })
    _write_csv(args.out, rows)
    flagged = sum(r["is_candidate"] for r in rows)
    print(f"pHash candidates: {flagged}/{len(rows)} = {flagged/len(rows):.1%} "
          f"(paper reports 63/375 = 16.8%)")


def _write_csv(path, rows) -> None:
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index")
    b.add_argument("--laion_index", required=True, help="JSONL of {item_id, phash_hex}")
    b.add_argument("--index_out", required=True)
    b.set_defaults(func=cmd_build_index)

    s = sub.add_parser("screen")
    s.add_argument("--train_dir", required=True, help="dir with <style>/*.jpg")
    s.add_argument("--index", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_screen)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
