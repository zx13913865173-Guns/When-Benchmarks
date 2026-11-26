"""Configuration constants for the two-stage proxy overlap-risk audit.

Mirrors paper section 2.6. All thresholds are the single fixed values used in the
paper; no threshold sweeps were performed (see paper section 4.3, Limitations).
"""
from __future__ import annotations

# --- Stage 1: perceptual hash candidate screening ---
PHASH_HASH_SIZE = 8          # imagehash.phash default -> 8x8 = 64-bit hash
HAMMING_THRESHOLD = 10       # paper: Hamming distance threshold of 10

# --- Stage 2: CLIP embedding cosine verification ---
CLIP_MODEL_NAME = "ViT-L-14/openai"   # same backbone as the default interrogator
CLIP_HIGH_SIMILARITY = 0.95           # cos >= 0.95  -> high-similarity candidate
CLIP_BORDERLINE_LOW = 0.90            # 0.90 <= cos < 0.95 -> borderline candidate
CLIP_BATCH_SIZE = 64

# --- Reference corpus ---
PROXY_CORPUS = "LAION-2B-en"          # public proxy for CLIP's non-auditable WIT-400M

# --- Training set ---
STYLES = ["ink_wash", "ukiyo_e", "cyberpunk", "baroque", "minimalist"]
TRAIN_IMAGES_PER_STYLE = 75
N_TRAIN_IMAGES = len(STYLES) * TRAIN_IMAGES_PER_STYLE   # 375

# --- Fusion pairs used for the pair-level ratio (paper section 3.1) ---
FUSION_PAIRS = {
    "ink_cyber":     ("ink_wash", "cyberpunk"),
    "ukiyo_baroque": ("ukiyo_e", "baroque"),
    "cyber_baroque": ("cyberpunk", "baroque"),
    "ink_baroque":   ("ink_wash", "baroque"),
    "ink_minimal":   ("ink_wash", "minimalist"),
    "ukiyo_minimal": ("ukiyo_e", "minimalist"),
}

# --- Tertile assignment (relative split; paper section 3.1) ---
# The two highest pair ratios form the high tertile, the two lowest the low tertile.
TERTILE_IMAGES_PER_STRATUM = 40
