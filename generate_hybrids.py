#!/usr/bin/env python3
"""Generate the 140-image hybrid benchmark with weighted multi-LoRA fusion (SDXL Base 1.0).

Corresponds to paper section 2.3 "Hybrid image generation" and Table 1.

Produced layout (paper Table 1):
    asymmetric primary hybrids           4 pairs x 2 ratios x 5 reps = 40
    asymmetric generalization hybrids    2 pairs x 2 ratios x 5 reps = 20
    equal-weight primary hybrids         4 pairs x 2 blocks x 5 reps = 40
    equal-weight generalization hybrids  2 pairs x 2 blocks x 5 reps = 20
    pure-style controls                  5 styles x 4 images          = 20
    ---------------------------------------------------------------------
    total                                                            = 140

Generation settings (paper section 2.3):
    CFG scale 7.5 | 30 inference steps | DPM++ 2M Karras | 1024 x 1024

Weighted multi-LoRA fusion uses diffusers ``set_adapters`` with per-adapter weights,
so that a pair (A, B) with ratio (0.7, 0.3) loads A at weight 0.7 and B at weight 0.3.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
CFG_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
RESOLUTION = 1024

# Short, style-defining prompts appended to every generation.
STYLE_PROMPTS = {
    "ink_wash": "Chinese ink wash painting, monochrome brush strokes, xuan paper texture",
    "ukiyo_e": "Japanese ukiyo-e woodblock print, flat color planes, bold outlines",
    "cyberpunk": "cyberpunk, neon signage, dense urban nightscape, high saturation",
    "baroque": "Baroque oil painting, chiaroscuro lighting, ornate drapery",
    "minimalist": "minimalist composition, large negative space, restrained palette",
}

NEGATIVE_PROMPT = "text, watermark, signature, jpeg artifacts, low quality, blurry"


def seed_everything(seed: int) -> None:
    """Fix all RNGs so a given seed always yields the same image."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pipeline(lora_dir: Path, adapter_names: list[str]) -> StableDiffusionXLPipeline:
    """Load SDXL Base 1.0 and register one LoRA adapter per style.

    Each adapter is registered under its style name; ``set_adapters`` below applies
    the per-pair weights at generation time.
    """
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True
    )
    for name in adapter_names:
        if name == "__base__":
            continue
        weight_path = lora_dir / name / f"{name}_sdxl.safetensors"
        pipe.load_lora_weights(str(weight_path), adapter_name=name)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    return pipe


def build_prompt(pair_styles: tuple[str, str], weights: tuple[float, float]) -> str:
    """Compose the text prompt from the two style triggers, weighted by magnitude."""
    a, b = pair_styles
    wa, wb = weights
    parts = []
    if wa > 0:
        parts.append(f"({STYLE_PROMPTS[a]}:{wa:.1f})")
    if wb > 0:
        parts.append(f"({STYLE_PROMPTS[b]}:{wb:.1f})")
    return ", ".join(parts) if parts else STYLE_PROMPTS[a]


def generate_one(pipe, prompt, seed) -> "object":
    seed_everything(seed)
    return pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=CFG_SCALE,
        width=RESOLUTION,
        height=RESOLUTION,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora_dir", default="../outputs/lora")
    ap.add_argument("--out_dir", default="../outputs/benchmark_140")
    ap.add_argument("--meta", default="../outputs/generation_metadata.csv")
    ap.add_argument("--config", default="fusion_config.json")
    ap.add_argument("--seeds", default="seeds.json")
    args = ap.parse_args()

    lora_dir = Path(args.lora_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads(Path(args.config).read_text())
    seeds_cfg = json.loads(Path(args.seeds).read_text())
    base_seeds = seeds_cfg["base_seeds"]
    block_offset = seeds_cfg["block_seed_offset"]

    all_styles = list(cfg["taxonomy"]["2x2_cultural_origin_x_feature"].keys()) + \
        cfg["taxonomy"]["prompt_conditioned_no_culture"]
    pipe = load_pipeline(lora_dir, ["__base__"] + all_styles)

    rows: list[dict] = []

    def emit(pair_id, style_a, style_b, wa, wb, block, rep, seed, category):
        """Generate, save and record one image."""
        pipe.set_adapters([style_a, style_b], adapter_weights=[wa, wb])
        prompt = build_prompt((style_a, style_b), (wa, wb))
        fname = f"{pair_id}_{wa:.1f}-{wb:.1f}_b{block}_s{seed}.png"
        img = generate_one(pipe, prompt, seed)
        img.save(out_dir / fname)
        rows.append(dict(
            image_id=fname.replace(".png", ""), category=category, pair=pair_id,
            style_a=style_a, style_b=style_b, weight_a=f"{wa:.1f}", weight_b=f"{wb:.1f}",
            block=block, rep=rep, seed=seed,
            dominant_style=(style_a if wa > wb else style_b) if wa != wb else "",
            secondary_style=(style_b if wa > wb else style_a) if wa != wb else "",
            file_name=fname, prompt=prompt,
        ))

    # 1) asymmetric hybrids (primary + generalization), ratios 0.3:0.7 and 0.7:0.3
    for kind, pair_key, category in (
        ("primary", "primary_pairs", "asymmetric_primary"),
        ("generalization", "generalization_pairs", "asymmetric_generalization"),
    ):
        for pair in cfg[pair_key]:
            for (wa, wb) in cfg["asymmetric_ratios"]:
                for rep, seed in enumerate(base_seeds):
                    emit(pair["id"], pair["style_a"], pair["style_b"], wa, wb, 0, rep, seed, category)

    # 2) equal-weight hybrids, two blocks, ratio 0.5:0.5
    for pair_key, category in (("primary_pairs", "equal_weight_primary"),
                               ("generalization_pairs", "equal_weight_generalization")):
        for pair in cfg[pair_key]:
            for block in range(cfg["equal_weight"]["n_blocks"]):
                for rep, base_seed in enumerate(base_seeds):
                    seed = base_seed + block_offset * block
                    wa, wb = cfg["equal_weight"]["ratio"]
                    emit(pair["id"], pair["style_a"], pair["style_b"], wa, wb, block, rep, seed, category)

    # 3) pure-style controls: 1.0:0.0 and 0.0:1.0 for each style (4 images per style)
    for style in all_styles:
        for rep in range(cfg["pure_style_controls"]["images_per_style"]):
            seed = base_seeds[rep]
            pipe.set_adapters([style], adapter_weights=[1.0])
            prompt = STYLE_PROMPTS[style]
            fname = f"pure_{style}_s{seed}.png"
            generate_one(pipe, prompt, seed).save(out_dir / fname)
            rows.append(dict(
                image_id=fname.replace(".png", ""), category="pure_style_control",
                pair="", style_a=style, style_b="", weight_a="1.0", weight_b="0.0",
                block="", rep=rep, seed=seed, dominant_style="", secondary_style="",
                file_name=fname, prompt=prompt,
            ))

    with open(args.meta, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} images into {out_dir}")
    expected = cfg["expected_counts"]
    assert len(rows) == expected["total"], f"expected {expected['total']}, got {len(rows)}"


if __name__ == "__main__":
    main()
