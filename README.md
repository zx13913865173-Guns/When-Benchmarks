[README.md](https://github.com/user-attachments/files/35439637/README.md)

# When Benchmarks Overlap — Reproducibility Repository

This repository provides the complete reproduction materials for the paper:
**"When Benchmarks Overlap: Proxy Overlap Risk and Its Limited Role in Cross-Cultural Style Decomposition Using CLIP Interrogator"**

It contains the **end-to-end pipeline** including LoRA training, hybrid image generation, 140-image benchmark construction, proxy overlap-risk auditing, CLIP Interrogator evaluation, statistical testing, and figure generation. The repository structure corresponds one-to-one with the paper's sections, allowing for the reproduction of all numerical results reported in the manuscript.

---

## 1. Repository Structure

| Directory | Contents | Corresponding Section in Paper |
|---|---|---|
| `00_environment/` | Software/hardware environment and dependency versions | Methods |
| `01_lora_training/` | Training configurations and scripts for 5 style LoRAs (kohya-ss/sd-scripts) | §2.2 LoRA training on 375 images |
| `02_hybrid_generation/` | SDXL Base 1.0 weighted multi-LoRA fusion scripts and seed lists | §2.3 Hybrid image generation |
| `03_benchmark/` | 140-image benchmark organization and metadata | §2.3, Table 1 |
| `04_proxy_overlap_audit/` | Proxy overlap-risk audit: pHash screening + CLIP embedding verification | §2.6, §3.1 |
| `05_interrogator_eval/` | CLIP Interrogator application, regional re-interrogation, metric calculation | §2.4, §2.7, §3.4–3.7 |
| `06_statistics/` | R scripts: permutation test, mixed-effects models, FDR correction | §2.8, §3.2–3.5 |
| `07_annotation/` | Expert annotation templates, Fleiss κ calculation, label adjudication | §2.5 |
| `08_figures/` | Plotting scripts and captions for Fig. 2–7 | Results |

---

## 2. Reproduction Pipeline

Strictly follow the sequence below, as the output of each step serves as the input for the next:

1. **Environment Setup** → `00_environment/`, install dependencies (`requirements.txt`).
2. **LoRA Training** → `01_lora_training/`, train one LoRA adapter for each of the 5 styles (5 total).
3. **Hybrid Image Generation** → `02_hybrid_generation/`, generate 120 hybrid images (6 fusion pairs × weight ratios × 5 seeds) + 20 pure-style controls = 140 images total.
4. **Benchmark Metadata Construction** → `03_benchmark/`, organize the 140 images and generation metadata.
5. **Expert Annotation & Adjudication** → `07_annotation/`, obtain ground-truth labels and calculate Fleiss κ.
6. **Proxy Overlap Audit** → `04_proxy_overlap_audit/`, perform the two-stage audit on 375 LoRA training images to obtain style-level and pair-level ratios and tertile stratification.
7. **CLIP Interrogator Evaluation** → `05_interrogator_eval/`, compute recall / precision / F1 / coverage, dominant/secondary metrics, and ablations.
8. **Statistical Analysis** → `06_statistics/`, permutation tests, mixed-effects models, and FDR correction.
9. **Figure Generation** → `08_figures/`, generate Fig. 2–7.

---

## 3. Key Hyperparameters

| Parameter | Value |
|---|---|
| Base Model | SDXL Base 1.0 (`stabilityai/stable-diffusion-xl-base-1.0`) |
| LoRA rank / alpha | 32 / 1.0 |
| Training steps / LR / batch size | 3000 / 1e-4 / 2 |
| Optimizer | AdamW |
| Training set size | 375 images (75 / style), public-domain or CC0 |
| Generation resolution / CFG / steps / sampler | 1024×1024 / 7.5 / 30 / DPM++ 2M Karras |
| Weight ratios | 0.3:0.7, 0.5:0.5, 0.7:0.3 (0.5:0.5 repeated twice) |
| Repetitions / Seeds | 5 repetitions / seeds 42–46 |
| CLIP Interrogator | v0.6.0, ViT-L-14/openai, classic mode |
| pHash Hamming distance threshold | ≤ 10 |
| CLIP cosine threshold | ≥0.95 high-similarity; 0.90–<0.95 borderline; <0.90 clean |
| Multiple comparison correction | Benjamini–Hochberg FDR |

---

## 4. Data Acquisition & Ethics

- **Style Training Images (375)**: Sourced from `WikiArt`, `The Met Open Access`, and `Rijksmuseum`. Only public-domain or CC0 licensed images are used. See `01_lora_training/prepare_dataset.md`.
- **Proxy Corpus**: `LAION-2B-en`, obtained via `img2dataset` and indexed with pHash buckets. See `04_proxy_overlap_audit/README` (scripts contain comments on scale limitations).
- **Generated 140 Images**: Produced by scripts in `02_hybrid_generation/`; metadata is included in the repository.

> **Ethics Statement**: Generated images are used solely for academic reproduction. Training images are in the public domain or CC0. Expert annotations were obtained anonymously and voluntarily without compensation, containing no personally identifiable information.

---

## 5. Environment Setup

```bash
# Python dependencies
pip install -r requirements.txt

# R dependencies
Rscript -e 'install.packages(c("lme4", "lmerTest", "emmeans", "ggplot2", "dplyr", "tidyr"))'
